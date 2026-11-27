"""Colab-only Drive-to-local SSD fast model staging with tiered verification (v1.2.1)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GW_SRC = _REPO_ROOT / "nexus-gateway" / "src"
if str(_GW_SRC) not in sys.path:
    sys.path.insert(0, str(_GW_SRC))

from models.base import (  # noqa: E402
    SelectedImageSpec,
    SelectedLanguageSpec,
    SessionRuntimeConfig,
    save_session_runtime_config,
)

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MiB streaming chunks
SAMPLE_SLICE_BYTES = 4 * 1024 * 1024  # 4 MiB head + 4 MiB mid + 4 MiB tail sampled fingerprint
DEFAULT_MAX_WORKERS = 4
MANIFEST_FILENAME = ".nexus-stage-manifest.json"
MANIFEST_VERSION = 4
SKIPPED_STAGE_DIRS = frozenset({".cache", "__pycache__", ".git", ".ipynb_checkpoints"})

VerifyMode = Literal["fast", "local_full", "strict_source"]
TransportMode = Literal["auto", "drive_api", "fuse"]


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when local staging SSD lacks required space + safety margin."""


def _make_fast_hasher(preferred: Optional[str] = None) -> tuple[str, Any]:
    """
    Create the fastest available streaming hasher:
    xxh3_128 (30-60 GB/s) -> blake3 (3-15 GB/s) -> stdlib blake2b (~1 GB/s).
    """
    if preferred in (None, "xxh3_128"):
        try:
            import xxhash  # type: ignore[import-not-found]

            return "xxh3_128", xxhash.xxh3_128()
        except ImportError:
            if preferred == "xxh3_128":
                raise
    if preferred in (None, "blake3"):
        try:
            import blake3  # type: ignore[import-not-found]

            return "blake3", blake3.blake3()
        except ImportError:
            if preferred == "blake3":
                raise
    if preferred == "sha256":
        return "sha256", hashlib.sha256()
    if preferred == "md5":
        return "md5", hashlib.md5()
    return "blake2b", hashlib.blake2b(digest_size=32)


def _hash_file_full(path: Path, *, algo: Optional[str] = None) -> tuple[str, str]:
    """Compute a full streaming digest of `path` using the selected fast hash algorithm."""
    used_algo, h = _make_fast_hasher(algo)
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
    return used_algo, h.hexdigest()


def _sample_hash_file(path: Path, *, algo: Optional[str] = None) -> tuple[str, str]:
    """
    Compute a fast sampled fingerprint over head (4 MiB) + middle (4 MiB) + tail (4 MiB)
    plus exact file size. Runs in <15ms on local NVMe even for 20+ GiB GGUFs.
    """
    used_algo, h = _make_fast_hasher(algo)
    size = path.stat().st_size
    h.update(f"size:{size}:".encode("ascii"))
    with path.open("rb") as fh:
        if size <= SAMPLE_SLICE_BYTES * 3:
            while True:
                buf = fh.read(CHUNK_SIZE)
                if not buf:
                    break
                h.update(buf)
        else:
            # Head
            fh.seek(0)
            h.update(fh.read(SAMPLE_SLICE_BYTES))
            # Middle
            mid = max(0, (size // 2) - (SAMPLE_SLICE_BYTES // 2))
            fh.seek(mid)
            h.update(fh.read(SAMPLE_SLICE_BYTES))
            # Tail
            fh.seek(max(0, size - SAMPLE_SLICE_BYTES))
            h.update(fh.read(SAMPLE_SLICE_BYTES))
    return used_algo, h.hexdigest()


def _load_stage_manifest(dest_dir: Path) -> Optional[dict[str, Any]]:
    manifest_path = dest_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("status") == "complete" else None
    except Exception:
        return None


def _write_stage_manifest(dest_dir: Path, payload: dict[str, Any]) -> None:
    manifest_path = dest_dir / MANIFEST_FILENAME
    tmp_path = dest_dir / f"{MANIFEST_FILENAME}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def _iter_stage_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    # If this is a diffusers pipeline directory with complete components,
    # skip staging redundant monolithic single-file checkpoints in root (e.g. *.safetensors > 2GB in root)
    has_diffusers_components = (
        (source_dir / "model_index.json").is_file()
        and (source_dir / "unet").is_dir()
        and (source_dir / "vae").is_dir()
    )

    for p in sorted(source_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(source_dir)
        if p.name == MANIFEST_FILENAME or any(seg in SKIPPED_STAGE_DIRS for seg in rel.parts):
            continue
        if (
            has_diffusers_components
            and len(rel.parts) == 1
            and p.name.endswith(".safetensors")
            and p.stat().st_size > 2 * 1024 * 1024 * 1024
        ):
            # Redundant monolithic single-file checkpoint in root; diffusers components take precedence
            continue
        files.append(p)
    return files


def _verify_cached_file(
    *,
    src_file: Path,
    dst_file: Path,
    entry: dict[str, Any],
    verify_mode: VerifyMode,
) -> bool:
    """Verify one cached local SSD file against its manifest entry under `verify_mode`."""
    if not dst_file.is_file():
        return False
    src_st = src_file.stat()
    dst_st = dst_file.stat()
    if dst_st.st_size != src_st.st_size or entry.get("size") != src_st.st_size:
        return False

    # Check source mtime match in fast/local_full modes (zero FUSE data read).
    recorded_mtime = entry.get("source_mtime_ns")
    if verify_mode != "strict_source":
        if recorded_mtime is not None and int(recorded_mtime) != int(src_st.st_mtime_ns):
            return False

    if verify_mode == "fast":
        sample_algo = entry.get("sample_algo")
        sample_hash = entry.get("sample_hash")
        if isinstance(sample_algo, str) and isinstance(sample_hash, str) and sample_hash:
            try:
                _, actual = _sample_hash_file(dst_file, algo=sample_algo)
                return actual == sample_hash
            except Exception:
                return False
        # Legacy v3 manifest without sample_hash: upgrade seamlessly by trusting matching size+mtime on local SSD
        # only if sha256 entry exists; otherwise re-verify.
        return isinstance(entry.get("sha256"), str) and bool(entry.get("sha256")) and recorded_mtime is not None

    # local_full or strict_source
    hash_algo = entry.get("hash_algo") or ("sha256" if entry.get("sha256") else None)
    expected_hash = entry.get("hash") or entry.get("sha256")
    if not isinstance(hash_algo, str) or not isinstance(expected_hash, str) or not expected_hash:
        return False
    try:
        _, dst_hash = _hash_file_full(dst_file, algo=hash_algo)
        if dst_hash != expected_hash:
            return False
        if verify_mode == "strict_source":
            _, src_hash = _hash_file_full(src_file, algo=hash_algo)
            return src_hash == expected_hash
        return True
    except Exception:
        return False


def _copy_file_fuse(
    *,
    src_file: Path,
    dst_file: Path,
    on_bytes: Callable[[int], None],
) -> dict[str, Any]:
    """Stream-copy one file from FUSE to local SSD with inline fast hashing and atomic rename."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    src_st = src_file.stat()
    tmp_file = dst_file.with_name(f".{dst_file.name}.staging.tmp")
    hash_algo, h = _make_fast_hasher()
    written = 0
    try:
        with src_file.open("rb") as f_in, tmp_file.open("wb") as f_out:
            while True:
                buf = f_in.read(CHUNK_SIZE)
                if not buf:
                    break
                f_out.write(buf)
                h.update(buf)
                written += len(buf)
                on_bytes(len(buf))
        if written != src_st.st_size:
            raise IOError(f"Short copy for {src_file.name}: expected {src_st.st_size}, wrote {written}")
        os.replace(tmp_file, dst_file)
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    sample_algo, sample_hash = _sample_hash_file(dst_file, algo=hash_algo)
    return {
        "size": int(src_st.st_size),
        "source_mtime_ns": int(src_st.st_mtime_ns),
        "hash_algo": hash_algo,
        "hash": h.hexdigest(),
        "sample_algo": sample_algo,
        "sample_hash": sample_hash,
    }


def stage_model_directory(
    source_dir: Path,
    local_stage_root: Path,
    *,
    model_id: str,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
    verify_mode: VerifyMode = "fast",
    transport: TransportMode = "auto",
    max_workers: int = DEFAULT_MAX_WORKERS,
    on_status: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Stage a single Drive model directory onto local Colab SSD with tiered verification.
    - `verify_mode="fast"` (default): zero-FUSE-read warm cache validation (stat + 12 MiB sampled NVMe check).
    - `transport="auto"` (default): uses Drive API v3 parallel Range downloads + server md5Checksum when
      available in Colab, falling back cleanly to 4-worker parallel FUSE (16 MiB chunks).
    """
    log = on_status or (lambda _msg: None)
    dest_dir = local_stage_root / model_id
    files = _iter_stage_files(source_dir)
    total_bytes = sum(p.stat().st_size for p in files)
    total_files = len(files)

    t0 = time.monotonic()
    existing_manifest = _load_stage_manifest(dest_dir)
    if existing_manifest is not None and dest_dir.is_dir():
        manifest_files = existing_manifest.get("files") or {}
        cache_valid = len(manifest_files) == total_files
        if cache_valid:
            log(f"[STAGE] Verifying cached local copy for {model_id} ({total_files} files, mode={verify_mode})...")
            for src_file in files:
                rel = src_file.relative_to(source_dir).as_posix()
                entry = manifest_files.get(rel)
                if not isinstance(entry, dict) or not _verify_cached_file(
                    src_file=src_file,
                    dst_file=dest_dir / rel,
                    entry=entry,
                    verify_mode=verify_mode,
                ):
                    cache_valid = False
                    break
        if cache_valid:
            dt = time.monotonic() - t0
            log(f"[STAGE] Reusing verified local SSD cache for {model_id} ({dt:.2f}s).")
            return dest_dir

    local_stage_root.mkdir(parents=True, exist_ok=True)
    already_matching = sum(
        (dest_dir / f.relative_to(source_dir)).stat().st_size
        for f in files
        if (dest_dir / f.relative_to(source_dir)).is_file()
        and (dest_dir / f.relative_to(source_dir)).stat().st_size == f.stat().st_size
    )
    remaining_needed = max(0, total_bytes - already_matching)
    free_bytes = shutil.disk_usage(local_stage_root).free
    if free_bytes < remaining_needed + min_free_margin_bytes:
        raise InsufficientDiskSpaceError(
            f"Insufficient local disk for {model_id}: need {remaining_needed + min_free_margin_bytes} B "
            f"(remaining={remaining_needed} B + margin={min_free_margin_bytes} B), only {free_bytes} B free"
        )

    # Keep old manifest entries in memory so unchanged files can be reused on partial resume.
    prior_entries: dict[str, Any] = (existing_manifest.get("files") or {}) if existing_manifest else {}
    (dest_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Optionally initialize Drive API v3 transport
    drive_token: Optional[str] = None
    drive_remote_map: dict[str, Any] = {}
    active_transport = "fuse"
    if transport in ("auto", "drive_api"):
        try:
            import drive_fast_transfer  # type: ignore[import-not-found]

            drive_token, drive_remote_map = drive_fast_transfer.list_drive_model_files(source_dir)
            if drive_remote_map:
                active_transport = "drive_api"
        except Exception as exc:
            if transport == "drive_api":
                raise
            log(f"[STAGE] Drive API v3 unavailable ({exc}); using parallel FUSE ({max_workers} workers).")

    log(
        f"[STAGE] Staging {model_id} ({total_bytes / (1024 ** 3):.2f} GiB across {total_files} files, "
        f"transport={active_transport}, verify={verify_mode})..."
    )

    progress_lock = threading.Lock()
    copied_bytes = 0
    last_log_t = time.monotonic()

    def _on_bytes(n: int) -> None:
        nonlocal copied_bytes, last_log_t
        with progress_lock:
            copied_bytes += n
            now = time.monotonic()
            if now - last_log_t >= 4.0 and total_bytes > 0:
                elapsed = max(0.1, now - t0)
                mbps = (copied_bytes / (1024 * 1024)) / elapsed
                pct = min(100.0, (copied_bytes / total_bytes) * 100.0)
                log(
                    f"[STAGE] {model_id}: {copied_bytes / (1024 ** 3):.2f} / {total_bytes / (1024 ** 3):.2f} GiB "
                    f"({pct:.1f}%) — {mbps:.1f} MB/s"
                )
                last_log_t = now

    def _stage_one(src_file: Path) -> tuple[str, dict[str, Any]]:
        rel_path = src_file.relative_to(source_dir)
        rel_key = rel_path.as_posix()
        dst_file = dest_dir / rel_path
        src_st = src_file.stat()

        prior = prior_entries.get(rel_key)
        if isinstance(prior, dict) and _verify_cached_file(
            src_file=src_file,
            dst_file=dst_file,
            entry=prior,
            verify_mode=verify_mode,
        ):
            _on_bytes(int(src_st.st_size))
            return rel_key, prior

        if active_transport == "drive_api" and drive_token and rel_key in drive_remote_map:
            try:
                import drive_fast_transfer  # type: ignore[import-not-found]

                remote = drive_remote_map[rel_key]
                verified_md5 = drive_fast_transfer.download_file_via_drive_api(
                    remote=remote,
                    dest_file=dst_file,
                    token=drive_token,
                    on_bytes=_on_bytes,
                )
                sample_algo, sample_hash = _sample_hash_file(dst_file)
                return rel_key, {
                    "size": int(src_st.st_size),
                    "source_mtime_ns": int(src_st.st_mtime_ns),
                    "hash_algo": "md5" if verified_md5 else sample_algo,
                    "hash": verified_md5 or sample_hash,
                    "sample_algo": sample_algo,
                    "sample_hash": sample_hash,
                }
            except Exception as exc:
                if transport == "drive_api":
                    raise
                log(f"[STAGE] Drive API fallback to FUSE for {rel_key}: {exc}")

        entry = _copy_file_fuse(src_file=src_file, dst_file=dst_file, on_bytes=_on_bytes)
        return rel_key, entry

    # Stage large files first so workers stay balanced, capped to `max_workers` concurrent files.
    ordered_files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    worker_count = max(1, min(max_workers, len(ordered_files)))
    manifest_entries: dict[str, Any] = {}

    if worker_count == 1:
        for src_file in ordered_files:
            rel_key, entry = _stage_one(src_file)
            manifest_entries[rel_key] = entry
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futs = {pool.submit(_stage_one, src_file): src_file for src_file in ordered_files}
            for fut in as_completed(futs):
                rel_key, entry = fut.result()
                manifest_entries[rel_key] = entry

    _write_stage_manifest(
        dest_dir,
        {
            "version": MANIFEST_VERSION,
            "model_id": model_id,
            "source_dir": str(source_dir),
            "transport": active_transport,
            "verify_mode": verify_mode,
            "status": "complete",
            "total_bytes": total_bytes,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "files": manifest_entries,
        },
    )
    elapsed = max(0.01, time.monotonic() - t0)
    avg_mbps = (total_bytes / (1024 * 1024)) / elapsed
    log(
        f"[STAGE] Completed and verified {model_id} -> {dest_dir} "
        f"({total_bytes / (1024 ** 3):.2f} GiB in {elapsed:.1f}s, avg {avg_mbps:.1f} MB/s)"
    )
    return dest_dir


def stage_selected_models(
    *,
    local_stage_root: Path,
    language_candidate: Any,
    image_candidate: Any = None,
    manifest_path: Optional[Path] = None,
    drive_manifest_copy: Optional[Path] = None,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
    verify_mode: VerifyMode = "fast",
    transport: TransportMode = "auto",
    defer_image_staging: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
) -> SessionRuntimeConfig:
    """
    Stage the single selected language model and optional single image model to local SSD,
    then write the locked `session-runtime.json` manifest consumed by Nexus Gateway.
    When `defer_image_staging=True`, writes the expected staged image path immediately so
    the Gateway can warm-load the language model right away while the image model stages
    in a background thread.
    """
    staged_lang_dir = stage_model_directory(
        Path(language_candidate.source_path),
        local_stage_root,
        model_id=language_candidate.id,
        min_free_margin_bytes=min_free_margin_bytes,
        verify_mode=verify_mode,
        transport=transport,
        on_status=on_status,
    )
    lang_spec = SelectedLanguageSpec(
        id=language_candidate.id,
        display_name=language_candidate.id,
        model_path=str(staged_lang_dir),
        implementation=language_candidate.implementation,
        architecture=language_candidate.architecture,
        modalities=list(language_candidate.modalities),
        context_window=int(language_candidate.context_window),
        max_output_tokens=int(language_candidate.max_output_tokens),
        reasoning_mode=language_candidate.reasoning_mode,
        reasoning_efforts=language_candidate.reasoning_efforts,
        default_reasoning_effort=language_candidate.default_reasoning_effort,
        min_vram_gib=float(language_candidate.min_vram_gib),
        size_bytes=int(language_candidate.size_bytes),
    )

    img_spec: Optional[SelectedImageSpec] = None
    if image_candidate is not None:
        if defer_image_staging:
            staged_img_dir = local_stage_root / image_candidate.id
        else:
            staged_img_dir = stage_model_directory(
                Path(image_candidate.source_path),
                local_stage_root,
                model_id=image_candidate.id,
                min_free_margin_bytes=min_free_margin_bytes,
                verify_mode=verify_mode,
                transport=transport,
                on_status=on_status,
            )
        img_spec = SelectedImageSpec(
            id=image_candidate.id,
            display_name=image_candidate.id,
            model_path=str(staged_img_dir),
            implementation=image_candidate.implementation,
            architecture=image_candidate.architecture,
            default_steps=int(image_candidate.default_steps),
            min_vram_gib=float(image_candidate.min_vram_gib),
            size_bytes=int(image_candidate.size_bytes),
        )

    session_cfg = SessionRuntimeConfig(
        version="1.2.1",
        created_at=datetime.now(timezone.utc).isoformat(),
        language_model=lang_spec,
        image_model=img_spec,
    )
    out_manifest = manifest_path or (local_stage_root / "session-runtime.json")
    save_session_runtime_config(out_manifest, session_cfg)
    if drive_manifest_copy is not None:
        save_session_runtime_config(drive_manifest_copy, session_cfg)
    return session_cfg
