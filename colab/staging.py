"""Colab-only upfront Drive-to-local SSD model staging with SHA-256 verification (v1.1.0)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

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

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB chunks
MANIFEST_FILENAME = ".nexus-stage-manifest.json"
SKIPPED_STAGE_DIRS = frozenset({".cache", "__pycache__", ".git", ".ipynb_checkpoints"})


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when local staging SSD lacks required space + safety margin."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


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
        if has_diffusers_components and len(rel.parts) == 1 and p.name.endswith(".safetensors") and p.stat().st_size > 2 * 1024 * 1024 * 1024:
            # Redundant monolithic single-file checkpoint in root; diffusers components take precedence
            continue
        files.append(p)
    return files


def stage_model_directory(
    source_dir: Path,
    local_stage_root: Path,
    *,
    model_id: str,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
    on_status: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Stage a single Drive model directory onto local Colab SSD with full SHA-256 verification.
    Skips `.cache/` lock/metadata clutter and fails loud on insufficient disk space.
    """
    log = on_status or (lambda _msg: None)
    dest_dir = local_stage_root / model_id
    files = _iter_stage_files(source_dir)
    total_bytes = sum(p.stat().st_size for p in files)
    total_files = len(files)

    existing_manifest = _load_stage_manifest(dest_dir)
    if existing_manifest is not None and dest_dir.is_dir():
        manifest_files = existing_manifest.get("files") or {}
        cache_valid = len(manifest_files) == total_files
        if cache_valid:
            log(f"[STAGE] Verifying cached local copy for {model_id} ({total_files} files)...")
            for src_file in files:
                rel = src_file.relative_to(source_dir).as_posix()
                dst_file = dest_dir / rel
                entry = manifest_files.get(rel)
                if not isinstance(entry, dict) or not dst_file.is_file():
                    cache_valid = False
                    break
                src_st = src_file.stat()
                dst_st = dst_file.stat()
                if dst_st.st_size != src_st.st_size or entry.get("size") != src_st.st_size:
                    cache_valid = False
                    break
                expected_hash = entry.get("sha256")
                if not isinstance(expected_hash, str) or not expected_hash:
                    cache_valid = False
                    break
                if _sha256_file(src_file) != expected_hash or _sha256_file(dst_file) != expected_hash:
                    cache_valid = False
                    break
        if cache_valid:
            log(f"[STAGE] Reusing verified local SSD cache for {model_id}.")
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

    (dest_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    manifest_entries: dict[str, Any] = {}
    log(f"[STAGE] Staging {model_id} ({total_bytes / (1024 ** 3):.2f} GiB across {total_files} files)...")

    for idx, src_file in enumerate(files, start=1):
        rel_path = src_file.relative_to(source_dir)
        rel_key = rel_path.as_posix()
        dst_file = dest_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        src_st = src_file.stat()

        if dst_file.is_file() and dst_file.stat().st_size == src_st.st_size:
            src_hash = _sha256_file(src_file)
            dst_hash = _sha256_file(dst_file)
            if src_hash == dst_hash:
                copied += src_st.st_size
                manifest_entries[rel_key] = {
                    "size": int(src_st.st_size),
                    "source_mtime_ns": int(src_st.st_mtime_ns),
                    "sha256": src_hash,
                }
                continue

        tmp_file = dst_file.with_name(f".{dst_file.name}.staging.tmp")
        h = hashlib.sha256()
        try:
            with src_file.open("rb") as f_in, tmp_file.open("wb") as f_out:
                while True:
                    buf = f_in.read(CHUNK_SIZE)
                    if not buf:
                        break
                    f_out.write(buf)
                    h.update(buf)
                    copied += len(buf)
                f_out.flush()
                os.fsync(f_out.fileno())
            os.replace(tmp_file, dst_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)
            raise

        manifest_entries[rel_key] = {
            "size": int(src_st.st_size),
            "source_mtime_ns": int(src_st.st_mtime_ns),
            "sha256": h.hexdigest(),
        }
        pct = min(100.0, round(copied / max(1, total_bytes) * 100.0, 1))
        log(f"[STAGE] {model_id}: {idx}/{total_files} ({pct:.1f}%) — {src_file.name}")

    _write_stage_manifest(
        dest_dir,
        {
            "version": 3,
            "model_id": model_id,
            "source_dir": str(source_dir),
            "status": "complete",
            "total_bytes": total_bytes,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "files": manifest_entries,
        },
    )
    log(f"[STAGE] Completed and verified {model_id} -> {dest_dir}")
    return dest_dir


def stage_selected_models(
    *,
    local_stage_root: Path,
    language_candidate: Any,
    image_candidate: Any = None,
    manifest_path: Optional[Path] = None,
    drive_manifest_copy: Optional[Path] = None,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
    on_status: Optional[Callable[[str], None]] = None,
) -> SessionRuntimeConfig:
    """
    Stage the single selected language model and optional single image model to local SSD,
    then write the locked `session-runtime.json` manifest consumed by Nexus Gateway.
    """
    staged_lang_dir = stage_model_directory(
        Path(language_candidate.source_path),
        local_stage_root,
        model_id=language_candidate.id,
        min_free_margin_bytes=min_free_margin_bytes,
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
        staged_img_dir = stage_model_directory(
            Path(image_candidate.source_path),
            local_stage_root,
            model_id=image_candidate.id,
            min_free_margin_bytes=min_free_margin_bytes,
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
        version="1.1.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        language_model=lang_spec,
        image_model=img_spec,
    )
    out_manifest = manifest_path or (local_stage_root / "session-runtime.json")
    save_session_runtime_config(out_manifest, session_cfg)
    if drive_manifest_copy is not None:
        save_session_runtime_config(drive_manifest_copy, session_cfg)
    return session_cfg
