"""Manifest-backed Drive-to-local model staging with full SHA-256 verification and live progress."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from services.progress import ProgressService

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB chunks
MANIFEST_FILENAME = ".nexus-stage-manifest.json"


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when local staging disk lacks required space + safety margin."""


def _sha256_file(path: Path, on_chunk: Optional[Callable[[int], None]] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
            if on_chunk is not None:
                on_chunk(len(buf))
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


def stage_model_directory(
    source_dir: Path,
    local_cache_root: Path,
    *,
    model_id: str,
    progress: ProgressService,
    operation_id: Optional[str] = None,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
    allow_drive_fallback: bool = False,
) -> Path:
    """
    Stage a Drive model directory onto Colab local disk with full SHA-256 integrity
    verification on ALL files (including large weights), live byte/file progress
    during hashing and copying, and no silent Drive fallback by default.
    """
    dest_dir = local_cache_root / model_id
    op = (
        progress.start(
            "model_staging",
            model_id,
            phase="staging",
            source="google_drive",
            destination="local_disk",
            message=f"Checking staged cache for {model_id}",
        )
        if operation_id is None
        else None
    )
    op_id = operation_id or (op.operation_id if op else "")

    files = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.name != MANIFEST_FILENAME)
    total_bytes = sum(p.stat().st_size for p in files)
    total_files = len(files)

    # Full SHA-256 cache validation (detects same-size, preserved-mtime source changes).
    existing_manifest = _load_stage_manifest(dest_dir)
    if existing_manifest is not None and dest_dir.is_dir():
        manifest_files = existing_manifest.get("files") or {}
        cache_valid = len(manifest_files) == total_files
        verified_bytes = 0
        if cache_valid:
            for idx, src_file in enumerate(files, start=1):
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

                def _on_verify_chunk(n: int, fname: str = src_file.name, i: int = idx) -> None:
                    nonlocal verified_bytes
                    verified_bytes += n
                    pct = min(100.0, round(verified_bytes / max(1, total_bytes) * 100.0, 2))
                    progress.update(
                        op_id,
                        phase="checkpoint_validation",
                        bytes_copied=verified_bytes,
                        total_bytes=total_bytes,
                        progress=pct,
                        message=f"Verifying {fname} ({i}/{total_files})",
                    )

                src_hash = _sha256_file(src_file, on_chunk=_on_verify_chunk)
                dst_hash = _sha256_file(dst_file)
                if src_hash != expected_hash or dst_hash != expected_hash:
                    cache_valid = False
                    break

        if cache_valid:
            progress.update(
                op_id,
                phase="checkpoint_validation",
                bytes_copied=total_bytes,
                total_bytes=total_bytes,
                progress=100.0,
                message=f"Using verified local staged cache for {model_id}",
            )
            return dest_dir

    local_cache_root.mkdir(parents=True, exist_ok=True)
    # Check free disk space with safety margin before writing.
    already_matching_bytes = 0
    for src_file in files:
        dst_file = dest_dir / src_file.relative_to(source_dir)
        if dst_file.is_file() and dst_file.stat().st_size == src_file.stat().st_size:
            already_matching_bytes += dst_file.stat().st_size
    remaining_needed = max(0, total_bytes - already_matching_bytes)

    free_bytes = shutil.disk_usage(local_cache_root).free
    if free_bytes < remaining_needed + min_free_margin_bytes:
        msg = (
            f"Insufficient local disk for {model_id}: need {remaining_needed + min_free_margin_bytes} B "
            f"(remaining={remaining_needed} B + margin={min_free_margin_bytes} B), only {free_bytes} B free"
        )
        if not allow_drive_fallback:
            progress.fail(op_id, msg)
            raise InsufficientDiskSpaceError(msg)
        progress.update(op_id, phase="checkpoint_validation", bytes_copied=0, total_bytes=total_bytes, message=msg)
        return source_dir

    (dest_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    manifest_entries: dict[str, Any] = {}
    progress.update(op_id, phase="staging", bytes_copied=0, total_bytes=total_bytes, message=f"Staging {model_id}")

    for idx, src_file in enumerate(files, start=1):
        rel_path = src_file.relative_to(source_dir)
        rel_key = rel_path.as_posix()
        dst_file = dest_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        src_st = src_file.stat()

        # Safe resume: reuse local file ONLY when full SHA-256 matches source.
        if dst_file.is_file() and dst_file.stat().st_size == src_st.st_size:
            progress.update(
                op_id,
                phase="checkpoint_validation",
                bytes_copied=copied,
                total_bytes=total_bytes,
                message=f"Hashing existing {src_file.name} ({idx}/{total_files})",
            )
            src_hash = _sha256_file(src_file)
            dst_hash = _sha256_file(dst_file)
            if src_hash == dst_hash:
                copied += src_st.st_size
                manifest_entries[rel_key] = {
                    "size": int(src_st.st_size),
                    "source_mtime_ns": int(src_st.st_mtime_ns),
                    "sha256": src_hash,
                }
                progress.update(op_id, phase="staging", bytes_copied=copied, total_bytes=total_bytes)
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
                    progress.update(
                        op_id,
                        phase="staging",
                        bytes_copied=copied,
                        total_bytes=total_bytes,
                        message=f"Staging {src_file.name} ({idx}/{total_files})",
                    )
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

    _write_stage_manifest(
        dest_dir,
        {
            "version": 2,
            "model_id": model_id,
            "source_dir": str(source_dir),
            "status": "complete",
            "total_bytes": total_bytes,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "files": manifest_entries,
        },
    )
    progress.update(
        op_id,
        phase="checkpoint_validation",
        bytes_copied=copied,
        total_bytes=total_bytes,
        progress=100.0,
        message=f"Staged and verified {model_id}",
    )
    return dest_dir
