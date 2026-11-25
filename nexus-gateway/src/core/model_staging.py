"""Manifest-backed Drive-to-local model staging with checksum validation and byte progress."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from services.progress import ProgressService

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB chunks
MANIFEST_FILENAME = ".nexus-stage-manifest.json"
SMALL_HASH_LIMIT = 16 * 1024 * 1024  # Full SHA-256 for config/tokenizer/small files


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when local staging disk lacks required space + safety margin."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _file_fingerprint(path: Path, *, full_hash: bool = False) -> dict[str, Any]:
    st = path.stat()
    fp: dict[str, Any] = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    if full_hash or st.st_size <= SMALL_HASH_LIMIT:
        fp["sha256"] = _sha256_file(path)
    return fp


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
    allow_drive_fallback: bool = True,
    verify_full_hash: bool = False,
) -> Path:
    """
    Stage a Drive model directory onto Colab local disk with manifest integrity checks.
    Uses atomic per-file temp copies and a completion manifest so partial copies are
    never treated as valid cache hits.
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

    # Validate complete staging manifest + file sizes/hashes on cache hit.
    existing_manifest = _load_stage_manifest(dest_dir)
    if existing_manifest is not None and dest_dir.is_dir():
        manifest_files = existing_manifest.get("files") or {}
        cache_valid = len(manifest_files) == len(files)
        if cache_valid:
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
                if entry.get("source_mtime_ns") != int(src_st.st_mtime_ns):
                    cache_valid = False
                    break
                if "sha256" in entry and (verify_full_hash or src_st.st_size <= SMALL_HASH_LIMIT):
                    if _sha256_file(dst_file) != entry["sha256"]:
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
    # Count already-matching local files so resume doesn't double-count disk needs.
    already_present_bytes = 0
    for src_file in files:
        dst_file = dest_dir / src_file.relative_to(source_dir)
        if dst_file.is_file() and dst_file.stat().st_size == src_file.stat().st_size:
            already_present_bytes += dst_file.stat().st_size
    remaining_needed = max(0, total_bytes - already_present_bytes)

    free_bytes = shutil.disk_usage(local_cache_root).free
    if free_bytes < remaining_needed + min_free_margin_bytes:
        msg = (
            f"Insufficient local disk for {model_id}: need {remaining_needed + min_free_margin_bytes} B "
            f"(including safety margin), only {free_bytes} B free"
        )
        if not allow_drive_fallback:
            progress.fail(op_id, msg)
            raise InsufficientDiskSpaceError(msg)
        progress.update(op_id, phase="checkpoint_validation", bytes_copied=0, total_bytes=total_bytes, message=msg)
        return source_dir

    # Invalidate any prior manifest before writing new files.
    (dest_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    manifest_entries: dict[str, Any] = {}
    progress.update(op_id, phase="staging", bytes_copied=0, total_bytes=total_bytes, message=f"Staging {model_id}")

    for src_file in files:
        rel_path = src_file.relative_to(source_dir)
        rel_key = rel_path.as_posix()
        dst_file = dest_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        src_st = src_file.stat()

        # Safe resume: reuse intact file with matching size (and hash for small/verified files).
        if dst_file.is_file() and dst_file.stat().st_size == src_st.st_size:
            if not verify_full_hash and src_st.st_size > SMALL_HASH_LIMIT:
                copied += src_st.st_size
                manifest_entries[rel_key] = {
                    "size": int(src_st.st_size),
                    "source_mtime_ns": int(src_st.st_mtime_ns),
                }
                progress.update(op_id, phase="staging", bytes_copied=copied, total_bytes=total_bytes)
                continue

        tmp_file = dst_file.with_name(f".{dst_file.name}.staging.tmp")
        h = hashlib.sha256() if (verify_full_hash or src_st.st_size <= SMALL_HASH_LIMIT) else None
        try:
            with src_file.open("rb") as f_in, tmp_file.open("wb") as f_out:
                while True:
                    buf = f_in.read(CHUNK_SIZE)
                    if not buf:
                        break
                    f_out.write(buf)
                    if h is not None:
                        h.update(buf)
                    copied += len(buf)
                    progress.update(
                        op_id,
                        phase="staging",
                        bytes_copied=copied,
                        total_bytes=total_bytes,
                        message=f"Staging {model_id} ({copied}/{total_bytes} bytes)",
                    )
                f_out.flush()
                os.fsync(f_out.fileno())
            os.replace(tmp_file, dst_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)
            raise

        entry: dict[str, Any] = {
            "size": int(src_st.st_size),
            "source_mtime_ns": int(src_st.st_mtime_ns),
        }
        if h is not None:
            entry["sha256"] = h.hexdigest()
        manifest_entries[rel_key] = entry

    _write_stage_manifest(
        dest_dir,
        {
            "version": 1,
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
