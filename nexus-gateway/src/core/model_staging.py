"""First-class Drive-to-local model staging with real byte-level progress."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from services.progress import ProgressService

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB chunks


def stage_model_directory(
    source_dir: Path,
    local_cache_root: Path,
    *,
    model_id: str,
    progress: ProgressService,
    operation_id: Optional[str] = None,
    min_free_margin_bytes: int = 2 * 1024 * 1024 * 1024,
) -> Path:
    """
    Stage a Drive model directory onto Colab local disk with real byte progress.
    Returns local path on cache hit or successful copy, or falls back to `source_dir`
    if local disk space is insufficient.
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

    files = [p for p in source_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)

    # Cache hit check: all files exist with matching byte sizes.
    if dest_dir.is_dir():
        cached_ok = all(
            (dest_dir / p.relative_to(source_dir)).is_file()
            and (dest_dir / p.relative_to(source_dir)).stat().st_size == p.stat().st_size
            for p in files
        )
        if cached_ok:
            progress.update(
                op_id,
                phase="checkpoint_validation",
                bytes_copied=total_bytes,
                total_bytes=total_bytes,
                progress=100.0,
                message=f"Using local staged cache for {model_id}",
            )
            return dest_dir

    local_cache_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(local_cache_root).free
    if free_bytes < total_bytes + min_free_margin_bytes:
        progress.update(
            op_id,
            phase="checkpoint_validation",
            bytes_copied=0,
            total_bytes=total_bytes,
            message=f"Insufficient local disk ({free_bytes} B free); loading directly from Drive",
        )
        return source_dir

    copied = 0
    progress.update(op_id, phase="staging", bytes_copied=0, total_bytes=total_bytes, message=f"Staging {model_id}")
    for src_file in files:
        rel = src_file.relative_to(source_dir)
        dst_file = dest_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with src_file.open("rb") as f_in, dst_file.open("wb") as f_out:
            while True:
                buf = f_in.read(CHUNK_SIZE)
                if not buf:
                    break
                f_out.write(buf)
                copied += len(buf)
                progress.update(
                    op_id,
                    phase="staging",
                    bytes_copied=copied,
                    total_bytes=total_bytes,
                    message=f"Staging {model_id} ({copied}/{total_bytes} bytes)",
                )

    progress.update(op_id, phase="checkpoint_validation", bytes_copied=copied, total_bytes=total_bytes, progress=100.0)
    return dest_dir
