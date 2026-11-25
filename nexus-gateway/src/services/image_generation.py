"""Persistent Drive image artifact saving with sidecar generation metadata."""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _slug(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (cleaned[:max_len] or "image").strip("-")


def save_generated_image_artifact(
    *,
    output_dir: Path,
    b64_png: str,
    model_id: str,
    prompt: str,
    quality: str,
    width: int,
    height: int,
    steps: int,
    seed: Optional[int],
    operation_id: str,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    """Atomically persist a generated PNG master copy and metadata JSON sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{_slug(prompt)}_{artifact_id}.png"
    master_path = output_dir / filename
    meta_path = output_dir / f"{filename}.json"

    png_bytes = base64.b64decode(b64_png)
    tmp_png = output_dir / f".{filename}.tmp"
    tmp_png.write_bytes(png_bytes)
    os.replace(tmp_png, master_path)

    metadata: dict[str, Any] = {
        "artifact_id": artifact_id,
        "type": "generated_image",
        "filename": filename,
        "master_path": str(master_path),
        "workspace": workspace or "",
        "model": model_id,
        "quality": quality,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "operation_id": operation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bytes": len(png_bytes),
    }
    tmp_meta = output_dir / f".{filename}.json.tmp"
    tmp_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "master_path": str(master_path),
        "metadata_path": str(meta_path),
        "media_type": "image/png",
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "quality": quality,
        "model": model_id,
        "operation_id": operation_id,
        "b64_png": b64_png,
    }
