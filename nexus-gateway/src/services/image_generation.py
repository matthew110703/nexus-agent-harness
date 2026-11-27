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


def resolve_image_target_dir(
    default_output_dir: Path,
    workspace: Optional[str] = None,
    drive_root: Optional[Path] = None,
) -> Path:
    """
    Resolve the directory where generated images should be saved.
    If workspace is provided:
      1. Resolves relative to drive_root (or absolute if inside drive_root or /content).
      2. Prevents path traversal outside permitted Drive / workspace boundaries.
      3. Targets <resolved_workspace>/generated_images/.
    Falls back safely to default_output_dir if workspace is missing or invalid.
    """
    if not workspace or not str(workspace).strip():
        return default_output_dir

    raw_ws = str(workspace).strip()
    root = (drive_root or default_output_dir).resolve()
    try:
        ws_path = Path(raw_ws).resolve()
        # If relative or not rooted in Drive / /content, join with root
        if not ws_path.is_absolute():
            ws_path = (root / raw_ws).resolve()

        # Path traversal guard: must not resolve to filesystem root or traverse outside
        # Accept paths under drive_root, /content/drive, or default_output_dir
        allowed_parents = [root, default_output_dir.resolve(), Path("/content/drive").resolve()]
        if any(p.exists() for p in allowed_parents):
            is_sub = False
            for parent in allowed_parents:
                try:
                    ws_path.relative_to(parent)
                    is_sub = True
                    break
                except ValueError:
                    continue
            if not is_sub and ws_path != root:
                # Disallowed traversal, fallback to default_output_dir
                return default_output_dir

        target_dir = ws_path / "generated_images"
        return target_dir
    except Exception:
        return default_output_dir


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
    drive_root: Optional[Path] = None,
    session_id: Optional[str] = None,
    reference_image_used: bool = False,
    strength: Optional[float] = None,
) -> dict[str, Any]:
    """Atomically persist a generated PNG master copy to Drive and optionally mirror into workspace/generated_images."""
    drive_dir = resolve_image_target_dir(output_dir, workspace=None, drive_root=drive_root)
    drive_dir.mkdir(parents=True, exist_ok=True)
    ws_dir = resolve_image_target_dir(output_dir, workspace=workspace, drive_root=drive_root) if workspace else None
    if ws_dir is not None and ws_dir != drive_dir:
        ws_dir.mkdir(parents=True, exist_ok=True)
    else:
        ws_dir = None

    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{_slug(prompt)}_{artifact_id}.png"
    drive_path = drive_dir / filename
    drive_meta_path = drive_dir / f"{filename}.json"

    png_bytes = base64.b64decode(b64_png)
    tmp_png = drive_dir / f".{filename}.tmp"
    tmp_png.write_bytes(png_bytes)
    os.replace(tmp_png, drive_path)

    workspace_path: Optional[Path] = None
    if ws_dir is not None:
        workspace_path = ws_dir / filename
        tmp_ws_png = ws_dir / f".{filename}.tmp"
        tmp_ws_png.write_bytes(png_bytes)
        os.replace(tmp_ws_png, workspace_path)

    master_path = workspace_path or drive_path
    metadata: dict[str, Any] = {
        "artifact_id": artifact_id,
        "type": "generated_image",
        "filename": filename,
        "master_path": str(master_path),
        "drive_path": str(drive_path),
        "workspace_path": str(workspace_path) if workspace_path is not None else "",
        "workspace": workspace or "",
        "session_id": session_id or "",
        "model": model_id,
        "quality": quality,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "reference_image_used": bool(reference_image_used),
        "strength": strength,
        "operation_id": operation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bytes": len(png_bytes),
    }
    meta_json = json.dumps(metadata, indent=2) + "\n"
    tmp_meta = drive_dir / f".{filename}.json.tmp"
    tmp_meta.write_text(meta_json, encoding="utf-8")
    os.replace(tmp_meta, drive_meta_path)
    if ws_dir is not None and workspace_path is not None:
        ws_meta_path = ws_dir / f"{filename}.json"
        tmp_ws_meta = ws_dir / f".{filename}.json.tmp"
        tmp_ws_meta.write_text(meta_json, encoding="utf-8")
        os.replace(tmp_ws_meta, ws_meta_path)

    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "master_path": str(master_path),
        "drive_path": str(drive_path),
        "workspace_path": str(workspace_path) if workspace_path is not None else "",
        "metadata_path": str(drive_meta_path),
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
