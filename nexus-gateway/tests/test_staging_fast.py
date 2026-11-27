"""Unit tests for v1.2.1 fast staging, zero-source-read warm cache, and corruption detection."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT / "colab", _REPO_ROOT / "nexus-gateway" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import staging  # noqa: E402


def _make_sample_model(root: Path, name: str = "Sample-Model") -> Path:
    model_dir = root / name
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "test"}) + "\n", encoding="utf-8")
    # Create a multi-megabyte shard (> 12 MiB so head/mid/tail sampling exercises all 3 slices)
    payload = b"A" * (5 * 1024 * 1024) + b"B" * (5 * 1024 * 1024) + b"C" * (5 * 1024 * 1024)
    (model_dir / "unet" / "weights.safetensors").write_bytes(payload)
    return model_dir


def test_fast_staging_warm_cache_skips_source_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _make_sample_model(tmp_path / "drive")
    stage_root = tmp_path / "local"

    # 1. Cold stage (fuse transport)
    dest = staging.stage_model_directory(
        src,
        stage_root,
        model_id="Sample-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    manifest = json.loads((dest / staging.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["version"] == 4
    assert manifest["status"] == "complete"
    assert "sample_hash" in manifest["files"]["unet/weights.safetensors"]

    # 2. Warm cache hit MUST NOT open/read source files (zero FUSE data reads)
    orig_open = Path.open
    src_reads: list[Path] = []

    def _spy_open(self: Path, *args: object, **kwargs: object):
        if str(self).startswith(str(src)):
            src_reads.append(self)
        return orig_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", _spy_open)
    dest2 = staging.stage_model_directory(
        src,
        stage_root,
        model_id="Sample-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    assert dest2 == dest
    assert src_reads == [], f"Expected zero source reads on warm fast cache hit, got: {src_reads}"


def test_fast_staging_detects_local_corruption_and_source_mtime_change(tmp_path: Path) -> None:
    src = _make_sample_model(tmp_path / "drive")
    stage_root = tmp_path / "local"

    dest = staging.stage_model_directory(
        src,
        stage_root,
        model_id="Sample-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    staged_weights = dest / "unet" / "weights.safetensors"

    # Flip 1 byte in the tail slice of the staged local copy (keeping exact file size!)
    data = bytearray(staged_weights.read_bytes())
    data[-10] ^= 0xFF
    staged_weights.write_bytes(data)

    # Re-staging in fast mode MUST detect the tail corruption and heal it
    staging.stage_model_directory(
        src,
        stage_root,
        model_id="Sample-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    assert staged_weights.read_bytes() == (src / "unet" / "weights.safetensors").read_bytes()

    # Now modify source file mtime + content (same size) -> fast mode MUST re-copy
    src_cfg = src / "config.json"
    old_st = src_cfg.stat()
    src_cfg.write_text(json.dumps({"model_type": "edit"}) + "\n", encoding="utf-8")
    os.utime(src_cfg, ns=(old_st.st_atime_ns, old_st.st_mtime_ns + 1_000_000_000))

    staging.stage_model_directory(
        src,
        stage_root,
        model_id="Sample-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    assert (dest / "config.json").read_text(encoding="utf-8") == src_cfg.read_text(encoding="utf-8")


def test_legacy_v3_manifest_backward_compatible(tmp_path: Path) -> None:
    src = tmp_path / "drive" / "Legacy-Model"
    src.mkdir(parents=True)
    cfg = src / "config.json"
    cfg.write_text('{"ok": true}\n', encoding="utf-8")
    st = cfg.stat()

    stage_root = tmp_path / "local"
    dest = stage_root / "Legacy-Model"
    dest.mkdir(parents=True)
    (dest / "config.json").write_bytes(cfg.read_bytes())
    sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    (dest / staging.MANIFEST_FILENAME).write_text(
        json.dumps({
            "version": 3,
            "model_id": "Legacy-Model",
            "source_dir": str(src),
            "status": "complete",
            "total_bytes": st.st_size,
            "files": {
                "config.json": {
                    "size": st.st_size,
                    "source_mtime_ns": st.st_mtime_ns,
                    "sha256": sha,
                }
            },
        }),
        encoding="utf-8",
    )

    # Should reuse legacy v3 cache without failing
    out = staging.stage_model_directory(
        src,
        stage_root,
        model_id="Legacy-Model",
        min_free_margin_bytes=0,
        verify_mode="fast",
        transport="fuse",
    )
    assert out == dest
