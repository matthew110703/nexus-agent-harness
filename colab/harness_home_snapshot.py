"""Versioned, SHA-256-verified, secret-excluding Harness-home snapshots for Drive."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_MANIFEST_NAME = "snapshot-manifest.json"
SNAPSHOT_FORMAT_VERSION = 1

# Only non-secret durable session/profile state is copied to Drive snapshots.
ALLOWED_TOP_LEVEL_DIRS = ("sessions", "attachments")
ALLOWED_FILENAMES: tuple[str, ...] = ()

SECRET_DIR_NAMES = frozenset({"credentials", ".ssh", ".gnupg"})
SECRET_FILE_PATTERNS = (
    "*token*",
    "*secret*",
    "*auth*",
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
)


def is_secret_path(rel_path: Path) -> bool:
    """Return True if any path segment matches an excluded credential/secret pattern."""
    parts = [p.lower() for p in rel_path.parts]
    for seg in parts[:-1]:
        if seg in SECRET_DIR_NAMES or any(fnmatch.fnmatch(seg, pat) for pat in ("*token*", "*secret*", "*auth*")):
            return True
    name = parts[-1] if parts else ""
    if name in SECRET_DIR_NAMES:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_PATTERNS)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _collect_allowed_files(home_dir: Path) -> list[Path]:
    """Collect relative paths under home_dir that are explicitly allowed and non-secret."""
    if not home_dir.is_dir():
        return []
    collected: list[Path] = []
    for top in ALLOWED_TOP_LEVEL_DIRS:
        root = home_dir / top
        if not root.is_dir():
            continue
        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(home_dir)
            if not is_secret_path(rel):
                collected.append(rel)

    for fname in ALLOWED_FILENAMES:
        direct = home_dir / fname
        if direct.is_file():
            rel = direct.relative_to(home_dir)
            if not is_secret_path(rel):
                collected.append(rel)
        profiles_dir = home_dir / "profiles"
        if profiles_dir.is_dir():
            for match in sorted(profiles_dir.rglob(fname)):
                if not match.is_file():
                    continue
                rel = match.relative_to(home_dir)
                if not is_secret_path(rel):
                    collected.append(rel)

    # Deduplicate while preserving deterministic order.
    seen: set[str] = set()
    unique: list[Path] = []
    for rel in collected:
        key = rel.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


def create_snapshot(
    home_dir: Path,
    backup_root: Path,
    *,
    keep_last: int = 5,
) -> dict[str, Any]:
    """Create a versioned, SHA-256-verified snapshot of non-secret Harness state."""
    snapshots_root = backup_root / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)

    rel_files = _collect_allowed_files(home_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    snap_dir = snapshots_root / snap_id
    snap_dir.mkdir(parents=True, exist_ok=False)

    manifest_files: list[dict[str, Any]] = []
    total_bytes = 0
    for rel in rel_files:
        src = home_dir / rel
        dst = snap_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        size = dst.stat().st_size
        total_bytes += size
        manifest_files.append(
            {
                "rel_path": rel.as_posix(),
                "size": size,
                "sha256": digest,
            }
        )

    manifest = {
        "version": SNAPSHOT_FORMAT_VERSION,
        "snapshot_id": snap_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(manifest_files),
        "total_bytes": total_bytes,
        "files": manifest_files,
    }
    tmp_manifest = snap_dir / f".{SNAPSHOT_MANIFEST_NAME}.tmp"
    final_manifest = snap_dir / SNAPSHOT_MANIFEST_NAME
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, final_manifest)

    # Verify newly written snapshot before pruning older snapshots.
    if not verify_snapshot(snap_dir):
        raise RuntimeError(f"Newly created snapshot failed verification: {snap_dir}")

    if keep_last > 0:
        valid_snaps = list_valid_snapshots(backup_root)
        for stale in valid_snaps[keep_last:]:
            shutil.rmtree(stale, ignore_errors=True)

    return {
        "snapshot_id": snap_id,
        "snapshot_dir": str(snap_dir),
        "file_count": len(manifest_files),
        "total_bytes": total_bytes,
    }


def verify_snapshot(snap_dir: Path) -> bool:
    """Verify snapshot-manifest.json and every listed file's size and SHA-256."""
    manifest_path = snap_dir / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_FORMAT_VERSION:
        return False
    files = data.get("files")
    if not isinstance(files, list):
        return False
    for entry in files:
        if not isinstance(entry, dict):
            return False
        rel = entry.get("rel_path")
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size")
        if not isinstance(rel, str) or not isinstance(expected_sha, str) or not isinstance(expected_size, int):
            return False
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or is_secret_path(rel_path):
            return False
        target = snap_dir / rel_path
        if not target.is_file():
            return False
        if target.stat().st_size != expected_size:
            return False
        if sha256_file(target) != expected_sha:
            return False
    return True


def list_valid_snapshots(backup_root: Path) -> list[Path]:
    """Return verified snapshot directories ordered newest-first."""
    snapshots_root = backup_root / "snapshots"
    if not snapshots_root.is_dir():
        return []
    candidates = sorted(
        (p for p in snapshots_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return [c for c in candidates if verify_snapshot(c)]


def restore_latest_snapshot(backup_root: Path, target_home: Path) -> dict[str, Any]:
    """Restore the newest verified snapshot into target_home before Harness starts."""
    target_home.mkdir(parents=True, exist_ok=True)
    valid = list_valid_snapshots(backup_root)
    if not valid:
        return {"restored": False, "snapshot_id": None, "file_count": 0}

    newest = valid[0]
    manifest = json.loads((newest / SNAPSHOT_MANIFEST_NAME).read_text(encoding="utf-8"))
    restored_count = 0
    for entry in manifest["files"]:
        rel = Path(entry["rel_path"])
        src = newest / rel
        dst = target_home / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored_count += 1

    return {
        "restored": True,
        "snapshot_id": manifest.get("snapshot_id", newest.name),
        "snapshot_dir": str(newest),
        "file_count": restored_count,
    }
