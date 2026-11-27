"""Google Drive build-artifact cache for fast Colab Harness startups (v2.0.0)."""

from __future__ import annotations

import hashlib
import os
import tarfile
import time
from pathlib import Path
from typing import Callable, Optional

CACHE_PREFIX = "artifacts-"
CACHE_SUFFIX = ".tar.gz"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_artifact_dirs(harness_dir: Path) -> list[Path]:
    """Collect all built `lib/` and `apps/web/dist` directories needed for Web runtime."""
    dirs: list[Path] = []
    for pattern in (
        "packages/*/*/lib",
        "vendor/*/lib",
        "apps/cli/lib",
        "apps/web/dist",
        "apps/web/lib",
    ):
        for candidate in sorted(harness_dir.glob(pattern)):
            if candidate.is_dir():
                dirs.append(candidate)
    # Also include client build record if present
    record = harness_dir / "node_modules" / ".cache" / "dsh-client-build.json"
    if record.is_file():
        dirs.append(record)
    return dirs


def restore_build_cache(
    harness_dir: Path,
    cache_dir: Path,
    fingerprint: str,
    *,
    on_status: Optional[Callable[[str], None]] = None,
) -> bool:
    """Restore prebuilt Harness Web artifacts from Drive if a verified cache tarball exists."""
    log = on_status or (lambda _msg: None)
    fp12 = fingerprint[:12]
    archive = cache_dir / f"{CACHE_PREFIX}{fp12}{CACHE_SUFFIX}"
    sha_file = cache_dir / f"{CACHE_PREFIX}{fp12}.sha256"
    if not archive.is_file() or not sha_file.is_file():
        return False

    t0 = time.monotonic()
    expected_sha = sha_file.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = _sha256_file(archive)
    if actual_sha != expected_sha:
        log(f"[CACHE] Build cache {archive.name} failed SHA-256 verification; rebuilding.")
        return False

    log(f"[CACHE] Restoring prebuilt Harness artifacts from {archive.name} ({archive.stat().st_size / (1024 ** 2):.1f} MiB)...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=harness_dir, filter="data")
    dt = time.monotonic() - t0
    log(f"[CACHE] Restored Harness build artifacts in {dt:.1f}s.")
    return True


def save_build_cache(
    harness_dir: Path,
    cache_dir: Path,
    fingerprint: str,
    *,
    keep_latest: int = 2,
    on_status: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Archive built Harness Web artifacts to Drive atomically and prune old caches."""
    log = on_status or (lambda _msg: None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp12 = fingerprint[:12]
    archive = cache_dir / f"{CACHE_PREFIX}{fp12}{CACHE_SUFFIX}"
    sha_file = cache_dir / f"{CACHE_PREFIX}{fp12}.sha256"
    tmp_archive = cache_dir / f".{archive.name}.tmp"
    tmp_sha = cache_dir / f".{sha_file.name}.tmp"

    targets = _collect_artifact_dirs(harness_dir)
    if not targets:
        return None

    t0 = time.monotonic()
    try:
        with tarfile.open(tmp_archive, "w:gz", compresslevel=1) as tf:
            for item in targets:
                arcname = item.relative_to(harness_dir).as_posix()
                tf.add(item, arcname=arcname)
        digest = _sha256_file(tmp_archive)
        tmp_sha.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        os.replace(tmp_archive, archive)
        os.replace(tmp_sha, sha_file)
    except Exception:
        tmp_archive.unlink(missing_ok=True)
        tmp_sha.unlink(missing_ok=True)
        raise

    dt = time.monotonic() - t0
    log(
        f"[CACHE] Saved Harness build cache {archive.name} "
        f"({archive.stat().st_size / (1024 ** 2):.1f} MiB in {dt:.1f}s)."
    )

    # Prune old build caches, keeping the newest `keep_latest` archives.
    existing = sorted(cache_dir.glob(f"{CACHE_PREFIX}*{CACHE_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in existing[keep_latest:]:
        stale.unlink(missing_ok=True)
        stale_sha = cache_dir / stale.name.replace(CACHE_SUFFIX, ".sha256")
        stale_sha.unlink(missing_ok=True)

    return archive
