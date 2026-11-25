"""Idempotent Google Drive layout initializer and hash-verified source sync for Nexus Colab."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

ChangeKind = Literal["added", "modified", "deleted", "conflict", "unchanged"]
RestartTarget = Literal[
    "none",
    "config_reload",
    "gateway_api",
    "model_worker",
    "supervisor",
    "harness_ts",
    "dependencies",
]

# Additive directories only; existing user asset directories (models, adapters,
# upscalers, datasets, finetunes, outputs, ui, jobs, video_runner) are never
# moved, renamed, or deleted.
ADDITIVE_DRIVE_DIRS: tuple[str, ...] = (
    "colab-src/nexus-gateway",
    "colab-src/supervisor",
    "colab-src/scripts",
    "colab-src/config",
    "workspaces/projects",
    "workspaces/shared",
    "harness-home",
    "runtime-manifests/source",
    "runtime-manifests/staging",
    "runtime-manifests/runtime",
    "huggingface-cache",
)

# Relative Drive colab-src/ prefix -> relative destination prefix inside runtime repo.
SYNC_ROOT_MAP: dict[str, str] = {
    "nexus-gateway": "nexus-gateway",
    "supervisor": "colab",
    "scripts": "colab",
    "config": "apps/cli/config/examples",
}

IGNORED_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
    ".pytest_cache",
    "sync-manifest.json",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relative_path(root: Path, rel: str) -> Path:
    if not rel or rel.startswith(("/", "\\")):
        raise ValueError(f"Unsafe absolute sync path: {rel!r}")
    parts = Path(rel).parts
    if any(p in ("..", "") for p in parts):
        raise ValueError(f"Path traversal rejected: {rel!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / rel).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Resolved path escapes root {resolved_root}: {candidate}")
    return candidate


def classify_restart_target(runtime_rel_path: str) -> RestartTarget:
    norm = runtime_rel_path.replace("\\", "/")
    if norm.endswith(("requirements.txt", "pyproject.toml", "package.json", "pnpm-lock.yaml")):
        return "dependencies"
    if norm.startswith("nexus-gateway/src/workers/") or norm.startswith("nexus-gateway/src/adapters/"):
        return "model_worker"
    if norm.startswith("nexus-gateway/"):
        return "gateway_api"
    if norm.startswith("colab/supervisor"):
        return "supervisor"
    if norm.startswith("packages/") or norm.startswith("apps/cli/src/"):
        return "harness_ts"
    if norm.endswith((".yml", ".yaml", ".json")):
        return "config_reload"
    return "none"


def init_drive_layout(drive_root: Path) -> dict[str, list[str]]:
    """Create only missing additive Nexus directories under `drive_root`."""
    created: list[str] = []
    existing: list[str] = []
    drive_root.mkdir(parents=True, exist_ok=True)
    for rel in ADDITIVE_DRIVE_DIRS:
        target = _safe_relative_path(drive_root, rel)
        if target.is_dir():
            existing.append(rel)
        else:
            target.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    return {"created": created, "existing": existing}


@dataclass
class ManifestEntry:
    source_path: str
    runtime_path: str
    baseline_hash: str
    drive_hash: str
    runtime_hash: str
    last_synced_at: str
    git_commit: str
    status: str = "synced"


@dataclass
class FileSyncAction:
    source_path: str
    runtime_path: str
    kind: ChangeKind
    restart_target: RestartTarget
    drive_hash: Optional[str] = None
    runtime_hash: Optional[str] = None
    baseline_hash: Optional[str] = None
    reason: str = ""


@dataclass
class SyncReport:
    dry_run: bool
    git_commit: str
    timestamp: str
    actions: list[FileSyncAction] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    restart_targets: list[RestartTarget] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"version": 1, "git_commit": "", "updated_at": "", "files": {}}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "git_commit": "", "updated_at": "", "files": {}, "recovered": True}


def _atomic_write_bytes(dest: Path, content: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".nexus-sync-", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(dest: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(dest, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def map_drive_to_runtime_rel(drive_rel: str) -> Optional[str]:
    norm = drive_rel.replace("\\", "/")
    parts = norm.split("/", 1)
    if len(parts) != 2:
        return None
    bucket, rest = parts
    if bucket not in SYNC_ROOT_MAP or not rest:
        return None
    if any(seg in IGNORED_NAMES or seg.endswith(".pyc") for seg in Path(rest).parts):
        return None
    return f"{SYNC_ROOT_MAP[bucket]}/{rest}"


def _scan_drive_files(drive_src: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not drive_src.is_dir():
        return found
    for path in sorted(drive_src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(drive_src).as_posix()
        if any(seg in IGNORED_NAMES or seg.endswith(".pyc") for seg in path.relative_to(drive_src).parts):
            continue
        if map_drive_to_runtime_rel(rel) is None:
            continue
        found[rel] = path
    return found


def seed_drive_baseline(
    runtime_root: Path,
    drive_src: Path,
    manifest_path: Path,
    *,
    git_commit: str,
    rel_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Seed missing files in `drive_src` from `runtime_root` without overwriting user Drive edits."""
    manifest = load_manifest(manifest_path)
    files_meta: dict[str, Any] = manifest.setdefault("files", {})
    seeded: list[str] = []
    skipped: list[str] = []
    now = _utc_now()
    for drive_rel, runtime_rel in rel_pairs:
        src_runtime = _safe_relative_path(runtime_root, runtime_rel)
        dst_drive = _safe_relative_path(drive_src, drive_rel)
        if not src_runtime.is_file():
            continue
        if dst_drive.is_file():
            skipped.append(drive_rel)
            continue
        content = src_runtime.read_bytes()
        _atomic_write_bytes(dst_drive, content)
        digest = sha256_file(dst_drive)
        files_meta[drive_rel] = asdict(
            ManifestEntry(
                source_path=drive_rel,
                runtime_path=runtime_rel,
                baseline_hash=digest,
                drive_hash=digest,
                runtime_hash=digest,
                last_synced_at=now,
                git_commit=git_commit,
                status="seeded",
            )
        )
        seeded.append(drive_rel)
    manifest["git_commit"] = git_commit
    manifest["updated_at"] = now
    _atomic_write_json(manifest_path, manifest)
    return {"seeded": seeded, "skipped": skipped}


def plan_sync(
    drive_src: Path,
    runtime_root: Path,
    manifest_path: Path,
    *,
    git_commit: str = "",
) -> SyncReport:
    """Compute a 3-way hash diff between Drive source, runtime tree, and last-synced manifest."""
    manifest = load_manifest(manifest_path)
    prev_files: dict[str, Any] = manifest.get("files", {})
    drive_files = _scan_drive_files(drive_src)
    report = SyncReport(dry_run=True, git_commit=git_commit, timestamp=_utc_now())

    targets_seen: set[RestartTarget] = set()

    for drive_rel, drive_path in drive_files.items():
        runtime_rel = map_drive_to_runtime_rel(drive_rel)
        if runtime_rel is None:
            continue
        runtime_path = _safe_relative_path(runtime_root, runtime_rel)
        drive_hash = sha256_file(drive_path)
        runtime_hash = sha256_file(runtime_path) if runtime_path.is_file() else None
        prev = prev_files.get(drive_rel) or {}
        baseline_hash = prev.get("baseline_hash")
        last_drive_hash = prev.get("drive_hash")
        last_runtime_hash = prev.get("runtime_hash")

        if runtime_hash == drive_hash:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    kind="unchanged",
                    restart_target="none",
                    drive_hash=drive_hash,
                    runtime_hash=runtime_hash,
                    baseline_hash=baseline_hash,
                )
            )
            continue

        target = classify_restart_target(runtime_rel)

        # Conflict check: both Drive and upstream runtime changed relative to last sync.
        drive_changed = last_drive_hash is not None and drive_hash != last_drive_hash
        runtime_changed = (
            last_runtime_hash is not None
            and runtime_hash is not None
            and runtime_hash != last_runtime_hash
        )
        if drive_changed and runtime_changed:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    kind="conflict",
                    restart_target=target,
                    drive_hash=drive_hash,
                    runtime_hash=runtime_hash,
                    baseline_hash=baseline_hash,
                    reason="Both Drive source and upstream runtime changed since last sync",
                )
            )
            report.conflicts.append(drive_rel)
            continue

        kind: ChangeKind = "added" if runtime_hash is None else "modified"
        report.actions.append(
            FileSyncAction(
                source_path=drive_rel,
                runtime_path=runtime_rel,
                kind=kind,
                restart_target=target,
                drive_hash=drive_hash,
                runtime_hash=runtime_hash,
                baseline_hash=baseline_hash,
                reason="New file in Drive colab-src" if kind == "added" else "Drive source updated",
            )
        )
        if target != "none":
            targets_seen.add(target)

    # Detect files removed from Drive since last manifest (report safely, do not delete user files blindly).
    for prev_rel, prev_entry in prev_files.items():
        if prev_rel not in drive_files:
            runtime_rel = prev_entry.get("runtime_path") or map_drive_to_runtime_rel(prev_rel) or ""
            report.actions.append(
                FileSyncAction(
                    source_path=prev_rel,
                    runtime_path=runtime_rel,
                    kind="deleted",
                    restart_target="none",
                    baseline_hash=prev_entry.get("baseline_hash"),
                    reason="Absent from Drive colab-src; runtime copy preserved until explicit deletion",
                )
            )

    report.restart_targets = sorted(targets_seen)  # type: ignore[assignment]
    return report


def apply_sync(
    drive_src: Path,
    runtime_root: Path,
    manifest_path: Path,
    *,
    git_commit: str = "",
    dry_run: bool = False,
    allow_conflict_overwrite: bool = False,
) -> SyncReport:
    """Apply Drive source changes atomically with backup/rollback and syntax validation."""
    report = plan_sync(drive_src, runtime_root, manifest_path, git_commit=git_commit)
    report.dry_run = dry_run
    if dry_run:
        return report

    if report.conflicts and not allow_conflict_overwrite:
        report.errors.append(
            f"Refusing to overwrite {len(report.conflicts)} conflicting file(s): {', '.join(report.conflicts)}"
        )
        return report

    to_apply = [
        a
        for a in report.actions
        if a.kind in ("added", "modified") or (allow_conflict_overwrite and a.kind == "conflict")
    ]
    if not to_apply:
        return report

    # Pre-validate Python / JSON files before touching runtime tree.
    for action in to_apply:
        src = _safe_relative_path(drive_src, action.source_path)
        if src.suffix == ".py":
            try:
                compile(src.read_text(encoding="utf-8"), str(src), "exec")
            except SyntaxError as exc:
                report.errors.append(f"Python syntax error in {action.source_path}: {exc}")
        elif src.suffix == ".json":
            try:
                json.loads(src.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                report.errors.append(f"Invalid JSON in {action.source_path}: {exc}")
    if report.errors:
        return report

    backups: list[tuple[Path, Optional[bytes]]] = []
    manifest = load_manifest(manifest_path)
    files_meta: dict[str, Any] = manifest.setdefault("files", {})
    now = _utc_now()

    try:
        for action in to_apply:
            src = _safe_relative_path(drive_src, action.source_path)
            dst = _safe_relative_path(runtime_root, action.runtime_path)
            backups.append((dst, dst.read_bytes() if dst.is_file() else None))
            _atomic_write_bytes(dst, src.read_bytes())
            new_hash = sha256_file(dst)
            files_meta[action.source_path] = asdict(
                ManifestEntry(
                    source_path=action.source_path,
                    runtime_path=action.runtime_path,
                    baseline_hash=action.baseline_hash or new_hash,
                    drive_hash=new_hash,
                    runtime_hash=new_hash,
                    last_synced_at=now,
                    git_commit=git_commit,
                    status="synced",
                )
            )
            report.applied.append(action.runtime_path)
    except Exception as exc:
        report.errors.append(f"Sync failed and rolled back: {exc}")
        report.rolled_back = True
        for dst, previous in reversed(backups):
            try:
                if previous is None:
                    dst.unlink(missing_ok=True)
                else:
                    _atomic_write_bytes(dst, previous)
            except OSError:
                pass
        report.applied.clear()
        return report

    manifest["git_commit"] = git_commit
    manifest["updated_at"] = now
    _atomic_write_json(manifest_path, manifest)
    return report
