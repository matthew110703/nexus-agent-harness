"""Idempotent Google Drive layout initializer, v2 3-way source sync, and build fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

ChangeKind = Literal[
    "added",
    "modified",
    "runtime_advanced",
    "converged",
    "deleted",
    "conflict",
    "adoption_conflict",
    "unchanged",
]
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
    "harness-home/snapshots",
    "runtime-manifests/source",
    "runtime-manifests/staging",
    "runtime-manifests/runtime",
    "huggingface-cache",
)

# Explicit file-level mappings from Drive colab-src/ -> runtime repo path.
EXPLICIT_FILE_MAP: dict[str, str] = {
    "supervisor/supervisor.mjs": "colab/supervisor.mjs",
    "scripts/sync_source.py": "colab/sync_source.py",
    "scripts/harness_home_snapshot.py": "colab/harness_home_snapshot.py",
    "config/colab.overlay.yml": "apps/cli/config/examples/colab.overlay.yml",
    "config/plugin-index.json": "colab/plugin-index.json",
    "config/model-capabilities.example.json": "colab/model-capabilities.example.json",
}

# Drive-native runtime files consumed directly from Drive (never copied into apps/cli/config/examples/).
DRIVE_NATIVE_FILES: dict[str, RestartTarget] = {
    "config/model-capabilities.json": "gateway_api",
}

IGNORED_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
    ".pytest_cache",
    "sync-manifest.json",
})

BUILD_FINGERPRINT_ROOT_FILES: tuple[str, ...] = (
    "package.json",
    "pnpm-lock.yaml",
    "tsconfig.json",
    "tsconfig.host.json",
    "tsconfig.client.json",
)

BUILD_FINGERPRINT_DIRS: tuple[str, ...] = (
    "packages/client/nexus-runtime-ui",
    "packages/experimental/nexus-host",
    "packages/bundle/web-app",
    "apps/cli",
    "colab",
    "nexus-gateway/src",
)


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
    if norm.startswith("nexus-gateway/") or norm == "colab/plugin-index.json":
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


def validate_capabilities_sidecar(data: Any) -> list[str]:
    """Validate `model-capabilities.json` schema; return list of error strings."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Capability sidecar root must be a JSON object"]
    models = data.get("models")
    if not isinstance(models, dict):
        return ["Capability sidecar must contain a 'models' object"]
    for model_id, cfg in models.items():
        if not isinstance(cfg, dict):
            errors.append(f"models.{model_id} must be an object")
            continue
        if "tool_behavior_verified" in cfg and cfg["tool_behavior_verified"] is not None and not isinstance(cfg["tool_behavior_verified"], bool):
            errors.append(f"models.{model_id}.tool_behavior_verified must be a boolean or null")
        reasoning = cfg.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                errors.append(f"models.{model_id}.reasoning must be an object")
            else:
                status = reasoning.get("status")
                if status not in ("supported", "unsupported", "unknown", None):
                    errors.append(f"models.{model_id}.reasoning.status invalid: {status!r}")
                levels = reasoning.get("levels")
                if levels is not None and not isinstance(levels, dict):
                    errors.append(f"models.{model_id}.reasoning.levels must be an object")
    return errors


def ensure_capabilities_sidecar(
    drive_config_dir: Path,
    canonical_example_path: Path,
) -> dict[str, Any]:
    """
    Create `model-capabilities.json` from the canonical example only when absent.
    Preserves user edits on subsequent runs and validates the active sidecar schema.
    """
    drive_config_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = drive_config_dir / "model-capabilities.json"
    created = False
    if not sidecar_path.is_file() and canonical_example_path.is_file():
        _atomic_write_bytes(sidecar_path, canonical_example_path.read_bytes())
        created = True

    if not sidecar_path.is_file():
        return {"path": str(sidecar_path), "loaded": False, "created": False, "errors": []}

    try:
        parsed = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(sidecar_path), "loaded": False, "created": created, "errors": [f"Invalid JSON: {exc}"]}

    errors = validate_capabilities_sidecar(parsed)
    return {"path": str(sidecar_path), "loaded": len(errors) == 0, "created": created, "errors": errors}


@dataclass
class ManifestEntry:
    source_path: str
    runtime_path: str
    last_common_hash: str
    observed_runtime_hash: Optional[str] = None
    last_synced_at: str = ""
    git_commit: str = ""
    status: str = "synced"


@dataclass
class FileSyncAction:
    source_path: str
    runtime_path: str
    kind: ChangeKind
    restart_target: RestartTarget
    drive_hash: Optional[str] = None
    runtime_hash: Optional[str] = None
    last_common_hash: Optional[str] = None
    observed_runtime_hash: Optional[str] = None
    reason: str = ""


@dataclass
class SyncReport:
    dry_run: bool
    git_commit: str
    timestamp: str
    actions: list[FileSyncAction] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    adoption_conflicts: list[str] = field(default_factory=list)
    restart_targets: list[RestartTarget] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load v2 manifest, automatically migrating legacy v1 entries."""
    if not manifest_path.is_file():
        return {"version": 2, "git_commit": "", "updated_at": "", "files": {}}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            if data.get("version", 1) < 2:
                migrated: dict[str, Any] = {}
                for k, v in data["files"].items():
                    if not isinstance(v, dict):
                        continue
                    common = v.get("baseline_hash") or v.get("drive_hash") or v.get("runtime_hash") or ""
                    rt_obs = v.get("runtime_hash") if v.get("runtime_hash") and v.get("runtime_hash") != common else None
                    migrated[k] = asdict(
                        ManifestEntry(
                            source_path=str(v.get("source_path") or k),
                            runtime_path=str(v.get("runtime_path") or ""),
                            last_common_hash=str(common),
                            observed_runtime_hash=str(rt_obs) if rt_obs else None,
                            last_synced_at=str(v.get("last_synced_at") or ""),
                            git_commit=str(v.get("git_commit") or ""),
                            status=str(v.get("status") or "migrated"),
                        )
                    )
                return {
                    "version": 2,
                    "git_commit": str(data.get("git_commit") or ""),
                    "updated_at": str(data.get("updated_at") or ""),
                    "files": migrated,
                }
            return data
    except Exception:
        pass
    return {"version": 2, "git_commit": "", "updated_at": "", "files": {}, "recovered": True}


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
    if any(seg in IGNORED_NAMES or seg.endswith(".pyc") for seg in Path(norm).parts):
        return None
    if norm in DRIVE_NATIVE_FILES:
        return None
    if norm in EXPLICIT_FILE_MAP:
        return EXPLICIT_FILE_MAP[norm]
    if norm.startswith("nexus-gateway/") and len(norm) > len("nexus-gateway/"):
        return norm
    return None


def _scan_drive_files(drive_src: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    synced: dict[str, Path] = {}
    drive_native: dict[str, Path] = {}
    if not drive_src.is_dir():
        return synced, drive_native
    for path in sorted(drive_src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(drive_src).as_posix()
        if any(seg in IGNORED_NAMES or seg.endswith(".pyc") for seg in path.relative_to(drive_src).parts):
            continue
        if rel in DRIVE_NATIVE_FILES:
            drive_native[rel] = path
        elif map_drive_to_runtime_rel(rel) is not None:
            synced[rel] = path
    return synced, drive_native


def seed_drive_baseline(
    runtime_root: Path,
    drive_src: Path,
    manifest_path: Path,
    *,
    git_commit: str,
    rel_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """
    Seed missing files in `drive_src` from `runtime_root` and establish provenance for
    identical existing files. Never silently overwrites differing existing Drive files.
    """
    manifest = load_manifest(manifest_path)
    files_meta: dict[str, Any] = manifest.setdefault("files", {})
    seeded: list[str] = []
    adopted: list[str] = []
    adoption_conflicts: list[str] = []
    now = _utc_now()

    for drive_rel, runtime_rel in rel_pairs:
        src_runtime = _safe_relative_path(runtime_root, runtime_rel)
        dst_drive = _safe_relative_path(drive_src, drive_rel)
        if not src_runtime.is_file():
            continue
        rt_digest = sha256_file(src_runtime)
        if not dst_drive.is_file():
            _atomic_write_bytes(dst_drive, src_runtime.read_bytes())
            files_meta[drive_rel] = asdict(
                ManifestEntry(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    last_common_hash=rt_digest,
                    observed_runtime_hash=None,
                    last_synced_at=now,
                    git_commit=git_commit,
                    status="seeded",
                )
            )
            seeded.append(drive_rel)
            continue

        # Existing Drive file: establish baseline only if identical or already tracked.
        if drive_rel in files_meta:
            continue
        drive_digest = sha256_file(dst_drive)
        if drive_digest == rt_digest:
            files_meta[drive_rel] = asdict(
                ManifestEntry(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    last_common_hash=drive_digest,
                    observed_runtime_hash=None,
                    last_synced_at=now,
                    git_commit=git_commit,
                    status="adopted",
                )
            )
            adopted.append(drive_rel)
        else:
            adoption_conflicts.append(drive_rel)

    manifest["version"] = 2
    manifest["git_commit"] = git_commit
    manifest["updated_at"] = now
    _atomic_write_json(manifest_path, manifest)
    return {"seeded": seeded, "adopted": adopted, "adoption_conflicts": adoption_conflicts}


def adopt_baseline(
    drive_src: Path,
    runtime_root: Path,
    manifest_path: Path,
    source_path: str,
    *,
    prefer: Literal["runtime", "drive"],
    git_commit: str = "",
) -> ManifestEntry:
    """Explicitly resolve an untracked adoption conflict by choosing which side becomes baseline."""
    runtime_rel = map_drive_to_runtime_rel(source_path)
    if runtime_rel is None:
        raise ValueError(f"Unmapped source path: {source_path!r}")
    drive_path = _safe_relative_path(drive_src, source_path)
    runtime_path = _safe_relative_path(runtime_root, runtime_rel)
    if prefer == "runtime":
        if not runtime_path.is_file():
            raise FileNotFoundError(f"Runtime file does not exist: {runtime_path}")
        _atomic_write_bytes(drive_path, runtime_path.read_bytes())
        chosen_hash = sha256_file(drive_path)
    else:
        if not drive_path.is_file():
            raise FileNotFoundError(f"Drive file does not exist: {drive_path}")
        _atomic_write_bytes(runtime_path, drive_path.read_bytes())
        chosen_hash = sha256_file(runtime_path)

    manifest = load_manifest(manifest_path)
    now = _utc_now()
    entry = ManifestEntry(
        source_path=source_path,
        runtime_path=runtime_rel,
        last_common_hash=chosen_hash,
        observed_runtime_hash=None,
        last_synced_at=now,
        git_commit=git_commit,
        status=f"adopted_{prefer}",
    )
    manifest.setdefault("files", {})[source_path] = asdict(entry)
    manifest["version"] = 2
    manifest["git_commit"] = git_commit
    manifest["updated_at"] = now
    _atomic_write_json(manifest_path, manifest)
    return entry


def plan_sync(
    drive_src: Path,
    runtime_root: Path,
    manifest_path: Path,
    *,
    git_commit: str = "",
) -> SyncReport:
    """Compute a true 3-way hash diff using immutable `last_common_hash` + `observed_runtime_hash`."""
    manifest = load_manifest(manifest_path)
    prev_files: dict[str, Any] = manifest.get("files", {})
    drive_files, drive_native = _scan_drive_files(drive_src)
    report = SyncReport(dry_run=True, git_commit=git_commit, timestamp=_utc_now())
    targets_seen: set[RestartTarget] = set()

    # 1. Tracked Drive-native files (e.g. config/model-capabilities.json).
    for drive_rel, drive_path in drive_native.items():
        drive_hash = sha256_file(drive_path)
        prev = prev_files.get(drive_rel) or {}
        last_hash = prev.get("last_common_hash")
        target = DRIVE_NATIVE_FILES[drive_rel]
        if last_hash == drive_hash:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=drive_rel,
                    kind="unchanged",
                    restart_target="none",
                    drive_hash=drive_hash,
                    last_common_hash=last_hash,
                )
            )
        else:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=drive_rel,
                    kind="modified" if last_hash else "added",
                    restart_target=target,
                    drive_hash=drive_hash,
                    last_common_hash=last_hash,
                    reason="Drive-native capability sidecar updated",
                )
            )
            targets_seen.add(target)

    # 2. Mapped Drive -> Runtime files.
    for drive_rel, drive_path in drive_files.items():
        runtime_rel = map_drive_to_runtime_rel(drive_rel)
        if runtime_rel is None:
            continue
        runtime_path = _safe_relative_path(runtime_root, runtime_rel)
        drive_hash = sha256_file(drive_path)
        runtime_hash = sha256_file(runtime_path) if runtime_path.is_file() else None
        prev = prev_files.get(drive_rel)
        last_common = prev.get("last_common_hash") if isinstance(prev, dict) else None
        observed_rt = prev.get("observed_runtime_hash") if isinstance(prev, dict) else None
        target = classify_restart_target(runtime_rel)

        # Case A: Brand-new file in Drive with no runtime counterpart.
        if runtime_hash is None:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    kind="added",
                    restart_target=target,
                    drive_hash=drive_hash,
                    runtime_hash=None,
                    last_common_hash=last_common,
                    reason="New file in Drive colab-src",
                )
            )
            if target != "none":
                targets_seen.add(target)
            continue

        # Case B: Both sides currently match!
        if drive_hash == runtime_hash:
            if last_common != drive_hash or observed_rt is not None:
                report.actions.append(
                    FileSyncAction(
                        source_path=drive_rel,
                        runtime_path=runtime_rel,
                        kind="converged",
                        restart_target="none",
                        drive_hash=drive_hash,
                        runtime_hash=runtime_hash,
                        last_common_hash=last_common,
                        reason="Drive and runtime converged to identical content",
                    )
                )
            else:
                report.actions.append(
                    FileSyncAction(
                        source_path=drive_rel,
                        runtime_path=runtime_rel,
                        kind="unchanged",
                        restart_target="none",
                        drive_hash=drive_hash,
                        runtime_hash=runtime_hash,
                        last_common_hash=last_common,
                    )
                )
            continue

        # Case C: Existing Drive file with NO manifest provenance and runtime differs -> adoption conflict.
        if not last_common:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    kind="adoption_conflict",
                    restart_target=target,
                    drive_hash=drive_hash,
                    runtime_hash=runtime_hash,
                    reason="Existing Drive file has no baseline provenance and differs from runtime; explicit adoption required",
                )
            )
            report.adoption_conflicts.append(drive_rel)
            report.conflicts.append(drive_rel)
            continue

        drive_changed = drive_hash != last_common
        runtime_changed = runtime_hash != last_common

        # Case D: Both Drive and runtime changed independently since last_common_hash -> CONFLICT.
        if drive_changed and runtime_changed:
            report.actions.append(
                FileSyncAction(
                    source_path=drive_rel,
                    runtime_path=runtime_rel,
                    kind="conflict",
                    restart_target=target,
                    drive_hash=drive_hash,
                    runtime_hash=runtime_hash,
                    last_common_hash=last_common,
                    observed_runtime_hash=observed_rt,
                    reason="Both Drive and runtime changed independently since last common baseline",
                )
            )
            report.conflicts.append(drive_rel)
            continue

        # Case E: Runtime-only upstream change -> preserve runtime, keep last_common_hash immutable!
        if runtime_changed and not drive_changed:
            if observed_rt == runtime_hash:
                report.actions.append(
                    FileSyncAction(
                        source_path=drive_rel,
                        runtime_path=runtime_rel,
                        kind="unchanged",
                        restart_target="none",
                        drive_hash=drive_hash,
                        runtime_hash=runtime_hash,
                        last_common_hash=last_common,
                        observed_runtime_hash=observed_rt,
                        reason="Preserved upstream runtime change already recorded",
                    )
                )
            else:
                report.actions.append(
                    FileSyncAction(
                        source_path=drive_rel,
                        runtime_path=runtime_rel,
                        kind="runtime_advanced",
                        restart_target="none",
                        drive_hash=drive_hash,
                        runtime_hash=runtime_hash,
                        last_common_hash=last_common,
                        observed_runtime_hash=runtime_hash,
                        reason="Upstream runtime changed while Drive stayed at last common baseline; preserving runtime",
                    )
                )
            continue

        # Case F: Drive-only change -> safe to apply to runtime.
        report.actions.append(
            FileSyncAction(
                source_path=drive_rel,
                runtime_path=runtime_rel,
                kind="modified",
                restart_target=target,
                drive_hash=drive_hash,
                runtime_hash=runtime_hash,
                last_common_hash=last_common,
                reason="Drive source updated relative to last common baseline",
            )
        )
        if target != "none":
            targets_seen.add(target)

    # 3. Detect files removed from Drive since last manifest (never delete runtime blindly).
    for prev_rel, prev_entry in prev_files.items():
        if prev_rel not in drive_files and prev_rel not in drive_native:
            runtime_rel = prev_entry.get("runtime_path") or map_drive_to_runtime_rel(prev_rel) or ""
            report.actions.append(
                FileSyncAction(
                    source_path=prev_rel,
                    runtime_path=runtime_rel,
                    kind="deleted",
                    restart_target="none",
                    last_common_hash=prev_entry.get("last_common_hash"),
                    reason="Removed from Drive colab-src; runtime copy preserved until explicit deletion",
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
) -> SyncReport:
    """Apply Drive source changes atomically with validation, rollback, and v2 manifest updates."""
    report = plan_sync(drive_src, runtime_root, manifest_path, git_commit=git_commit)
    report.dry_run = dry_run
    if dry_run:
        return report

    if report.conflicts:
        report.errors.append(
            f"Refusing to overwrite {len(report.conflicts)} conflicting file(s): {', '.join(report.conflicts)}"
        )
        return report

    # Validate Drive files before touching runtime tree.
    to_copy = [a for a in report.actions if a.kind in ("added", "modified") and a.source_path not in DRIVE_NATIVE_FILES]
    sidecar_updates = [a for a in report.actions if a.kind in ("added", "modified") and a.source_path in DRIVE_NATIVE_FILES]
    manifest_only = [a for a in report.actions if a.kind in ("converged", "runtime_advanced")]

    for action in to_copy + sidecar_updates:
        src = _safe_relative_path(drive_src, action.source_path)
        if src.suffix == ".py":
            try:
                compile(src.read_text(encoding="utf-8"), str(src), "exec")
            except SyntaxError as exc:
                report.errors.append(f"Python syntax error in {action.source_path}: {exc}")
        elif src.suffix == ".json":
            try:
                parsed = json.loads(src.read_text(encoding="utf-8"))
                if action.source_path == "config/model-capabilities.json":
                    report.errors.extend(validate_capabilities_sidecar(parsed))
            except json.JSONDecodeError as exc:
                report.errors.append(f"Invalid JSON in {action.source_path}: {exc}")
    if report.errors:
        return report

    if not to_copy and not sidecar_updates and not manifest_only:
        return report

    backups: list[tuple[Path, Optional[bytes]]] = []
    manifest = load_manifest(manifest_path)
    files_meta: dict[str, Any] = manifest.setdefault("files", {})
    now = _utc_now()

    try:
        for action in to_copy:
            src = _safe_relative_path(drive_src, action.source_path)
            dst = _safe_relative_path(runtime_root, action.runtime_path)
            backups.append((dst, dst.read_bytes() if dst.is_file() else None))
            _atomic_write_bytes(dst, src.read_bytes())
            new_hash = sha256_file(dst)
            files_meta[action.source_path] = asdict(
                ManifestEntry(
                    source_path=action.source_path,
                    runtime_path=action.runtime_path,
                    last_common_hash=new_hash,
                    observed_runtime_hash=None,
                    last_synced_at=now,
                    git_commit=git_commit,
                    status="synced",
                )
            )
            report.applied.append(action.runtime_path)

        for action in sidecar_updates:
            files_meta[action.source_path] = asdict(
                ManifestEntry(
                    source_path=action.source_path,
                    runtime_path=action.source_path,
                    last_common_hash=action.drive_hash or "",
                    observed_runtime_hash=None,
                    last_synced_at=now,
                    git_commit=git_commit,
                    status="sidecar_synced",
                )
            )
            report.applied.append(action.source_path)

        for action in manifest_only:
            if action.kind == "converged":
                files_meta[action.source_path] = asdict(
                    ManifestEntry(
                        source_path=action.source_path,
                        runtime_path=action.runtime_path,
                        last_common_hash=action.drive_hash or "",
                        observed_runtime_hash=None,
                        last_synced_at=now,
                        git_commit=git_commit,
                        status="converged",
                    )
                )
            elif action.kind == "runtime_advanced":
                # Preserve immutable last_common_hash; record observed_runtime_hash separately!
                files_meta[action.source_path] = asdict(
                    ManifestEntry(
                        source_path=action.source_path,
                        runtime_path=action.runtime_path,
                        last_common_hash=action.last_common_hash or "",
                        observed_runtime_hash=action.runtime_hash,
                        last_synced_at=now,
                        git_commit=git_commit,
                        status="runtime_preserved",
                    )
                )
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

    manifest["version"] = 2
    manifest["git_commit"] = git_commit
    manifest["updated_at"] = now
    _atomic_write_json(manifest_path, manifest)
    return report


def compute_build_fingerprint(
    runtime_root: Path,
    drive_src: Optional[Path] = None,
    *,
    git_commit: str = "",
) -> str:
    """
    Deterministic SHA-256 fingerprint covering Git SHA, root manifests/lockfile,
    all build-affecting Nexus/CLI/web-app source trees, Drive overrides, and
    uncommitted working-tree modifications.
    """
    h = hashlib.sha256()
    h.update(f"git:{git_commit}\n".encode("utf-8"))

    def _hash_path(label: str, path: Path) -> None:
        if path.is_file():
            h.update(f"file:{label}:{sha256_file(path)}\n".encode("utf-8"))

    for rel in BUILD_FINGERPRINT_ROOT_FILES:
        _hash_path(rel, runtime_root / rel)

    for rel_dir in BUILD_FINGERPRINT_DIRS:
        base = runtime_root / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(base).parts
            if any(seg in ("node_modules", "lib", "__pycache__", ".pytest_cache") for seg in parts):
                continue
            _hash_path(path.relative_to(runtime_root).as_posix(), path)

    if drive_src is not None and drive_src.is_dir():
        for path in sorted(drive_src.rglob("*")):
            if not path.is_file():
                continue
            if any(seg in IGNORED_NAMES or seg.endswith(".pyc") for seg in path.relative_to(drive_src).parts):
                continue
            _hash_path("drive:" + path.relative_to(drive_src).as_posix(), path)

    # Capture uncommitted working-tree changes outside the explicit directories too.
    if (runtime_root / ".git").exists():
        try:
            porcelain = subprocess.check_output(
                ["git", "-C", str(runtime_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in sorted(porcelain.splitlines()):
                h.update(f"dirty:{line}\n".encode("utf-8"))
                rel = line[3:].strip().strip('"')
                if rel and " -> " not in rel:
                    p = runtime_root / rel
                    if p.is_file():
                        _hash_path(f"dirty-file:{rel}", p)
        except Exception:
            pass

    return h.hexdigest()
