"""Stage 1 regression tests for v2 3-way Drive source synchronization and build fingerprinting."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "colab") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "colab"))
import sync_source  # noqa: E402


def _setup_pair(tmp_path: Path, rel_drive: str = "nexus-gateway/src/sample.py", rel_rt: str = "nexus-gateway/src/sample.py", initial: str = "X = 1\n"):
    drive_src = tmp_path / "drive" / "colab-src"
    runtime_root = tmp_path / "runtime"
    manifest = drive_src / "sync-manifest.json"
    rt_file = runtime_root / rel_rt
    rt_file.parent.mkdir(parents=True, exist_ok=True)
    rt_file.write_text(initial, encoding="utf-8")
    sync_source.seed_drive_baseline(
        runtime_root, drive_src, manifest, git_commit="commit1", rel_pairs=[(rel_drive, rel_rt)]
    )
    return drive_src, runtime_root, manifest, drive_src / rel_drive, rt_file


def test_1_drive_only_modification(tmp_path: Path) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    dfile.write_text("X = 2\n", encoding="utf-8")
    res = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit1")
    assert res.applied == ["nexus-gateway/src/sample.py"]
    assert rfile.read_text(encoding="utf-8") == "X = 2\n"
    assert "gateway_api" in res.restart_targets


def test_2_and_3_runtime_only_preserved_idempotent_and_later_drive_edit_conflicts(tmp_path: Path) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    # Upstream runtime changes; Drive stays at last_common_hash.
    rfile.write_text("X = 99\n", encoding="utf-8")
    first = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert first.applied == []
    assert first.conflicts == []
    assert rfile.read_text(encoding="utf-8") == "X = 99\n"

    # Re-running sync must be an idempotent no-op with zero false changes.
    second = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert second.applied == []
    assert second.conflicts == []
    assert all(a.kind == "unchanged" for a in second.actions)

    # Now if Drive is later edited, last_common_hash was kept immutable -> MUST detect conflict!
    dfile.write_text("X = 42\n", encoding="utf-8")
    third = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert third.conflicts == ["nexus-gateway/src/sample.py"]
    assert third.errors
    assert rfile.read_text(encoding="utf-8") == "X = 99\n"
    assert dfile.read_text(encoding="utf-8") == "X = 42\n"


def test_4_both_sides_changed_differently_is_conflict(tmp_path: Path) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    dfile.write_text("X = 10\n", encoding="utf-8")
    rfile.write_text("X = 20\n", encoding="utf-8")
    res = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert res.conflicts == ["nexus-gateway/src/sample.py"]
    assert res.errors
    assert rfile.read_text(encoding="utf-8") == "X = 20\n"


def test_5_both_sides_converged_to_identical_content(tmp_path: Path) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    dfile.write_text("X = 77\n", encoding="utf-8")
    rfile.write_text("X = 77\n", encoding="utf-8")
    res = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert res.conflicts == []
    assert any(a.kind == "converged" for a in res.actions)
    again = sync_source.apply_sync(drive_src, rt, manifest, git_commit="commit2")
    assert all(a.kind == "unchanged" for a in again.actions)


def test_6_untracked_existing_drive_file_is_adoption_conflict_until_explicitly_adopted(tmp_path: Path) -> None:
    drive_src = tmp_path / "drive" / "colab-src"
    rt = tmp_path / "runtime"
    manifest = drive_src / "sync-manifest.json"
    dfile = drive_src / "supervisor" / "supervisor.mjs"
    rfile = rt / "colab" / "supervisor.mjs"
    dfile.parent.mkdir(parents=True)
    rfile.parent.mkdir(parents=True)
    dfile.write_text("// drive version\n", encoding="utf-8")
    rfile.write_text("// runtime version\n", encoding="utf-8")

    res = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert res.adoption_conflicts == ["supervisor/supervisor.mjs"]
    assert res.errors
    assert rfile.read_text(encoding="utf-8") == "// runtime version\n"

    # Explicit adoption choosing runtime brings both into sync.
    sync_source.adopt_baseline(drive_src, rt, manifest, "supervisor/supervisor.mjs", prefer="runtime", git_commit="c1")
    assert dfile.read_text(encoding="utf-8") == "// runtime version\n"
    clean = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert clean.conflicts == []
    assert clean.applied == []


def test_7_and_8_new_file_and_drive_deletion_preserves_runtime(tmp_path: Path) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    added = drive_src / "nexus-gateway" / "src" / "extra.py"
    added.write_text("Y = 1\n", encoding="utf-8")
    res = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert "nexus-gateway/src/extra.py" in res.applied
    assert (rt / "nexus-gateway" / "src" / "extra.py").is_file()

    # Delete from Drive -> runtime file MUST be preserved.
    added.unlink()
    after_del = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert any(a.kind == "deleted" and a.source_path == "nexus-gateway/src/extra.py" for a in after_del.actions)
    assert (rt / "nexus-gateway" / "src" / "extra.py").is_file()


def test_9_and_10_plugin_index_mapping_and_capability_sidecar_handling(tmp_path: Path) -> None:
    assert sync_source.map_drive_to_runtime_rel("config/plugin-index.json") == "colab/plugin-index.json"
    assert sync_source.map_drive_to_runtime_rel("config/colab.overlay.yml") == "apps/cli/config/examples/colab.overlay.yml"
    assert sync_source.map_drive_to_runtime_rel("config/model-capabilities.json") is None

    drive_cfg = tmp_path / "drive" / "colab-src" / "config"
    canonical = _REPO_ROOT / "colab" / "model-capabilities.example.json"
    init_res = sync_source.ensure_capabilities_sidecar(drive_cfg, canonical)
    assert init_res["created"] is True
    assert init_res["loaded"] is True
    # Canonical example must NOT default tool_behavior_verified to True.
    parsed = json.loads((drive_cfg / "model-capabilities.json").read_text(encoding="utf-8"))
    for model_cfg in parsed["models"].values():
        assert model_cfg.get("tool_behavior_verified") is not True


def test_11_and_12_invalid_syntax_rejected_and_multi_file_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drive_src, rt, manifest, dfile, rfile = _setup_pair(tmp_path)
    dfile.write_text("def broken(:\n", encoding="utf-8")
    bad = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert bad.errors
    assert rfile.read_text(encoding="utf-8") == "X = 1\n"

    # Rollback when second file write fails mid-batch.
    dfile.write_text("X = 2\n", encoding="utf-8")
    second = drive_src / "nexus-gateway" / "src" / "z_second.py"
    second.write_text("Z = 2\n", encoding="utf-8")
    real_write = sync_source._atomic_write_bytes
    calls = {"n": 0}

    def flaky_write(dest: Path, content: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated disk error")
        real_write(dest, content)

    monkeypatch.setattr(sync_source, "_atomic_write_bytes", flaky_write)
    rolled = sync_source.apply_sync(drive_src, rt, manifest, git_commit="c1")
    assert rolled.rolled_back is True
    assert rfile.read_text(encoding="utf-8") == "X = 1\n"


def test_13_and_14_build_fingerprint_and_module_reload(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    (rt / "apps" / "cli").mkdir(parents=True)
    (rt / "apps" / "cli" / "package.json").write_text('{"name":"cli"}\n', encoding="utf-8")
    drive_src = tmp_path / "drive" / "colab-src"
    (drive_src / "config").mkdir(parents=True)
    (drive_src / "config" / "colab.overlay.yml").write_text("a: 1\n", encoding="utf-8")

    fp1 = sync_source.compute_build_fingerprint(rt, drive_src, git_commit="same-sha")
    (drive_src / "config" / "colab.overlay.yml").write_text("a: 2\n", encoding="utf-8")
    fp2 = sync_source.compute_build_fingerprint(rt, drive_src, git_commit="same-sha")
    assert fp1 != fp2

    # Reloading sync_source module succeeds cleanly.
    reloaded = importlib.reload(sync_source)
    assert hasattr(reloaded, "apply_sync")
