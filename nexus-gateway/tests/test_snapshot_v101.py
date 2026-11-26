"""Unit tests for Stage 6 versioned, secret-excluding Harness-home snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

COLAB_DIR = Path(__file__).resolve().parents[2] / "colab"
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

import harness_home_snapshot as hhs


def test_snapshot_excludes_secrets_and_restores_verified_state(tmp_path: Path) -> None:
    home = tmp_path / "home"
    backup = tmp_path / "drive-backup"

    # Allowed durable state
    (home / "sessions" / "ws1").mkdir(parents=True)
    (home / "sessions" / "ws1" / "session.v4.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
    (home / "attachments").mkdir(parents=True)
    (home / "attachments" / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n1234")

    # Excluded secrets / credentials
    (home / "credentials").mkdir(parents=True)
    (home / "credentials" / "oauth.json").write_text('{"access_token":"secret"}\n', encoding="utf-8")
    (home / "sessions" / "api_token.txt").write_text("leaked", encoding="utf-8")
    (home / ".env").write_text("NEXUS_API_KEY=secret\n", encoding="utf-8")
    (home / "auth-state.json").write_text("secret", encoding="utf-8")

    result = hhs.create_snapshot(home, backup)
    assert result["file_count"] == 2
    snap_dir = Path(result["snapshot_dir"])
    assert not (snap_dir / "credentials").exists()
    assert not (snap_dir / "sessions" / "api_token.txt").exists()
    assert not (snap_dir / ".env").exists()
    assert not (snap_dir / "auth-state.json").exists()
    assert (snap_dir / "snapshot-manifest.json").is_file()

    restored_home = tmp_path / "restored-home"
    restored = hhs.restore_latest_snapshot(backup, restored_home)
    assert restored["restored"] is True
    assert restored["file_count"] == 2
    assert (restored_home / "sessions" / "ws1" / "session.v4.jsonl").read_text(encoding="utf-8") == '{"seq":1}\n'
    assert (restored_home / "attachments" / "img1.png").read_bytes() == b"\x89PNG\r\n\x1a\n1234"
    assert not (restored_home / "credentials").exists()


def test_restore_falls_back_to_prior_valid_snapshot_when_newest_corrupted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    backup = tmp_path / "drive-backup"
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "s1.jsonl").write_text("v1\n", encoding="utf-8")

    snap1 = hhs.create_snapshot(home, backup)

    # Create a newer but incomplete/corrupted snapshot directory
    corrupt_dir = backup / "snapshots" / "99991231T235959Z_corrupt"
    (corrupt_dir / "sessions").mkdir(parents=True)
    (corrupt_dir / "sessions" / "s1.jsonl").write_text("corrupt", encoding="utf-8")
    # Missing snapshot-manifest.json -> must be ignored

    restored_home = tmp_path / "restored"
    restored = hhs.restore_latest_snapshot(backup, restored_home)
    assert restored["restored"] is True
    assert restored["snapshot_id"] == snap1["snapshot_id"]
    assert (restored_home / "sessions" / "s1.jsonl").read_text(encoding="utf-8") == "v1\n"
