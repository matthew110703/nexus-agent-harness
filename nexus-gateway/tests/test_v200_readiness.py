"""Unit tests for v2.0.0 readiness: build cache, prompt_tools flattening, and all 9 language implementations."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT / "colab", _REPO_ROOT / "nexus-gateway" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_cache  # noqa: E402
from adapters.tool_calls import create_tool_parser, has_native_tool_parser  # noqa: E402
from models.base import LANGUAGE_IMPLEMENTATIONS, load_dedicated_language_runtime  # noqa: E402
from models.language.prompt_tools import extract_text_content, inject_tools_into_messages  # noqa: E402


def test_build_cache_save_restore_and_prune(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    pkg_lib = harness / "packages" / "experimental" / "nexus-host" / "lib"
    web_dist = harness / "apps" / "web" / "dist"
    pkg_lib.mkdir(parents=True)
    web_dist.mkdir(parents=True)
    (pkg_lib / "index.js").write_text("export const ok = true\n", encoding="utf-8")
    (web_dist / "index.html").write_text("<html>ok</html>\n", encoding="utf-8")

    cache_dir = tmp_path / "drive-cache"
    archive = build_cache.save_build_cache(harness, cache_dir, "aaaaaaaaaaaa1111", keep_latest=2)
    assert archive is not None and archive.is_file()

    # Restore into a fresh harness checkout
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert build_cache.restore_build_cache(fresh, cache_dir, "aaaaaaaaaaaa1111") is True
    assert (fresh / "packages" / "experimental" / "nexus-host" / "lib" / "index.js").read_text(encoding="utf-8") == "export const ok = true\n"
    assert (fresh / "apps" / "web" / "dist" / "index.html").read_text(encoding="utf-8") == "<html>ok</html>\n"

    # Prune keeps only latest 2
    build_cache.save_build_cache(harness, cache_dir, "bbbbbbbbbbbb2222", keep_latest=2)
    build_cache.save_build_cache(harness, cache_dir, "cccccccccccc3333", keep_latest=2)
    archives = list(cache_dir.glob("artifacts-*.tar.gz"))
    assert len(archives) == 2


def test_prompt_tools_flattens_structured_content_blocks() -> None:
    blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    assert extract_text_content(blocks) == "Hello\nWorld"
    msgs = inject_tools_into_messages(
        [{"role": "user", "content": blocks}],
        [{"type": "function", "function": {"name": "nexus_generate_image", "parameters": {"type": "object"}}}],
        strict_alternation=True,
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Hello\nWorld" in msgs[0]["content"]
    assert "[{'type'" not in msgs[0]["content"]


def test_all_language_implementations_load_and_have_parsers(tmp_path: Path) -> None:
    for impl in LANGUAGE_IMPLEMENTATIONS:
        assert has_native_tool_parser(impl), f"Missing tool parser for {impl}"
        parser = create_tool_parser(impl)
        assert parser is not None
        rt = load_dedicated_language_runtime(implementation=impl, model_id=f"test-{impl}", model_path=tmp_path)
        assert hasattr(rt, "load") and hasattr(rt, "generate_stream") and hasattr(rt, "unload")
