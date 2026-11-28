"""Pure-Python GGUF binary header inspector and split-shard layout resolver."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional

GGUF_MAGIC = b"GGUF"
MIN_VALID_GGUF_BYTES = 64 * 1024 * 1024  # 64 MiB minimum for a real language checkpoint

SPLIT_SHARD_RE = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
RAW_PART_RE = re.compile(r"^(.+\.gguf)\.part(\d+)of(\d+)$", re.IGNORECASE)
AUXILIARY_GGUF_RE = re.compile(r"(?:^|[-_.])(mmproj|mtp|imatrix|draft)(?:[-_.]|$)", re.IGNORECASE)
QUANT_NAME_RE = re.compile(r"\b(IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+|F16|BF16|F32|MXFP4)\b", re.IGNORECASE)

# Subset of ggml_ftype / llama_ftype codes for human-readable reporting.
GGUF_FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
}

# GGUF metadata value type IDs (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
_SCALAR_STRUCT: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}


@dataclass
class GgufHeaderInfo:
    architecture: str = "unknown"
    name: str = ""
    quantization: str = "unknown"
    file_type: Optional[int] = None
    block_count: Optional[int] = None
    context_length: Optional[int] = None
    has_think_template: bool = False
    template_has_tools: bool = False
    template_has_ipython_compulsion: bool = False
    template_style: str = "unknown"
    tensor_count: int = 0
    version: int = 0


@dataclass
class GgufDirectoryLayout:
    valid: bool
    reason: str
    layout: Literal["single", "gguf_split", "raw_parts", "invalid"]
    root_dir: Path
    source_files: list[Path] = field(default_factory=list)
    assembled_filename: str = ""
    mmproj_file: Optional[Path] = None
    total_size_bytes: int = 0
    header: Optional[GgufHeaderInfo] = None


def _read_exact(fh: BinaryIO, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise EOFError("Unexpected EOF reading GGUF header")
    return data


def _read_gguf_string(fh: BinaryIO, *, max_len: int = 1_048_576) -> str:
    (length,) = struct.unpack("<Q", _read_exact(fh, 8))
    if length > max_len:
        # Skip huge metadata strings cleanly without allocating large buffers.
        fh.seek(length, 1)
        return ""
    return _read_exact(fh, int(length)).decode("utf-8", errors="replace")


def _skip_gguf_value(fh: BinaryIO, vtype: int) -> None:
    if vtype in _SCALAR_STRUCT:
        _, width = _SCALAR_STRUCT[vtype]
        fh.seek(width, 1)
        return
    if vtype == 8:  # STRING
        (length,) = struct.unpack("<Q", _read_exact(fh, 8))
        fh.seek(int(length), 1)
        return
    if vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", _read_exact(fh, 12))
        if elem_type in _SCALAR_STRUCT:
            _, width = _SCALAR_STRUCT[elem_type]
            fh.seek(width * int(count), 1)
            return
        for _ in range(int(count)):
            _skip_gguf_value(fh, elem_type)
        return
    raise ValueError(f"Unknown GGUF metadata type {vtype}")


def _read_gguf_scalar_or_str(fh: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALAR_STRUCT:
        fmt, width = _SCALAR_STRUCT[vtype]
        return struct.unpack(fmt, _read_exact(fh, width))[0]
    if vtype == 8:
        return _read_gguf_string(fh)
    _skip_gguf_value(fh, vtype)
    return None


def has_gguf_magic(path: Path) -> bool:
    """Return True if `path` is a readable file starting with the 4-byte `GGUF` magic."""
    try:
        with path.open("rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def infer_quantization_from_name(name: str) -> str:
    m = QUANT_NAME_RE.search(name)
    return m.group(1).upper() if m else "unknown"


def _classify_template(tpl: str) -> tuple[bool, bool, str]:
    low = (tpl or "").lower()
    has_tools = bool(
        "tools" in low
        and (
            "tool_calls" in low
            or "tool_call" in low
            or "python_tag" in low
            or "python_start" in low
            or "start_function_call" in low
        )
    )
    has_ipython_compulsion = "environment: ipython" in low
    if "<|header_start|>" in tpl or "<|python_start|>" in tpl:
        style = "llama4"
    elif "<|start_header_id|>" in tpl or "<|eot_id|>" in tpl:
        style = "llama3"
    elif "<|im_start|>" in tpl or "<|im_end|>" in tpl:
        style = "chatml"
    elif "<start_of_turn>" in tpl or "<end_of_turn>" in tpl:
        style = "gemma"
    else:
        style = "unknown"
    return has_tools, has_ipython_compulsion, style


def read_gguf_header(path: Path, *, max_kv_scan: int = 256) -> Optional[GgufHeaderInfo]:
    """Read core GGUF metadata keys from `path` in milliseconds without loading weights."""
    try:
        with path.open("rb") as fh:
            if fh.read(4) != GGUF_MAGIC:
                return None
            version, tensor_count, kv_count = struct.unpack("<IQQ", _read_exact(fh, 20))
            if version not in (2, 3):
                return None

            raw_meta: dict[str, Any] = {}
            wanted_suffixes = (".block_count", ".context_length")
            wanted_exact = {
                "general.architecture",
                "general.name",
                "general.file_type",
                "tokenizer.chat_template",
            }
            for _ in range(min(int(kv_count), max_kv_scan)):
                key = _read_gguf_string(fh, max_len=4096)
                (vtype,) = struct.unpack("<I", _read_exact(fh, 4))
                if key in wanted_exact or key.endswith(wanted_suffixes):
                    val = _read_gguf_scalar_or_str(fh, vtype)
                    if val is not None:
                        raw_meta[key] = val
                else:
                    _skip_gguf_value(fh, vtype)

            arch = str(raw_meta.get("general.architecture") or "unknown").lower()
            ftype = raw_meta.get("general.file_type")
            ftype_int = int(ftype) if isinstance(ftype, int) else None
            quant = GGUF_FILE_TYPE_NAMES.get(ftype_int, "") if ftype_int is not None else ""
            if not quant:
                quant = infer_quantization_from_name(path.name)

            block_count = raw_meta.get(f"{arch}.block_count")
            ctx_len = raw_meta.get(f"{arch}.context_length")
            tpl = str(raw_meta.get("tokenizer.chat_template") or "")
            tpl_has_tools, tpl_ipython, tpl_style = _classify_template(tpl)

            return GgufHeaderInfo(
                architecture=arch,
                name=str(raw_meta.get("general.name") or ""),
                quantization=quant,
                file_type=ftype_int,
                block_count=int(block_count) if isinstance(block_count, int) and block_count > 0 else None,
                context_length=int(ctx_len) if isinstance(ctx_len, int) and ctx_len > 0 else None,
                has_think_template="<think>" in tpl.lower(),
                template_has_tools=tpl_has_tools,
                template_has_ipython_compulsion=tpl_ipython,
                template_style=tpl_style,
                tensor_count=int(tensor_count),
                version=int(version),
            )
    except Exception:
        return None


def resolve_gguf_implementation(architecture: str, model_id: str = "") -> str:
    """Map a GGUF `general.architecture` (with `model_id` fallback) to a Nexus implementation."""
    arch = (architecture or "").strip().lower()
    low_id = (model_id or "").strip().lower()

    if "deepseek" in low_id and ("qwen" in low_id or arch in ("qwen35", "qwen35moe", "qwen3", "qwen2")):
        return "gguf_deepseek_qwen3_5"
    if arch in ("qwen35", "qwen35moe", "qwen3_5", "qwen3_5_moe"):
        return "gguf_qwen3_5"
    if arch == "llama4":
        return "gguf_llama4"
    if arch == "llama":
        return "gguf_llama3"
    if arch in ("gemma4", "gemma3", "gemma2", "gemma"):
        return "gguf_gemma4"

    # Fallback on folder naming if header architecture was unavailable.
    if "gemma" in low_id:
        return "gguf_gemma4"
    if "llama-4" in low_id or "llama4" in low_id or "scout" in low_id:
        return "gguf_llama4"
    if "llama" in low_id:
        return "gguf_llama3"
    if "qwen3.5" in low_id or "qwen3_5" in low_id or "qwen35" in low_id:
        return "gguf_qwen3_5"
    return "gguf_generic"


def _is_auxiliary_gguf(name: str) -> bool:
    return AUXILIARY_GGUF_RE.search(name) is not None


def _pick_search_root(model_dir: Path) -> Path:
    if any(p.is_file() for p in model_dir.iterdir()):
        return model_dir
    for pat in ("snapshots/*", "*/snapshots/*"):
        for cand in sorted(model_dir.glob(pat)):
            if cand.is_dir() and any(p.is_file() for p in cand.iterdir()):
                return cand
    return model_dir


def inspect_gguf_directory(model_dir: Path) -> GgufDirectoryLayout:
    """
    Inspect a GGUF directory and resolve single `.gguf`, `llama-gguf-split` shards
    (`*-00001-of-0000N.gguf`), or raw byte-split parts (`*.gguf.part1ofN`).
    Rejects 0-byte / truncated / incomplete downloads cleanly.
    """
    if not model_dir.is_dir():
        return GgufDirectoryLayout(
            valid=False,
            reason=f"Directory does not exist: {model_dir}",
            layout="invalid",
            root_dir=model_dir,
        )

    root = _pick_search_root(model_dir)
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and not p.name.endswith(".copying") and not p.name.startswith(".")
    )
    mmproj_candidates = [
        p
        for p in files
        if p.name.lower().endswith(".gguf")
        and any(tag in p.name.lower() for tag in ("mmproj", "vision-projector", "projector"))
        and p.stat().st_size > 0
    ]
    if not mmproj_candidates and root != model_dir:
        mmproj_candidates = sorted(
            p
            for p in model_dir.rglob("*.gguf")
            if p.is_file()
            and any(tag in p.name.lower() for tag in ("mmproj", "vision-projector", "projector"))
            and p.stat().st_size > 0
        )
    mmproj_file = mmproj_candidates[0] if mmproj_candidates else None

    # 1. Check raw byte-split parts (`<name>.gguf.part1ofN` ... `.partNofN`)
    raw_groups: dict[str, dict[int, tuple[int, Path]]] = {}
    for p in files:
        m = RAW_PART_RE.match(p.name)
        if not m or _is_auxiliary_gguf(m.group(1)):
            continue
        base_name = m.group(1)
        idx, total = int(m.group(2)), int(m.group(3))
        raw_groups.setdefault(base_name, {})[idx] = (total, p)

    if raw_groups:
        base_name, parts_map = sorted(raw_groups.items(), key=lambda kv: sum(x[1].stat().st_size for x in kv[1].values()), reverse=True)[0]
        expected_total = next(iter(parts_map.values()))[0]
        missing = [i for i in range(1, expected_total + 1) if i not in parts_map]
        if missing:
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Incomplete split GGUF '{base_name}': missing part(s) {missing} of {expected_total}",
                layout="invalid",
                root_dir=root,
            )
        ordered = [parts_map[i][1] for i in range(1, expected_total + 1)]
        total_bytes = sum(p.stat().st_size for p in ordered)
        if any(p.stat().st_size == 0 for p in ordered) or total_bytes < MIN_VALID_GGUF_BYTES:
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Incomplete or 0-byte split GGUF '{base_name}' ({total_bytes / (1024**2):.1f} MiB)",
                layout="invalid",
                root_dir=root,
            )
        if not has_gguf_magic(ordered[0]):
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Corrupted GGUF header in {ordered[0].name} (missing 'GGUF' magic)",
                layout="invalid",
                root_dir=root,
            )
        header = read_gguf_header(ordered[0])
        return GgufDirectoryLayout(
            valid=True,
            reason=f"Verified {expected_total}-part split GGUF ({base_name})",
            layout="raw_parts",
            root_dir=root,
            source_files=ordered,
            assembled_filename=base_name,
            mmproj_file=mmproj_file,
            total_size_bytes=total_bytes,
            header=header,
        )

    # 2. Check `llama-gguf-split` shards (`<stem>-00001-of-0000N.gguf`)
    shard_groups: dict[str, dict[int, tuple[int, Path]]] = {}
    for p in files:
        m = SPLIT_SHARD_RE.match(p.name)
        if not m or _is_auxiliary_gguf(p.name):
            continue
        stem = m.group(1)
        idx, total = int(m.group(2)), int(m.group(3))
        shard_groups.setdefault(stem, {})[idx] = (total, p)

    if shard_groups:
        stem, shards_map = sorted(shard_groups.items(), key=lambda kv: sum(x[1].stat().st_size for x in kv[1].values()), reverse=True)[0]
        expected_total = next(iter(shards_map.values()))[0]
        missing = [i for i in range(1, expected_total + 1) if i not in shards_map]
        if missing:
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Incomplete sharded GGUF '{stem}': missing shard(s) {missing} of {expected_total}",
                layout="invalid",
                root_dir=root,
            )
        ordered = [shards_map[i][1] for i in range(1, expected_total + 1)]
        total_bytes = sum(p.stat().st_size for p in ordered)
        if any(p.stat().st_size == 0 for p in ordered) or total_bytes < MIN_VALID_GGUF_BYTES:
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Incomplete or 0-byte sharded GGUF '{stem}' ({total_bytes / (1024**2):.1f} MiB)",
                layout="invalid",
                root_dir=root,
            )
        if not has_gguf_magic(ordered[0]):
            return GgufDirectoryLayout(
                valid=False,
                reason=f"Corrupted GGUF header in {ordered[0].name} (missing 'GGUF' magic)",
                layout="invalid",
                root_dir=root,
            )
        header = read_gguf_header(ordered[0])
        return GgufDirectoryLayout(
            valid=True,
            reason=f"Verified {expected_total}-shard GGUF ({ordered[0].name})",
            layout="gguf_split",
            root_dir=root,
            source_files=ordered,
            assembled_filename=ordered[0].name,
            mmproj_file=mmproj_file,
            total_size_bytes=total_bytes,
            header=header,
        )

    # 3. Single primary `.gguf` file
    primary_candidates = [p for p in files if p.name.lower().endswith(".gguf") and not _is_auxiliary_gguf(p.name)]
    if not primary_candidates:
        return GgufDirectoryLayout(
            valid=False,
            reason="No primary GGUF weights found (incomplete download or missing .gguf)",
            layout="invalid",
            root_dir=root,
        )

    primary = max(primary_candidates, key=lambda p: p.stat().st_size)
    size_bytes = primary.stat().st_size
    if size_bytes < MIN_VALID_GGUF_BYTES:
        return GgufDirectoryLayout(
            valid=False,
            reason=f"Incomplete GGUF download: {primary.name} is {size_bytes} B (expected >= 64 MiB)",
            layout="invalid",
            root_dir=root,
        )
    if not has_gguf_magic(primary):
        return GgufDirectoryLayout(
            valid=False,
            reason=f"Corrupted GGUF file {primary.name} (missing 'GGUF' header)",
            layout="invalid",
            root_dir=root,
        )
    header = read_gguf_header(primary)
    return GgufDirectoryLayout(
        valid=True,
        reason=f"Verified GGUF ({primary.name})",
        layout="single",
        root_dir=root,
        source_files=[primary],
        assembled_filename=primary.name,
        mmproj_file=mmproj_file,
        total_size_bytes=size_bytes,
        header=header,
    )
