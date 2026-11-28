"""Colab-side Google Drive model inventory, structural validation, and VRAM+RAM fit checks (v2.1.0)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

_GW_SRC = Path(__file__).resolve().parents[1] / "nexus-gateway" / "src"
if str(_GW_SRC) not in sys.path:
    sys.path.insert(0, str(_GW_SRC))

from core.memory_planner import plan_gguf_memory  # noqa: E402
from models.gguf_metadata import (  # noqa: E402
    infer_quantization_from_name,
    inspect_gguf_directory,
    resolve_gguf_implementation,
)

GIB = 1024 ** 3

SUPPORTED_DIFFUSERS_PIPELINES: dict[str, dict[str, Any]] = {
    "Flux2KleinPipeline": {"implementation": "flux2_klein", "default_steps": 6},
    "Flux2KleinKVPipeline": {"implementation": "flux2_klein", "default_steps": 6},
    "StableDiffusionXLPipeline": {"implementation": "sdxl", "default_steps": 30},
}

OPTIONAL_DIFFUSERS_COMPONENTS = frozenset({
    "safety_checker",
    "feature_extractor",
    "image_encoder",
    "watermarker",
})

NON_CHAT_MODEL_TYPES = frozenset({
    "t5", "byt5", "umt5", "mt5", "clip", "clip_text_model", "siglip", "bert", "roberta",
    "autoencoder", "vae", "wan", "hunyuan_video", "hunyuanvideo", "ltx_video", "ltx", "cogvideox", "mochi",
})
VIDEO_OR_ENCODER_NAME_HINTS = ("hunyuanvideo", "wan2", "ltx-", "byt5", "umt5", "t5-v1", "video")

HF_WEIGHT_FILENAME_RE = re.compile(
    r"^(?:model(?:(?:\.safetensors)?-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin|consolidated(?:\.\d+)?\.safetensors)$"
)

SKIPPED_DIR_NAMES = frozenset({".cache", "__pycache__", ".git", ".ipynb_checkpoints"})


@dataclass
class InventoryCandidate:
    id: str
    role: Literal["language", "image", "excluded"]
    source_path: str
    implementation: str = ""
    architecture: str = "unknown"
    size_bytes: int = 0
    min_vram_gib: float = 0.0
    fits_current_gpu: Optional[bool] = None
    modalities: list[str] = field(default_factory=lambda: ["text"])
    context_window: int = 32768
    max_output_tokens: int = 4096
    reasoning_mode: Literal["none", "binary_toggle", "effort_levels"] = "none"
    reasoning_efforts: Any = False
    default_reasoning_effort: Optional[str] = None
    default_steps: int = 6
    quantization: str = "unknown"
    compatibility_tier: Optional[str] = None
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    threads: Optional[int] = None
    can_coreside_image: bool = True
    shard_count: int = 1
    layout: str = "single"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriveInventory:
    language_models: list[InventoryCandidate] = field(default_factory=list)
    image_models: list[InventoryCandidate] = field(default_factory=list)
    excluded_models: list[InventoryCandidate] = field(default_factory=list)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIPPED_DIR_NAMES for part in p.relative_to(root).parts):
            continue
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _resolve_snapshot_root(model_dir: Path, marker_file: str) -> Optional[Path]:
    if (model_dir / marker_file).is_file():
        return model_dir
    for pattern in (f"snapshots/*/{marker_file}", f"*/snapshots/*/{marker_file}"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0].parent
    return None


def _read_chat_template(model_root: Path) -> str:
    fragments: list[str] = []
    tok_cfg = _read_json(model_root / "tokenizer_config.json") or {}
    template = tok_cfg.get("chat_template")
    if isinstance(template, list):
        fragments.extend(str(item.get("template", "")) for item in template if isinstance(item, dict))
    elif isinstance(template, str):
        fragments.append(template)
    if (model_root / "chat_template.jinja").is_file():
        fragments.append((model_root / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(fragments)


def _validate_hf_weights(model_root: Path) -> tuple[bool, str]:
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_path = model_root / idx_name
        if idx_path.is_file():
            idx_data = _read_json(idx_path)
            if not isinstance(idx_data, dict) or not isinstance(idx_data.get("weight_map"), dict) or not idx_data["weight_map"]:
                return False, f"Malformed shard index {idx_name}"
            shards = sorted({str(fname) for fname in idx_data["weight_map"].values()})
            missing = [fname for fname in shards if not (model_root / fname).is_file()]
            if missing:
                return False, f"Incomplete shards ({len(missing)}/{len(shards)} missing, e.g. {missing[0]})"
            return True, f"All {len(shards)} indexed shard(s) verified"
    known = [p.name for p in model_root.iterdir() if p.is_file() and HF_WEIGHT_FILENAME_RE.match(p.name)]
    if known:
        return True, f"Unsharded weight file verified ({known[0]})"
    return False, "Missing Hugging Face safetensors/bin weights"


def _validate_diffusers_pipeline(diffusers_root: Path, model_index: dict[str, Any]) -> tuple[bool, str, str, int]:
    class_name = str(model_index.get("_class_name") or "").strip()
    meta = SUPPORTED_DIFFUSERS_PIPELINES.get(class_name)
    if meta is None:
        return False, f"Unsupported Diffusers pipeline class {class_name!r}", "", 0

    required_components: list[str] = []
    for key, spec in model_index.items():
        if key.startswith("_") or key in OPTIONAL_DIFFUSERS_COMPONENTS:
            continue
        if not isinstance(spec, list) or len(spec) != 2:
            continue
        if all(item is None for item in spec):
            continue
        if not all(isinstance(item, str) and item.strip() for item in spec):
            continue
        required_components.append(key)

    if not required_components:
        return False, "model_index.json declares no pipeline components", "", 0

    for comp in required_components:
        comp_dir = diffusers_root / comp
        if not comp_dir.is_dir() or not any(p.is_file() for p in comp_dir.iterdir()):
            return False, f"Missing or empty Diffusers component '{comp}'", "", 0

    if importlib.util.find_spec("diffusers") is not None:
        try:
            import diffusers  # type: ignore[import-not-found]

            if not hasattr(diffusers, class_name):
                return False, f"Installed diffusers does not export {class_name}", "", 0
        except Exception as exc:
            return False, f"Could not verify diffusers export for {class_name}: {exc}", "", 0

    return True, f"Verified {class_name}", str(meta["implementation"]), int(meta["default_steps"])


def _estimate_min_vram_gib(size_bytes: int, role: str, implementation: str, model_id: str) -> float:
    size_gib = size_bytes / GIB
    if role == "image":
        if implementation == "flux2_klein" and "9b" in model_id.lower():
            return 20.0
        return round(max(6.0, size_gib * 1.15), 1)
    return round(max(6.0, size_gib * 1.12), 1)


def inspect_single_drive_dir(
    model_dir: Path,
    *,
    total_vram_bytes: Optional[int] = None,
    available_ram_bytes: int = 0,
    cpu_count: int = 4,
) -> InventoryCandidate:
    model_id = model_dir.name
    vram_gib = (total_vram_bytes / GIB) if (total_vram_bytes is not None and total_vram_bytes > 0) else None

    # 1. Diffusers image-generation pipeline?
    diffusers_root = _resolve_snapshot_root(model_dir, "model_index.json")
    if diffusers_root is not None:
        model_index = _read_json(diffusers_root / "model_index.json") or {}
        class_name = str(model_index.get("_class_name") or "unknown")
        ok, reason, impl, default_steps = _validate_diffusers_pipeline(diffusers_root, model_index)
        size_bytes = _dir_size_bytes(diffusers_root)
        if not ok:
            return InventoryCandidate(
                id=model_id,
                role="excluded",
                source_path=str(diffusers_root),
                architecture=class_name,
                size_bytes=size_bytes,
                reason=reason,
            )
        min_vram = _estimate_min_vram_gib(size_bytes, "image", impl, model_id)
        return InventoryCandidate(
            id=model_id,
            role="image",
            source_path=str(diffusers_root),
            implementation=impl,
            architecture=class_name,
            size_bytes=size_bytes,
            min_vram_gib=min_vram,
            fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
            modalities=["image"],
            default_steps=default_steps,
            reason=reason,
        )

    # 2. Hugging Face Transformers language / VLM model?
    hf_root = _resolve_snapshot_root(model_dir, "config.json")
    if hf_root is not None:
        config = _read_json(hf_root / "config.json") or {}
        model_type = str(config.get("model_type") or "unknown").lower()
        archs = config.get("architectures") or []
        arch = str(archs[0]) if isinstance(archs, list) and archs else model_type
        low_id = model_id.lower()
        low_arch = arch.lower()
        size_bytes = _dir_size_bytes(hf_root)

        if (
            model_type in NON_CHAT_MODEL_TYPES
            or any(hint in low_id for hint in VIDEO_OR_ENCODER_NAME_HINTS)
            or any(tag in low_arch for tag in ("video", "t5encoder", "cliptext", "autoencoder", "vae", "transformer3d"))
        ):
            return InventoryCandidate(
                id=model_id,
                role="excluded",
                source_path=str(hf_root),
                architecture=arch,
                size_bytes=size_bytes,
                reason=f"Non-chat video or encoder checkpoint ({model_type}/{arch})",
            )

        weights_ok, weights_msg = _validate_hf_weights(hf_root)
        if not weights_ok:
            return InventoryCandidate(
                id=model_id,
                role="excluded",
                source_path=str(hf_root),
                architecture=arch,
                size_bytes=size_bytes,
                reason=weights_msg,
            )

        template = _read_chat_template(hf_root)
        low_tpl = template.lower()
        text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        ctx_raw = config.get("max_position_embeddings") or text_cfg.get("max_position_embeddings")
        ctx_len = int(ctx_raw) if isinstance(ctx_raw, int) and ctx_raw > 0 else 32768

        if model_type in ("qwen3_vl", "qwen2_5_vl", "qwen2_vl"):
            if "tools" not in low_tpl or "tool_call" not in low_tpl:
                return InventoryCandidate(
                    id=model_id,
                    role="excluded",
                    source_path=str(hf_root),
                    architecture=arch,
                    size_bytes=size_bytes,
                    reason="Missing native tool_call support in chat_template",
                )
            min_vram = _estimate_min_vram_gib(size_bytes, "language", "qwen3_vl", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="qwen3_vl",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text", "image"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="none",
                reasoning_efforts=False,
                default_reasoning_effort=None,
                reason=f"Dedicated Qwen3-VL runtime ({weights_msg})",
            )

        if model_type in ("qwen3_5", "qwen3_5_moe"):
            if "tools" not in low_tpl or "tool_call" not in low_tpl:
                return InventoryCandidate(
                    id=model_id,
                    role="excluded",
                    source_path=str(hf_root),
                    architecture=arch,
                    size_bytes=size_bytes,
                    reason="Missing native tool_call support in chat_template",
                )
            min_vram = _estimate_min_vram_gib(size_bytes, "language", "qwen3_5", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="qwen3_5",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text", "image"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="binary_toggle",
                reasoning_efforts={"off": "off", "high": "high"},
                default_reasoning_effort="high",
                reason=f"Dedicated Qwen3.5/3.6 VLM + Thinking runtime ({weights_msg})",
            )

        if model_type == "gpt_oss":
            if "tools" not in low_tpl or "<|call|>" not in low_tpl:
                return InventoryCandidate(
                    id=model_id,
                    role="excluded",
                    source_path=str(hf_root),
                    architecture=arch,
                    size_bytes=size_bytes,
                    reason="Missing Harmony tool-call template",
                )
            min_vram = 15.0 if size_bytes > 20 * GIB else _estimate_min_vram_gib(size_bytes, "language", "gpt_oss", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="gpt_oss",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="effort_levels",
                reasoning_efforts={"low": "low", "medium": "medium", "high": "high"},
                default_reasoning_effort="medium",
                reason=f"Dedicated GPT-OSS Harmony runtime (4-bit auto-fit on L4, {weights_msg})",
            )

        if model_type == "gemma3":
            min_vram = 14.0 if size_bytes > 20 * GIB else _estimate_min_vram_gib(size_bytes, "language", "gemma3", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="gemma3",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text", "image"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="none",
                reasoning_efforts=False,
                default_reasoning_effort=None,
                reason=f"Dedicated Gemma 3 runtime (4-bit auto-fit on L4, {weights_msg})",
            )

        if model_type in ("deepseek_vl_v2", "deepseek_vlv2", "deepseek_vl2"):
            min_vram = 16.0 if size_bytes > 20 * GIB else _estimate_min_vram_gib(size_bytes, "language", "deepseek_vl2", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="deepseek_vl2",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text", "image"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="none",
                reasoning_efforts=False,
                default_reasoning_effort=None,
                reason=f"Dedicated DeepSeek-VL2 runtime (4-bit auto-fit on L4, {weights_msg})",
            )

        if model_type in ("multi_modality", "deepseek_vl"):
            min_vram = _estimate_min_vram_gib(size_bytes, "language", "deepseek_vl", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="deepseek_vl",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                compatibility_tier="RECOMMENDED" if (vram_gib is not None and vram_gib >= min_vram) else None,
                modalities=["text", "image"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="none",
                reasoning_efforts=False,
                default_reasoning_effort=None,
                reason=f"Dedicated DeepSeek-VL runtime ({weights_msg})",
            )

        return InventoryCandidate(
            id=model_id,
            role="excluded",
            source_path=str(hf_root),
            architecture=arch,
            size_bytes=size_bytes,
            reason=f"No dedicated tool-calling implementation for model_type={model_type!r} (tools_in_template={'tools' in low_tpl})",
        )

    # 3. GGUF package (single, -00001-of-0000N shards, or .gguf.part1ofN split parts)
    gguf_layout = inspect_gguf_directory(model_dir)
    if gguf_layout.valid:
        hdr = gguf_layout.header
        arch = hdr.architecture if hdr and hdr.architecture != "unknown" else model_id.lower()
        quant = hdr.quantization if hdr and hdr.quantization != "unknown" else infer_quantization_from_name(gguf_layout.assembled_filename or model_id)
        impl = resolve_gguf_implementation(arch, model_id)
        mmproj_bytes = gguf_layout.mmproj_file.stat().st_size if gguf_layout.mmproj_file else 0
        total_bytes = gguf_layout.total_size_bytes + mmproj_bytes
        max_ctx = (hdr.context_length if hdr and hdr.context_length else 32768)
        plan = plan_gguf_memory(
            weight_bytes=gguf_layout.total_size_bytes,
            mmproj_bytes=mmproj_bytes,
            architecture=arch,
            block_count=hdr.block_count if hdr else None,
            max_model_ctx=max_ctx,
            total_vram_bytes=total_vram_bytes or 0,
            available_ram_bytes=available_ram_bytes,
            cpu_count=cpu_count,
        )
        min_vram = round(max(6.0, (total_bytes / GIB) * 1.12), 1)
        has_think = (hdr.has_think_template if hdr else False) or impl in ("gguf_qwen3_5", "gguf_deepseek_qwen3_5")
        modalities = ["text", "image"] if gguf_layout.mmproj_file is not None else ["text"]
        return InventoryCandidate(
            id=model_id,
            role="language",
            source_path=str(gguf_layout.root_dir),
            implementation=impl,
            architecture=arch,
            size_bytes=total_bytes,
            min_vram_gib=min_vram,
            fits_current_gpu=(plan.tier != "INSUFFICIENT_MEMORY") if total_vram_bytes else None,
            modalities=modalities,
            context_window=plan.n_ctx,
            max_output_tokens=4096,
            reasoning_mode="binary_toggle" if has_think else "none",
            reasoning_efforts={"off": "off", "high": "high"} if has_think else False,
            default_reasoning_effort="high" if has_think else None,
            quantization=quant,
            compatibility_tier=plan.tier if total_vram_bytes else None,
            n_ctx=plan.n_ctx,
            n_gpu_layers=plan.n_gpu_layers,
            batch_size=plan.batch_size,
            ubatch_size=plan.ubatch_size,
            threads=plan.threads,
            shard_count=len(gguf_layout.source_files),
            layout=gguf_layout.layout,
            reason=f"{gguf_layout.reason} | {plan.reason}",
        )

    # Report explicit GGUF validation failures (e.g., 0-byte downloads or missing split parts).
    if any(model_dir.rglob("*.gguf*")) or (model_dir / ".cache").is_dir():
        return InventoryCandidate(
            id=model_id,
            role="excluded",
            source_path=str(model_dir),
            architecture="gguf",
            size_bytes=_dir_size_bytes(model_dir),
            reason=gguf_layout.reason,
        )

    return InventoryCandidate(
        id=model_id,
        role="excluded",
        source_path=str(model_dir),
        architecture="unknown",
        size_bytes=_dir_size_bytes(model_dir),
        reason="Missing config.json, Diffusers model_index.json, or valid GGUF weights",
    )


def inspect_drive_models(
    models_root: Path,
    *,
    total_vram_bytes: Optional[int] = None,
    available_ram_bytes: int = 0,
    cpu_count: int = 4,
) -> DriveInventory:
    """Scan Drive `models/` once during Colab pre-run inventory and classify every folder."""
    inv = DriveInventory()
    if not models_root.is_dir():
        return inv
    for child in sorted(models_root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        item = inspect_single_drive_dir(
            child,
            total_vram_bytes=total_vram_bytes,
            available_ram_bytes=available_ram_bytes,
            cpu_count=cpu_count,
        )
        if item.role == "language":
            inv.language_models.append(item)
        elif item.role == "image":
            inv.image_models.append(item)
        else:
            inv.excluded_models.append(item)
    return inv


def validate_selected_pair(
    models_root: Path,
    language_model_id: str,
    image_model_id: str = "",
    *,
    total_vram_bytes: Optional[int] = None,
    available_ram_bytes: int = 0,
    cpu_count: int = 4,
    free_disk_bytes: Optional[int] = None,
    min_disk_margin_bytes: int = 2 * GIB,
    enforce_gpu_fit: bool = True,
    n_ctx_override: int = 0,
    n_gpu_layers_override: int = -1,
    batch_size_override: int = 0,
    threads_override: int = 0,
    allow_cpu_offload: bool = True,
    enable_image_coresidency: bool = True,
) -> tuple[InventoryCandidate, Optional[InventoryCandidate]]:
    """
    Validate the single selected language model and optional single image model before staging.
    Recomputes the optimal MemoryPlanner configuration with any user overrides and image co-residency.
    """
    lang_id = language_model_id.strip()
    img_id = image_model_id.strip()
    if not lang_id:
        raise ValueError("SELECTED_LANGUAGE_MODEL is required. Copy one language model name from the Inventory cell.")

    lang_dir = models_root / lang_id
    if not lang_dir.is_dir():
        raise ValueError(f"Selected language model directory not found on Drive: {lang_dir}")
    lang = inspect_single_drive_dir(
        lang_dir,
        total_vram_bytes=total_vram_bytes,
        available_ram_bytes=available_ram_bytes,
        cpu_count=cpu_count,
    )
    if lang.role != "language":
        raise ValueError(f"'{lang_id}' is not a runnable language model: {lang.reason}")

    img: Optional[InventoryCandidate] = None
    if img_id:
        img_dir = models_root / img_id
        if not img_dir.is_dir():
            raise ValueError(f"Selected image model directory not found on Drive: {img_dir}")
        img = inspect_single_drive_dir(
            img_dir,
            total_vram_bytes=total_vram_bytes,
            available_ram_bytes=available_ram_bytes,
            cpu_count=cpu_count,
        )
        if img.role != "image":
            raise ValueError(f"'{img_id}' is not a runnable image-generation model: {img.reason}")
        if enforce_gpu_fit and img.fits_current_gpu is False:
            avail_gib = round((total_vram_bytes or 0) / GIB, 1)
            raise ValueError(
                f"Selected image model '{img_id}' requires ~{img.min_vram_gib:.1f} GiB VRAM, "
                f"but current GPU only has {avail_gib:.1f} GiB."
            )

    if lang.implementation.startswith("gguf_"):
        gguf_layout = inspect_gguf_directory(lang_dir)
        hdr = gguf_layout.header
        mmproj_bytes = gguf_layout.mmproj_file.stat().st_size if gguf_layout.mmproj_file else 0
        plan = plan_gguf_memory(
            weight_bytes=gguf_layout.total_size_bytes,
            mmproj_bytes=mmproj_bytes,
            architecture=lang.architecture,
            block_count=hdr.block_count if hdr else None,
            max_model_ctx=hdr.context_length if hdr and hdr.context_length else 32768,
            total_vram_bytes=total_vram_bytes or 0,
            available_ram_bytes=available_ram_bytes,
            cpu_count=cpu_count,
            image_min_vram_gib=(img.min_vram_gib if (img and enable_image_coresidency) else 0.0),
            n_ctx_override=n_ctx_override,
            n_gpu_layers_override=n_gpu_layers_override,
            batch_size_override=batch_size_override,
            threads_override=threads_override,
            allow_cpu_offload=allow_cpu_offload,
        )
        lang.compatibility_tier = plan.tier
        lang.n_ctx = plan.n_ctx
        lang.context_window = plan.n_ctx
        lang.n_gpu_layers = plan.n_gpu_layers
        lang.batch_size = plan.batch_size
        lang.ubatch_size = plan.ubatch_size
        lang.threads = plan.threads
        lang.can_coreside_image = bool(enable_image_coresidency and plan.can_coreside_image)
        lang.fits_current_gpu = plan.tier != "INSUFFICIENT_MEMORY"
        lang.reason = plan.reason
        if enforce_gpu_fit and plan.tier == "INSUFFICIENT_MEMORY":
            raise ValueError(f"Selected language model '{lang_id}' cannot fit on this runtime: {plan.reason}")
    else:
        lang.can_coreside_image = bool(enable_image_coresidency)
        if enforce_gpu_fit and lang.fits_current_gpu is False:
            avail_gib = round((total_vram_bytes or 0) / GIB, 1)
            raise ValueError(
                f"Selected language model '{lang_id}' requires ~{lang.min_vram_gib:.1f} GiB VRAM, "
                f"but current GPU only has {avail_gib:.1f} GiB. Choose a smaller model or switch Colab GPU tier."
            )

    if free_disk_bytes is not None:
        needed = lang.size_bytes + (img.size_bytes if img is not None else 0) + min_disk_margin_bytes
        if free_disk_bytes < needed:
            raise ValueError(
                f"Insufficient local SSD space for upfront staging: need {needed / GIB:.1f} GiB "
                f"(language={lang.size_bytes / GIB:.1f} GiB + image={(img.size_bytes / GIB) if img else 0.0:.1f} GiB "
                f"+ margin={min_disk_margin_bytes / GIB:.1f} GiB), only {free_disk_bytes / GIB:.1f} GiB free."
            )

    return lang, img


def format_inventory_report(inv: DriveInventory, *, gpu_summary: str = "Unknown GPU") -> str:
    """Format a copy-pasteable inventory report for the Colab Inventory cell."""
    lines = [
        f"=== NEXUS AI LAB v2.3.0 — DRIVE MODEL INVENTORY ({gpu_summary}) ===",
        "",
        f"Selectable Language Models ({len(inv.language_models)}):",
    ]
    if not inv.language_models:
        lines.append("  (none found)")
    for m in inv.language_models:
        tier = m.compatibility_tier or (
            "RECOMMENDED" if m.fits_current_gpu else ("INSUFFICIENT_MEMORY" if m.fits_current_gpu is False else "GPU N/A")
        )
        quant_str = f"{m.quantization}" if m.quantization != "unknown" else "-"
        shard_str = f"{m.shard_count}p" if m.shard_count > 1 else "1f"
        ctx_str = f"ctx={m.n_ctx}" if m.n_ctx else f"ctx={m.context_window}"
        vis_str = "VISION" if "image" in (m.modalities or []) else "TEXT-ONLY"
        lines.append(
            f"  - {m.id:<34} | impl={m.implementation:<14} | {vis_str:<9} | {m.size_bytes / GIB:5.1f} GiB "
            f"| q={quant_str:<7} ({shard_str}) | {ctx_str:<9} | {tier}"
        )

    lines.extend(["", f"Selectable Image Models ({len(inv.image_models)}):"])
    if not inv.image_models:
        lines.append("  (none found)")
    for m in inv.image_models:
        fit = "FITS GPU" if m.fits_current_gpu else ("TOO LARGE FOR GPU" if m.fits_current_gpu is False else "GPU N/A")
        lines.append(
            f"  - {m.id:<34} | impl={m.implementation:<11} | {m.size_bytes / GIB:5.1f} GiB "
            f"| minVRAM={m.min_vram_gib:4.1f} GiB | steps={m.default_steps:<2} | {fit}"
        )

    lines.extend(["", f"Skipped / Incomplete Folders ({len(inv.excluded_models)}):"])
    for m in inv.excluded_models:
        lines.append(f"  - {m.id:<34} | {m.reason}")

    return "\n".join(lines)
