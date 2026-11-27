"""Colab-side Google Drive model inventory, structural validation, and GPU VRAM fit checks (v1.1.0)."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

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
            # Runnable on L4 24GB via model CPU offload when sole GPU resident.
            return 20.0
        return round(max(6.0, size_gib * 1.15), 1)
    # Language models need weights + KV cache headroom (~1.12x weights, min 6 GiB).
    return round(max(6.0, size_gib * 1.12), 1)


def inspect_single_drive_dir(model_dir: Path, *, total_vram_bytes: Optional[int] = None) -> InventoryCandidate:
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
            min_vram = _estimate_min_vram_gib(size_bytes, "language", "gpt_oss", model_id)
            return InventoryCandidate(
                id=model_id,
                role="language",
                source_path=str(hf_root),
                implementation="gpt_oss",
                architecture=arch,
                size_bytes=size_bytes,
                min_vram_gib=min_vram,
                fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
                modalities=["text"],
                context_window=ctx_len,
                max_output_tokens=4096,
                reasoning_mode="effort_levels",
                reasoning_efforts={"low": "low", "medium": "medium", "high": "high"},
                default_reasoning_effort="medium",
                reason=f"Dedicated GPT-OSS Harmony runtime ({weights_msg})",
            )

        if model_type == "gemma3":
            # Gemma 3 supports tool calling via prompt injection and strict alternation
            # Size on disk ~22.7 GiB, fits in 4-bit NF4 on L4 22 GiB
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
            reason=f"No dedicated v1.2.0 tool-calling implementation for model_type={model_type!r} (tools_in_template={'tools' in low_tpl})",
        )

    # 3. GGUF package (managed via llama-server)
    gguf_files = list(model_dir.glob("*.gguf"))
    if gguf_files:
        low_id = model_id.lower()
        size_bytes = _dir_size_bytes(model_dir)
        # Identify GGUF implementation
        if "gemma" in low_id or "gemma4" in low_id:
            impl = "gguf_gemma4"
            arch = "gemma4"
        elif "deepseek" in low_id:
            impl = "gguf_deepseek_qwen3_5"
            arch = "deepseek_v4_pro"
        else:
            impl = "gguf_qwen3_5"
            arch = "qwen3_5"

        min_vram = round(max(6.0, (size_bytes / GIB) * 1.1), 1)
        return InventoryCandidate(
            id=model_id,
            role="language",
            source_path=str(model_dir),
            implementation=impl,
            architecture=arch,
            size_bytes=size_bytes,
            min_vram_gib=min_vram,
            fits_current_gpu=(vram_gib >= min_vram) if vram_gib is not None else None,
            modalities=["text"],
            context_window=32768,
            max_output_tokens=4096,
            reasoning_mode="none",
            reasoning_efforts=False,
            default_reasoning_effort=None,
            reason="Dedicated GGUF runtime via llama-server (CUDA GPU offload)",
        )

    return InventoryCandidate(
        id=model_id,
        role="excluded",
        source_path=str(model_dir),
        architecture="unknown",
        size_bytes=_dir_size_bytes(model_dir),
        reason="Missing config.json or Diffusers model_index.json (incomplete or single-file checkpoint)",
    )


def inspect_drive_models(models_root: Path, *, total_vram_bytes: Optional[int] = None) -> DriveInventory:
    """Scan Drive `models/` once during Colab pre-run inventory and classify every folder."""
    inv = DriveInventory()
    if not models_root.is_dir():
        return inv
    for child in sorted(models_root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        item = inspect_single_drive_dir(child, total_vram_bytes=total_vram_bytes)
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
    free_disk_bytes: Optional[int] = None,
    min_disk_margin_bytes: int = 2 * GIB,
    enforce_gpu_fit: bool = True,
) -> tuple[InventoryCandidate, Optional[InventoryCandidate]]:
    """
    Validate the single selected language model and optional single image model before staging.
    Raises ValueError with an actionable message if either selection is invalid or does not fit.
    """
    lang_id = language_model_id.strip()
    img_id = image_model_id.strip()
    if not lang_id:
        raise ValueError("SELECTED_LANGUAGE_MODEL is required. Copy one language model name from the Inventory cell.")

    lang_dir = models_root / lang_id
    if not lang_dir.is_dir():
        raise ValueError(f"Selected language model directory not found on Drive: {lang_dir}")
    lang = inspect_single_drive_dir(lang_dir, total_vram_bytes=total_vram_bytes)
    if lang.role != "language":
        raise ValueError(f"'{lang_id}' is not a runnable v1.1.0 language model: {lang.reason}")
    if enforce_gpu_fit and lang.fits_current_gpu is False:
        avail_gib = round((total_vram_bytes or 0) / GIB, 1)
        raise ValueError(
            f"Selected language model '{lang_id}' requires ~{lang.min_vram_gib:.1f} GiB VRAM, "
            f"but current GPU only has {avail_gib:.1f} GiB. Choose a smaller model or switch Colab GPU tier."
        )

    img: Optional[InventoryCandidate] = None
    if img_id:
        img_dir = models_root / img_id
        if not img_dir.is_dir():
            raise ValueError(f"Selected image model directory not found on Drive: {img_dir}")
        img = inspect_single_drive_dir(img_dir, total_vram_bytes=total_vram_bytes)
        if img.role != "image":
            raise ValueError(f"'{img_id}' is not a runnable v1.1.0 image-generation model: {img.reason}")
        if enforce_gpu_fit and img.fits_current_gpu is False:
            avail_gib = round((total_vram_bytes or 0) / GIB, 1)
            raise ValueError(
                f"Selected image model '{img_id}' requires ~{img.min_vram_gib:.1f} GiB VRAM, "
                f"but current GPU only has {avail_gib:.1f} GiB."
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
        f"=== NEXUS AI LAB v1.1.0 — DRIVE MODEL INVENTORY ({gpu_summary}) ===",
        "",
        f"Selectable Language Models ({len(inv.language_models)}):",
    ]
    if not inv.language_models:
        lines.append("  (none found)")
    for m in inv.language_models:
        fit = "FITS GPU" if m.fits_current_gpu else ("TOO LARGE FOR GPU" if m.fits_current_gpu is False else "GPU N/A")
        lines.append(
            f"  - {m.id:<34} | impl={m.implementation:<8} | {m.size_bytes / GIB:5.1f} GiB "
            f"| minVRAM={m.min_vram_gib:4.1f} GiB | reasoning={m.reasoning_mode:<13} | {fit}"
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

    lines.extend(["", f"Skipped / Non-Runnable Folders ({len(inv.excluded_models)}):"])
    for m in inv.excluded_models:
        lines.append(f"  - {m.id:<34} | {m.reason}")

    return "\n".join(lines)
