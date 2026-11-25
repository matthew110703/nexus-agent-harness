"""Dynamic Drive model, adapter, upscaler, and reasoning-capability discovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

SourceType = Literal["hf_directory", "diffusers_pipeline", "gguf_package", "adapter", "upscaler", "unknown"]
ToolCompatibility = Literal["yes", "no", "unknown"]
ReasoningMode = Literal["none", "binary_toggle", "effort_levels", "unknown"]

SUPPORTED_FLUX_PIPELINES = frozenset({
    "FluxPipeline",
    "Flux2Pipeline",
    "Flux2KleinPipeline",
    "Flux2KleinKVPipeline",
    "StableDiffusionXLPipeline",
    "QwenImagePipeline",
})
VLM_MODEL_TYPES = frozenset({
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "llava",
    "idefics2",
    "idefics3",
    "internvl_chat",
    "pixtral",
})
NON_CHAT_MODEL_TYPES = frozenset({
    "t5",
    "byt5",
    "umt5",
    "mt5",
    "clip",
    "clip_text_model",
    "siglip",
    "bert",
    "roberta",
    "autoencoder",
    "vae",
    "wan",
    "hunyuan_video",
    "hunyuanvideo",
    "ltx_video",
    "ltx",
    "cogvideox",
    "mochi",
})
VIDEO_OR_ENCODER_NAME_HINTS = ("hunyuanvideo", "wan2", "ltx-", "byt5", "umt5", "t5-v1", "video")


@dataclass
class ReasoningCapability:
    status: Literal["supported", "unsupported", "unknown"] = "unknown"
    mode: ReasoningMode = "unknown"
    levels: dict[str, Optional[str]] = field(default_factory=dict)
    default: Optional[str] = None
    source: str = "heuristic"
    confidence: Literal["verified", "inferred", "unverified"] = "unverified"

    def to_harness_reasoning_efforts(self) -> Any:
        """Project into Harness `llm-pi-ai` `reasoningEfforts` (dict or `false` or `None`)."""
        if self.status == "unsupported":
            return False
        if self.status == "supported" and self.levels:
            return dict(self.levels)
        return None


@dataclass
class DiscoveredModel:
    id: str
    display_name: str
    source_path: str
    source_type: SourceType
    category: Literal["chat", "vlm", "image", "adapter", "upscaler", "other"] = "other"
    architecture: str = "unknown"
    loader_family: str = "unknown"
    tool_parser_family: str = "qwen"
    modalities: list[str] = field(default_factory=lambda: ["text"])
    tool_template_compatible: ToolCompatibility = "unknown"
    tool_template_reason: str = ""
    tool_behavior_verified: Optional[bool] = None
    validation_status: Literal["compatible", "unsupported", "unvalidated"] = "unvalidated"
    validation_reason: str = ""
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    is_chat_model: bool = False
    is_image_model: bool = False
    reasoning: ReasoningCapability = field(default_factory=ReasoningCapability)
    default_steps: Optional[int] = None

    def to_openai_model(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "owned_by": "nexus",
            "display_name": self.display_name,
            "category": self.category,
            "modalities": list(self.modalities),
            "context_window": self.context_window or 32768,
            "max_output_tokens": self.max_output_tokens or 4096,
            "tool_template_compatible": self.tool_template_compatible,
            "tool_behavior_verified": self.tool_behavior_verified,
            "reasoning": asdict(self.reasoning),
        }
        efforts = self.reasoning.to_harness_reasoning_efforts()
        if efforts is not None:
            payload["reasoning_efforts"] = efforts
        if self.default_steps is not None:
            payload["default_steps"] = self.default_steps
        return payload

    def to_diagnostic(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


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


def _inspect_chat_template(model_root: Path) -> tuple[ToolCompatibility, str, ReasoningCapability]:
    template = _read_chat_template(model_root)
    if not template.strip():
        return (
            "no",
            "No chat_template found",
            ReasoningCapability(status="unknown", mode="unknown", source="chat_template", confidence="unverified"),
        )
    low = template.lower()
    if "tools" in low and "tool_call" in low:
        tool_compat: ToolCompatibility = "yes"
        tool_reason = "Template references tools and tool_call blocks (requires live behavior verification)"
    else:
        tool_compat = "no"
        tool_reason = "Chat template does not declare tools/tool_call handling"

    # Detect verified binary thinking toggle (e.g. Qwen3 `enable_thinking`).
    if "enable_thinking" in low:
        reasoning = ReasoningCapability(
            status="supported",
            mode="binary_toggle",
            levels={"off": None, "high": "high"},
            default="high",
            source="chat_template:enable_thinking",
            confidence="verified",
        )
    elif "reasoning_effort" in low:
        reasoning = ReasoningCapability(
            status="supported",
            mode="effort_levels",
            levels={"off": None, "low": "low", "medium": "medium", "high": "high"},
            default="medium",
            source="chat_template:reasoning_effort",
            confidence="inferred",
        )
    else:
        reasoning = ReasoningCapability(
            status="unsupported",
            mode="none",
            levels={},
            default=None,
            source="chat_template",
            confidence="inferred",
        )
    return tool_compat, tool_reason, reasoning


def _check_shard_completeness(model_root: Path) -> tuple[bool, str]:
    """Validate weight index completeness if a sharded index file exists."""
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_path = model_root / idx_name
        if idx_path.is_file():
            idx_data = _read_json(idx_path) or {}
            weight_map = idx_data.get("weight_map")
            if isinstance(weight_map, dict) and weight_map:
                missing = sorted({str(fname) for fname in weight_map.values() if not (model_root / str(fname)).is_file()})
                if missing:
                    return False, f"Incomplete shards ({len(missing)} missing, e.g. {missing[0]})"
                return True, "All indexed weight shards present"
    return True, "Weight index check passed"


def _default_steps_for_diffusion(model_id: str, class_name: str) -> int:
    low_cls = class_name.lower().replace("_", "")
    low_name = model_id.lower()
    if "kleinkv" in low_cls or "klein-kv" in low_name or "klein-9b-kv" in low_name or "klein" in low_name:
        return 4
    if "flux2" in low_cls and "dev" in low_name:
        return 50
    return 25


def _apply_capability_override(model: DiscoveredModel, override: dict[str, Any]) -> None:
    if not isinstance(override, dict):
        return
    if isinstance(override.get("display_name"), str):
        model.display_name = override["display_name"]
    if isinstance(override.get("context_window"), int) and override["context_window"] > 0:
        model.context_window = override["context_window"]
    if isinstance(override.get("max_output_tokens"), int) and override["max_output_tokens"] > 0:
        model.max_output_tokens = override["max_output_tokens"]
    if isinstance(override.get("tool_behavior_verified"), bool):
        model.tool_behavior_verified = override["tool_behavior_verified"]
    reasoning_cfg = override.get("reasoning")
    if isinstance(reasoning_cfg, dict):
        levels_raw = reasoning_cfg.get("levels")
        levels: dict[str, Optional[str]] = {}
        if isinstance(levels_raw, dict):
            for k, v in levels_raw.items():
                levels[str(k)] = None if v is None else str(v)
        model.reasoning = ReasoningCapability(
            status=reasoning_cfg.get("status", "supported" if levels else "unsupported"),
            mode=reasoning_cfg.get("mode", "effort_levels" if len(levels) > 2 else ("binary_toggle" if levels else "none")),
            levels=levels,
            default=reasoning_cfg.get("default"),
            source=str(reasoning_cfg.get("source") or "user_sidecar_override"),
            confidence="verified",
        )


def inspect_model_directory(model_dir: Path, override: Optional[dict[str, Any]] = None) -> DiscoveredModel:
    model_id = model_dir.name

    diffusers_root = _resolve_snapshot_root(model_dir, "model_index.json")
    if diffusers_root is not None:
        model_index = _read_json(diffusers_root / "model_index.json") or {}
        class_name = str(model_index.get("_class_name") or "unknown")
        supported = class_name in SUPPORTED_FLUX_PIPELINES
        model = DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(diffusers_root),
            source_type="diffusers_pipeline",
            category="image",
            architecture=class_name,
            loader_family="diffusers",
            modalities=["image"],
            tool_template_compatible="no",
            validation_status="compatible" if supported else "unsupported",
            validation_reason=f"{'Supported' if supported else 'Unsupported'} Diffusers pipeline {class_name}",
            is_chat_model=False,
            is_image_model=supported,
            reasoning=ReasoningCapability(status="unsupported", mode="none", source="diffusers", confidence="verified"),
            default_steps=_default_steps_for_diffusion(model_id, class_name),
        )
        if override:
            _apply_capability_override(model, override)
        return model

    hf_root = _resolve_snapshot_root(model_dir, "config.json")
    if hf_root is not None:
        config = _read_json(hf_root / "config.json") or {}
        shards_ok, shards_msg = _check_shard_completeness(hf_root)
        model_type = str(config.get("model_type") or "unknown").lower()
        archs = config.get("architectures") or []
        arch = str(archs[0]) if isinstance(archs, list) and archs else model_type
        low_id = model_id.lower()
        low_arch = arch.lower()
        is_non_chat = (
            model_type in NON_CHAT_MODEL_TYPES
            or any(hint in low_id for hint in VIDEO_OR_ENCODER_NAME_HINTS)
            or any(tag in low_arch for tag in ("video", "t5encoder", "cliptext", "autoencoder", "vae", "transformer3d"))
        )
        if is_non_chat:
            return DiscoveredModel(
                id=model_id,
                display_name=model_id,
                source_path=str(hf_root),
                source_type="hf_directory",
                category="other",
                architecture=arch,
                loader_family="unsupported",
                modalities=[],
                tool_template_compatible="no",
                validation_status="unsupported",
                validation_reason=f"Non-chat video or encoder checkpoint ({model_type}/{arch})",
                is_chat_model=False,
                is_image_model=False,
                reasoning=ReasoningCapability(status="unsupported", mode="none", source="non_chat", confidence="verified"),
            )
        is_vlm = model_type in VLM_MODEL_TYPES or "vision_config" in config
        text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        ctx_len = config.get("max_position_embeddings") or text_cfg.get("max_position_embeddings")
        tool_compat, tool_reason, reasoning = _inspect_chat_template(hf_root)
        model = DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(hf_root),
            source_type="hf_directory",
            category="vlm" if is_vlm else "chat",
            architecture=arch,
            loader_family="transformers",
            tool_parser_family="qwen" if "qwen" in model_type or "qwen" in low_arch else "generic_json",
            modalities=["text", "image"] if is_vlm else ["text"],
            tool_template_compatible=tool_compat,
            tool_template_reason=tool_reason,
            validation_status="compatible" if shards_ok else "unsupported",
            validation_reason="Valid Hugging Face config.json" if shards_ok else shards_msg,
            context_window=int(ctx_len) if isinstance(ctx_len, int) and ctx_len > 0 else 32768,
            max_output_tokens=4096,
            is_chat_model=shards_ok,
            is_image_model=False,
            reasoning=reasoning,
        )
        if override:
            _apply_capability_override(model, override)
        return model

    if any(model_dir.glob("*.gguf")):
        return DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(model_dir),
            source_type="gguf_package",
            category="other",
            architecture="gguf",
            loader_family="ollama",
            validation_status="unsupported",
            validation_reason="GGUF requires optional Ollama backend",
        )

    return DiscoveredModel(
        id=model_id,
        display_name=model_id,
        source_path=str(model_dir),
        source_type="unknown",
        category="other",
        validation_status="unsupported",
        validation_reason="Missing config.json, model_index.json, or .gguf artifacts",
    )


class ModelRegistry:
    def __init__(
        self,
        models_root: Optional[Path] = None,
        *,
        adapters_root: Optional[Path] = None,
        upscalers_root: Optional[Path] = None,
        capabilities_sidecar: Optional[Path] = None,
    ) -> None:
        self._models_root = models_root
        self._adapters_root = adapters_root
        self._upscalers_root = upscalers_root
        self._capabilities_sidecar = capabilities_sidecar
        self._models: dict[str, DiscoveredModel] = {}
        self._adapters: list[dict[str, Any]] = []
        self._upscalers: list[dict[str, Any]] = []

    def register(self, model: DiscoveredModel) -> None:
        self._models[model.id] = model

    def _load_sidecar_overrides(self) -> dict[str, dict[str, Any]]:
        if self._capabilities_sidecar is None or not self._capabilities_sidecar.is_file():
            return {}
        data = _read_json(self._capabilities_sidecar)
        if not isinstance(data, dict):
            return {}
        models_map = data.get("models") if isinstance(data.get("models"), dict) else data
        return {str(k): v for k, v in models_map.items() if isinstance(v, dict)}

    def scan(self) -> list[DiscoveredModel]:
        overrides = self._load_sidecar_overrides()
        if self._models_root is not None and self._models_root.is_dir():
            for child in sorted(self._models_root.iterdir()):
                if child.is_dir() and not child.name.startswith((".", "_")):
                    self._models[child.name] = inspect_model_directory(child, overrides.get(child.name))

        self._adapters = []
        if self._adapters_root is not None and self._adapters_root.is_dir():
            for child in sorted(self._adapters_root.iterdir()):
                if not child.name.startswith("."):
                    self._adapters.append({"id": child.name, "path": str(child), "is_dir": child.is_dir()})

        self._upscalers = []
        if self._upscalers_root is not None and self._upscalers_root.is_dir():
            for child in sorted(self._upscalers_root.iterdir()):
                if not child.name.startswith("."):
                    self._upscalers.append({"id": child.name, "path": str(child), "is_dir": child.is_dir()})

        return list(self._models.values())

    def get(self, model_id: str) -> Optional[DiscoveredModel]:
        return self._models.get(model_id)

    def chat_models(self) -> list[DiscoveredModel]:
        return [m for m in self._models.values() if m.is_chat_model and m.validation_status == "compatible"]

    def image_models(self) -> list[DiscoveredModel]:
        return [m for m in self._models.values() if m.is_image_model and m.validation_status == "compatible"]

    def adapters(self) -> list[dict[str, Any]]:
        return list(self._adapters)

    def upscalers(self) -> list[dict[str, Any]]:
        return list(self._upscalers)

    def all_diagnostics(self) -> list[dict[str, Any]]:
        return [m.to_diagnostic() for m in self._models.values()]
