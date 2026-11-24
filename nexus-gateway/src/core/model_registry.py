"""Dynamic Drive model discovery and compatibility classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

SourceType = Literal["hf_directory", "diffusers_pipeline", "gguf_package", "unknown"]
ToolCompatibility = Literal["yes", "no", "unknown"]

SUPPORTED_FLUX_PIPELINES = frozenset({"Flux2Pipeline", "Flux2KleinPipeline", "Flux2KleinKVPipeline"})
VLM_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "llava", "idefics2", "idefics3", "internvl_chat", "pixtral"})


@dataclass
class DiscoveredModel:
    id: str
    display_name: str
    source_path: str
    source_type: SourceType
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

    def to_openai_model(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "owned_by": "nexus",
            "display_name": self.display_name,
            "modalities": list(self.modalities),
            "context_window": self.context_window or 32768,
            "max_output_tokens": self.max_output_tokens or 4096,
            "tool_template_compatible": self.tool_template_compatible,
            "tool_behavior_verified": self.tool_behavior_verified,
        }

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


def _inspect_chat_template(model_dir: Path) -> tuple[ToolCompatibility, str]:
    tok_cfg = _read_json(model_dir / "tokenizer_config.json") or {}
    template = tok_cfg.get("chat_template")
    if isinstance(template, list):
        template = "\n".join(str(item.get("template", "")) for item in template if isinstance(item, dict))
    if not template and (model_dir / "chat_template.jinja").is_file():
        template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore")
    if not isinstance(template, str) or not template.strip():
        return "no", "No chat_template found"
    if "tools" in template and "tool_call" in template:
        return "yes", "Template references tools and tool_call blocks (requires live behavior verification)"
    return "no", "Chat template does not declare tools/tool_call handling"


def inspect_model_directory(model_dir: Path) -> DiscoveredModel:
    model_id = model_dir.name
    model_index = _read_json(model_dir / "model_index.json")
    if model_index is not None:
        class_name = str(model_index.get("_class_name") or "unknown")
        supported = class_name in SUPPORTED_FLUX_PIPELINES
        return DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(model_dir),
            source_type="diffusers_pipeline",
            architecture=class_name,
            loader_family="diffusers",
            modalities=["image"],
            tool_template_compatible="no",
            validation_status="compatible" if supported else "unsupported",
            validation_reason=f"{'Supported' if supported else 'Unsupported'} Diffusers pipeline {class_name}",
            is_chat_model=False,
            is_image_model=supported,
        )

    config = _read_json(model_dir / "config.json")
    if config is not None:
        model_type = str(config.get("model_type") or "unknown").lower()
        archs = config.get("architectures") or []
        arch = str(archs[0]) if isinstance(archs, list) and archs else model_type
        is_vlm = model_type in VLM_MODEL_TYPES or "vision_config" in config
        ctx_len = config.get("max_position_embeddings") or config.get("text_config", {}).get("max_position_embeddings")
        tool_compat, tool_reason = _inspect_chat_template(model_dir)
        return DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(model_dir),
            source_type="hf_directory",
            architecture=arch,
            loader_family="transformers",
            tool_parser_family="qwen" if "qwen" in model_type or "qwen" in arch.lower() else "generic_json",
            modalities=["text", "image"] if is_vlm else ["text"],
            tool_template_compatible=tool_compat,
            tool_template_reason=tool_reason,
            validation_status="compatible",
            validation_reason="Valid Hugging Face config.json",
            context_window=int(ctx_len) if isinstance(ctx_len, int) and ctx_len > 0 else 32768,
            max_output_tokens=4096,
            is_chat_model=True,
            is_image_model=False,
        )

    if any(model_dir.glob("*.gguf")):
        return DiscoveredModel(
            id=model_id,
            display_name=model_id,
            source_path=str(model_dir),
            source_type="gguf_package",
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
        validation_status="unsupported",
        validation_reason="Missing config.json, model_index.json, or .gguf artifacts",
    )


class ModelRegistry:
    def __init__(self, models_root: Optional[Path] = None) -> None:
        self._models_root = models_root
        self._models: dict[str, DiscoveredModel] = {}

    def register(self, model: DiscoveredModel) -> None:
        self._models[model.id] = model

    def scan(self) -> list[DiscoveredModel]:
        if self._models_root is None or not self._models_root.is_dir():
            return list(self._models.values())
        for child in sorted(self._models_root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                self._models[child.name] = inspect_model_directory(child)
        return list(self._models.values())

    def get(self, model_id: str) -> Optional[DiscoveredModel]:
        return self._models.get(model_id)

    def chat_models(self) -> list[DiscoveredModel]:
        return [m for m in self._models.values() if m.is_chat_model and m.validation_status == "compatible"]

    def image_models(self) -> list[DiscoveredModel]:
        return [m for m in self._models.values() if m.is_image_model and m.validation_status == "compatible"]

    def all_diagnostics(self) -> list[dict[str, Any]]:
        return [m.to_diagnostic() for m in self._models.values()]
