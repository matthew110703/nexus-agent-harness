"""Shared minimal lifecycle contracts and lazy dynamic loader for Nexus v1.1.0."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Protocol

LanguageImplName = Literal[
    "qwen3_vl",
    "qwen3_5",
    "gpt_oss",
    "gemma3",
    "deepseek_vl2",
    "deepseek_vl",
    "gguf_qwen3_5",
    "gguf_gemma4",
    "gguf_deepseek_qwen3_5",
    "gguf_llama3",
    "gguf_llama4",
    "gguf_generic",
]
ImageImplName = Literal["flux2_klein", "sdxl"]
ReasoningMode = Literal["none", "binary_toggle", "effort_levels"]
ImageQualityProfile = Literal["normal", "detailed", "maximum"]

LANGUAGE_IMPLEMENTATIONS: dict[str, tuple[str, str]] = {
    "qwen3_vl": ("models.language.qwen3_vl", "Qwen3VLLanguageModel"),
    "qwen3_5": ("models.language.qwen3_5", "Qwen35LanguageModel"),
    "gpt_oss": ("models.language.gpt_oss", "GptOssLanguageModel"),
    "gemma3": ("models.language.gemma3", "Gemma3LanguageModel"),
    "deepseek_vl2": ("models.language.deepseek_vl2", "DeepSeekVl2LanguageModel"),
    "deepseek_vl": ("models.language.deepseek_vl", "DeepSeekVlLanguageModel"),
    "gguf_qwen3_5": ("models.language.gguf_qwen3_5", "GgufQwen35Model"),
    "gguf_gemma4": ("models.language.gguf_gemma4", "GgufGemma4Model"),
    "gguf_deepseek_qwen3_5": ("models.language.gguf_deepseek_qwen3_5", "GgufDeepSeekQwen35Model"),
    "gguf_llama3": ("models.language.gguf_llama3", "GgufLlama3Model"),
    "gguf_llama4": ("models.language.gguf_llama4", "GgufLlama4Model"),
    "gguf_generic": ("models.language.gguf_generic", "GgufGenericModel"),
}

IMAGE_IMPLEMENTATIONS: dict[str, tuple[str, str]] = {
    "flux2_klein": ("models.image.flux2_klein", "Flux2KleinImageModel"),
    "sdxl": ("models.image.sdxl", "SdxlImageModel"),
}


class GenerationCancelledError(RuntimeError):
    """Raised when an active image generation is cancelled by caller or disconnect."""


def select_torch_dtype(torch_mod: Any, precision: Optional[str] = None) -> Any:
    """Select a hardware-supported dtype without silently picking float16 on CPU."""
    cuda = bool(torch_mod.cuda.is_available())
    if not cuda:
        if precision == "bf16":
            return torch_mod.bfloat16
        return torch_mod.float32
    if precision == "fp32":
        return torch_mod.float32
    if precision == "fp16":
        return torch_mod.float16
    if precision == "bf16" or bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)()):
        return torch_mod.bfloat16
    return torch_mod.float16


@dataclass(frozen=True)
class QualityProfileSpec:
    name: ImageQualityProfile
    width: int
    height: int
    steps: int
    guidance_scale: Optional[float] = None


@dataclass
class SelectedLanguageSpec:
    id: str
    display_name: str
    model_path: str
    implementation: LanguageImplName
    architecture: str
    modalities: list[str] = field(default_factory=lambda: ["text"])
    context_window: int = 32768
    max_output_tokens: int = 4096
    reasoning_mode: ReasoningMode = "none"
    reasoning_efforts: Any = False
    default_reasoning_effort: Optional[str] = None
    min_vram_gib: float = 0.0
    size_bytes: int = 0
    quantization: str = "unknown"
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    threads: Optional[int] = None
    compatibility_tier: Optional[str] = None
    can_coreside_image: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_openai_model(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "owned_by": "nexus",
            "display_name": self.display_name,
            "implementation": self.implementation,
            "architecture": self.architecture,
            "modalities": list(self.modalities),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_mode": self.reasoning_mode,
            "reasoning_efforts": self.reasoning_efforts,
            "default_reasoning_effort": self.default_reasoning_effort,
            "min_vram_gib": self.min_vram_gib,
            "size_bytes": self.size_bytes,
            "quantization": self.quantization,
            "compatibility_tier": self.compatibility_tier,
            "can_coreside_image": self.can_coreside_image,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectedLanguageSpec":
        impl = str(data.get("implementation") or "").strip()
        if impl not in LANGUAGE_IMPLEMENTATIONS:
            raise ValueError(f"Unsupported language model implementation: {impl!r}")
        model_id = str(data.get("id") or "").strip()
        model_path = str(data.get("model_path") or "").strip()
        if not model_id or not model_path:
            raise ValueError("SelectedLanguageSpec requires non-empty id and model_path")
        return cls(
            id=model_id,
            display_name=str(data.get("display_name") or model_id),
            model_path=model_path,
            implementation=impl,  # type: ignore[arg-type]
            architecture=str(data.get("architecture") or "unknown"),
            modalities=[str(m) for m in (data.get("modalities") or ["text"])],
            context_window=int(data.get("context_window") or 32768),
            max_output_tokens=int(data.get("max_output_tokens") or 4096),
            reasoning_mode=str(data.get("reasoning_mode") or "none"),  # type: ignore[arg-type]
            reasoning_efforts=data.get("reasoning_efforts", False),
            default_reasoning_effort=data.get("default_reasoning_effort"),
            min_vram_gib=float(data.get("min_vram_gib") or 0.0),
            size_bytes=int(data.get("size_bytes") or 0),
            quantization=str(data.get("quantization") or "unknown"),
            n_ctx=int(data["n_ctx"]) if data.get("n_ctx") is not None else None,
            n_gpu_layers=int(data["n_gpu_layers"]) if data.get("n_gpu_layers") is not None else None,
            batch_size=int(data["batch_size"]) if data.get("batch_size") is not None else None,
            ubatch_size=int(data["ubatch_size"]) if data.get("ubatch_size") is not None else None,
            threads=int(data["threads"]) if data.get("threads") is not None else None,
            compatibility_tier=str(data["compatibility_tier"]) if data.get("compatibility_tier") else None,
            can_coreside_image=bool(data.get("can_coreside_image", True)),
        )


@dataclass
class SelectedImageSpec:
    id: str
    display_name: str
    model_path: str
    implementation: ImageImplName
    architecture: str
    default_steps: int = 6
    min_vram_gib: float = 0.0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectedImageSpec":
        impl = str(data.get("implementation") or "").strip()
        if impl not in IMAGE_IMPLEMENTATIONS:
            raise ValueError(f"Unsupported image model implementation: {impl!r}")
        model_id = str(data.get("id") or "").strip()
        model_path = str(data.get("model_path") or "").strip()
        if not model_id or not model_path:
            raise ValueError("SelectedImageSpec requires non-empty id and model_path")
        return cls(
            id=model_id,
            display_name=str(data.get("display_name") or model_id),
            model_path=model_path,
            implementation=impl,  # type: ignore[arg-type]
            architecture=str(data.get("architecture") or "unknown"),
            default_steps=int(data.get("default_steps") or 6),
            min_vram_gib=float(data.get("min_vram_gib") or 0.0),
            size_bytes=int(data.get("size_bytes") or 0),
        )


@dataclass
class SessionRuntimeConfig:
    language_model: SelectedLanguageSpec
    image_model: Optional[SelectedImageSpec] = None
    version: str = "1.1.0"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "language_model": self.language_model.to_dict(),
            "image_model": self.image_model.to_dict() if self.image_model is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRuntimeConfig":
        if not isinstance(data, dict) or not isinstance(data.get("language_model"), dict):
            raise ValueError("SessionRuntimeConfig requires a 'language_model' object")
        lang = SelectedLanguageSpec.from_dict(data["language_model"])
        raw_img = data.get("image_model")
        img = SelectedImageSpec.from_dict(raw_img) if isinstance(raw_img, dict) else None
        return cls(
            version=str(data.get("version") or "1.1.0"),
            created_at=str(data.get("created_at") or ""),
            language_model=lang,
            image_model=img,
        )


def load_session_runtime_config(path: Path) -> SessionRuntimeConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Session runtime manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionRuntimeConfig.from_dict(data)


def save_session_runtime_config(path: Path, cfg: SessionRuntimeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class LanguageModelRuntime(Protocol):
    model_id: str
    model_path: Path
    implementation: str

    def load(self, *, precision: Optional[str] = None) -> None: ...
    def is_ready(self) -> bool: ...
    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]: ...
    def unload(self) -> None: ...


class ImageModelRuntime(Protocol):
    model_id: str
    model_path: Path
    implementation: str

    def load(self, *, precision: Optional[str] = None) -> None: ...
    def is_ready(self) -> bool: ...
    def generate(
        self,
        req: dict[str, Any],
        *,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ) -> dict[str, Any]: ...
    def unload(self) -> None: ...


def load_dedicated_language_runtime(
    *,
    implementation: str,
    model_id: str,
    model_path: Path,
    architecture: str = "unknown",
    runtime_options: Optional[dict[str, Any]] = None,
) -> LanguageModelRuntime:
    """Lazy-import and instantiate only the selected language model's dedicated module."""
    target = LANGUAGE_IMPLEMENTATIONS.get(implementation)
    if target is None:
        raise ValueError(f"Unsupported language implementation {implementation!r} for model {model_id}")
    module_name, class_name = target
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    kwargs: dict[str, Any] = {
        "model_id": model_id,
        "model_path": model_path,
        "architecture": architecture,
    }
    if implementation.startswith("gguf_") and runtime_options:
        for key in ("n_ctx", "n_gpu_layers", "batch_size", "ubatch_size", "threads"):
            if runtime_options.get(key) is not None:
                kwargs[key] = int(runtime_options[key])
    return cls(**kwargs)


def load_dedicated_image_runtime(
    *,
    implementation: str,
    model_id: str,
    model_path: Path,
    architecture: str = "unknown",
) -> ImageModelRuntime:
    """Lazy-import and instantiate only the selected image model's dedicated module."""
    target = IMAGE_IMPLEMENTATIONS.get(implementation)
    if target is None:
        raise ValueError(f"Unsupported image implementation {implementation!r} for model {model_id}")
    module_name, class_name = target
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(model_id=model_id, model_path=model_path, architecture=architecture)
