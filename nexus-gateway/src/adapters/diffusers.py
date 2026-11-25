"""Independent Diffusers/FLUX.2 image generation adapter with real progress and cancellation."""

from __future__ import annotations

import base64
import gc
import inspect
import io
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from core.model_registry import SUPPORTED_DIFFUSERS_PIPELINES

ImageQualityProfile = Literal["normal", "detailed", "maximum"]

# Free VRAM threshold (GiB) above which a pipeline is placed directly on CUDA without CPU offload.
DIRECT_CUDA_FREE_VRAM_BYTES = 20 * 1024 ** 3


class GenerationCancelledError(RuntimeError):
    """Raised when an active Diffusers generation is cancelled by caller or disconnect."""


@dataclass(frozen=True)
class QualityProfileSpec:
    name: ImageQualityProfile
    width: int
    height: int
    steps_standard: int
    steps_distilled: int
    guidance_scale: Optional[float] = None


QUALITY_PROFILES: dict[ImageQualityProfile, QualityProfileSpec] = {
    "normal": QualityProfileSpec(name="normal", width=768, height=768, steps_standard=20, steps_distilled=4, guidance_scale=3.5),
    "detailed": QualityProfileSpec(name="detailed", width=1024, height=1024, steps_standard=32, steps_distilled=6, guidance_scale=3.5),
    "maximum": QualityProfileSpec(name="maximum", width=1024, height=1024, steps_standard=50, steps_distilled=8, guidance_scale=4.0),
}


def resolve_image_generation_params(
    *,
    model_id: str,
    architecture: str,
    quality: ImageQualityProfile = "detailed",
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    num_images: int = 1,
) -> dict[str, Any]:
    """Resolve explicit image generation settings for a given pipeline and quality profile."""
    if quality not in QUALITY_PROFILES:
        raise ValueError(f"Unsupported image quality profile {quality!r}; choose normal, detailed, or maximum")
    if num_images != 1:
        raise ValueError("Nexus v1.0.1 enforces exactly 1 image per generation call (num_images must be 1)")
    spec = QUALITY_PROFILES[quality]
    low = f"{model_id} {architecture}".lower()
    is_distilled = "klein" in low or "schnell" in low
    resolved_width = int(width) if width is not None else spec.width
    resolved_height = int(height) if height is not None else spec.height
    resolved_steps = int(steps) if steps is not None else (spec.steps_distilled if is_distilled else spec.steps_standard)

    if resolved_width < 256 or resolved_width > 1536 or resolved_width % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 256 and 1536")
    if resolved_height < 256 or resolved_height > 1536 or resolved_height % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 256 and 1536")
    if resolved_steps < 1 or resolved_steps > 100:
        raise ValueError("steps must be between 1 and 100")

    return {
        "quality": quality,
        "width": resolved_width,
        "height": resolved_height,
        "steps": resolved_steps,
        "num_images": 1,
        "guidance_scale": spec.guidance_scale,
        "profile_spec": asdict(spec),
    }


def validate_diffusers_model_index(model_path: Path) -> str:
    idx_path = model_path / "model_index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing model_index.json in {model_path}")
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    class_name = str(data.get("_class_name") or "").strip()
    if not class_name:
        raise ValueError(f"model_index.json in {model_path} has no _class_name")
    if class_name not in SUPPORTED_DIFFUSERS_PIPELINES:
        raise ValueError(f"Unsupported Diffusers pipeline class {class_name!r}")
    return class_name


def build_pipeline_call_kwargs(
    pipe: Any,
    architecture: str,
    *,
    prompt: str,
    params: dict[str, Any],
    generator: Any = None,
    step_callback: Optional[Callable[..., Any]] = None,
) -> tuple[dict[str, Any], bool]:
    """
    Build pipeline kwargs using both explicit pipeline-class compatibility metadata
    and `inspect.signature(pipe.__call__)` so unsupported kwargs are never passed blindly.
    Returns `(call_kwargs, step_callback_supported)`.
    """
    meta = SUPPORTED_DIFFUSERS_PIPELINES.get(architecture, {})
    try:
        sig_params = inspect.signature(pipe.__call__).parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
    except (TypeError, ValueError):
        sig_params = {}
        accepts_var_kw = False

    def _supports(name: str) -> bool:
        return name in sig_params or accepts_var_kw

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "width": params["width"],
        "height": params["height"],
        "num_inference_steps": params["steps"],
    }
    if _supports("num_images_per_prompt"):
        kwargs["num_images_per_prompt"] = 1
    if (
        params.get("guidance_scale") is not None
        and meta.get("supports_guidance_scale", False)
        and _supports("guidance_scale")
    ):
        kwargs["guidance_scale"] = params["guidance_scale"]
    if generator is not None and _supports("generator"):
        kwargs["generator"] = generator

    cb_supported = bool(
        step_callback is not None
        and meta.get("supports_step_callback", False)
        and _supports("callback_on_step_end")
    )
    if cb_supported:
        kwargs["callback_on_step_end"] = step_callback

    return kwargs, cb_supported


class DiffusersImageAdapter:
    """Isolated Diffusers pipeline adapter loaded on demand inside the GPU worker."""

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture or validate_diffusers_model_index(model_path)
        self.pipe: Any = None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from diffusers import DiffusionPipeline  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            raise RuntimeError("Diffusion image generation requires a CUDA GPU runtime")
        validate_diffusers_model_index(self.model_path)
        dtype = torch.bfloat16 if (precision == "bf16" or (precision is None and torch.cuda.is_bf16_supported())) else torch.float16
        pipe = DiffusionPipeline.from_pretrained(str(self.model_path), torch_dtype=dtype, local_files_only=True)

        # Hardware-aware placement: keep directly on CUDA when ample VRAM is free; offload only when needed.
        try:
            free_vram, _total_vram = torch.cuda.mem_get_info(0)
        except Exception:
            free_vram = 0
        if free_vram >= DIRECT_CUDA_FREE_VRAM_BYTES:
            pipe.to("cuda")
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def generate(
        self,
        req: dict[str, Any],
        *,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        if self.pipe is None:
            raise RuntimeError("DiffusersImageAdapter is not loaded")

        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        params = resolve_image_generation_params(
            model_id=self.model_id,
            architecture=self.architecture,
            quality=req.get("quality") or "detailed",
            width=req.get("width"),
            height=req.get("height"),
            steps=req.get("steps"),
            num_images=int(req.get("num_images") or 1),
        )

        generator = None
        seed = req.get("seed")
        if isinstance(seed, int):
            try:
                import torch  # type: ignore[import-not-found]

                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(seed)
            except Exception:
                generator = None

        steps_total = params["steps"]

        def _on_step_end(pipe_obj: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            if cancel_event is not None and cancel_event.is_set():
                setattr(pipe_obj, "_interrupt", True)
                raise GenerationCancelledError("Image generation cancelled during denoising")
            completed = min(steps_total, int(step) + 1)
            pct = round(completed / max(1, steps_total) * 100.0, 2)
            if on_progress is not None:
                on_progress({
                    "phase": "denoising",
                    "step": completed,
                    "steps_total": steps_total,
                    "progress": pct,
                    "indeterminate": False,
                    "message": f"Denoising step {completed}/{steps_total}",
                })
            return callback_kwargs

        call_kwargs, cb_supported = build_pipeline_call_kwargs(
            self.pipe,
            self.architecture,
            prompt=prompt,
            params=params,
            generator=generator,
            step_callback=_on_step_end,
        )
        if on_progress is not None and not cb_supported:
            on_progress({
                "phase": "denoising",
                "step": 0,
                "steps_total": steps_total,
                "progress": 0.0,
                "indeterminate": True,
                "message": f"Denoising with {self.model_id} (pipeline does not report per-step progress)",
            })

        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Image generation cancelled before start")

        result = self.pipe(**call_kwargs)
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Diffusers pipeline returned no images")

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "model": self.model_id,
            "architecture": self.architecture,
            "quality": params["quality"],
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "seed": seed if isinstance(seed, int) else None,
            "b64_png": b64_png,
        }

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
