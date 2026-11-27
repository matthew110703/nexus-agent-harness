"""Dedicated FLUX.2-klein Diffusers image-generation runtime (FLUX.2-klein-4B / 9B)."""

from __future__ import annotations

import base64
import gc
import inspect
import io
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from models.base import GenerationCancelledError, ImageQualityProfile, QualityProfileSpec

FLUX2_KLEIN_QUALITY_PROFILES: dict[ImageQualityProfile, QualityProfileSpec] = {
    "normal": QualityProfileSpec(name="normal", width=768, height=768, steps=4, guidance_scale=3.5),
    "detailed": QualityProfileSpec(name="detailed", width=1024, height=1024, steps=6, guidance_scale=3.5),
    "maximum": QualityProfileSpec(name="maximum", width=1024, height=1024, steps=8, guidance_scale=4.0),
}

DIRECT_CUDA_FREE_VRAM_4B_BYTES = 12 * 1024 ** 3
DIRECT_CUDA_FREE_VRAM_9B_BYTES = 22 * 1024 ** 3


def resolve_flux2_klein_params(
    *,
    quality: ImageQualityProfile = "detailed",
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    num_images: int = 1,
) -> dict[str, Any]:
    if quality not in FLUX2_KLEIN_QUALITY_PROFILES:
        raise ValueError(f"Unsupported image quality profile {quality!r}; choose normal, detailed, or maximum")
    if num_images != 1:
        raise ValueError("Nexus v1.1.0 enforces exactly 1 image per generation call (num_images must be 1)")
    spec = FLUX2_KLEIN_QUALITY_PROFILES[quality]
    w = int(width) if width is not None else spec.width
    h = int(height) if height is not None else spec.height
    s = int(steps) if steps is not None else spec.steps
    if w < 256 or w > 1536 or w % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 256 and 1536")
    if h < 256 or h > 1536 or h % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 256 and 1536")
    if s < 1 or s > 50:
        raise ValueError("steps must be between 1 and 50 for FLUX.2-klein")
    return {
        "quality": quality,
        "width": w,
        "height": h,
        "steps": s,
        "num_images": 1,
        "guidance_scale": spec.guidance_scale,
        "profile_spec": asdict(spec),
    }


class Flux2KleinImageModel:
    """Dedicated Diffusers runtime for distilled `Flux2KleinPipeline` checkpoints."""

    implementation = "flux2_klein"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "Flux2KleinPipeline") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.pipe: Any = None

    def is_ready(self) -> bool:
        return self.pipe is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from diffusers import DiffusionPipeline  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2-klein image generation requires a CUDA GPU runtime")

        dtype = torch.bfloat16 if (precision == "bf16" or (precision is None and torch.cuda.is_bf16_supported())) else torch.float16
        pipe = DiffusionPipeline.from_pretrained(str(self.model_path), torch_dtype=dtype, local_files_only=True)

        threshold = DIRECT_CUDA_FREE_VRAM_4B_BYTES if "4b" in self.model_id.lower() else DIRECT_CUDA_FREE_VRAM_9B_BYTES
        try:
            free_vram, _ = torch.cuda.mem_get_info(0)
        except Exception:
            free_vram = 0
        if free_vram >= threshold:
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
            raise RuntimeError(f"Flux2KleinImageModel ({self.model_id}) is not loaded")

        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        params = resolve_flux2_klein_params(
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

                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
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

        try:
            sig_params = inspect.signature(self.pipe.__call__).parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
        except (TypeError, ValueError):
            sig_params = {}
            accepts_var_kw = False

        def _supports(name: str) -> bool:
            return name in sig_params or accepts_var_kw

        ref_image = None
        ref_b64 = req.get("reference_image_b64")
        if isinstance(ref_b64, str) and ref_b64.strip():
            from PIL import Image
            raw_b64 = ref_b64.split(",", 1)[1] if "," in ref_b64 and ref_b64.startswith("data:") else ref_b64
            ref_image = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
            ref_image = ref_image.resize((params["width"], params["height"]), Image.Resampling.LANCZOS)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": params["width"],
            "height": params["height"],
            "num_inference_steps": steps_total,
        }
        if ref_image is not None and _supports("image"):
            call_kwargs["image"] = ref_image
        if ref_image is not None and _supports("strength"):
            strength = req.get("strength")
            call_kwargs["strength"] = max(0.1, min(1.0, float(strength))) if strength is not None else 0.75
        if _supports("num_images_per_prompt"):
            call_kwargs["num_images_per_prompt"] = 1
        if params.get("guidance_scale") is not None and _supports("guidance_scale"):
            call_kwargs["guidance_scale"] = params["guidance_scale"]
        if generator is not None and _supports("generator"):
            call_kwargs["generator"] = generator
        if _supports("callback_on_step_end"):
            call_kwargs["callback_on_step_end"] = _on_step_end
        elif on_progress is not None:
            on_progress({
                "phase": "denoising",
                "step": 0,
                "steps_total": steps_total,
                "progress": 0.0,
                "indeterminate": True,
                "message": f"Denoising with {self.model_id}",
            })

        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Image generation cancelled before start")

        result = self.pipe(**call_kwargs)
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("FLUX.2-klein pipeline returned no images")

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "model": self.model_id,
            "architecture": self.architecture,
            "quality": params["quality"],
            "width": params["width"],
            "height": params["height"],
            "steps": steps_total,
            "seed": seed if isinstance(seed, int) else None,
            "b64_png": b64_png,
        }

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
