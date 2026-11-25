"""Independent Diffusers/FLUX.2 image generation adapter with explicit quality profiles."""

from __future__ import annotations

import base64
import gc
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

ImageQualityProfile = Literal["normal", "detailed", "maximum"]


@dataclass(frozen=True)
class QualityProfileSpec:
    name: ImageQualityProfile
    width: int
    height: int
    steps_standard: int
    steps_distilled: int
    guidance_scale: Optional[float] = None


# Quality-first profiles; never silently downgraded.
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
    if num_images < 1 or num_images > 4:
        raise ValueError("num_images must be between 1 and 4")
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
        "num_images": num_images,
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
    return class_name


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
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        import torch  # type: ignore[import-not-found]

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
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": params["width"],
            "height": params["height"],
            "num_inference_steps": params["steps"],
            "num_images_per_prompt": params["num_images"],
        }
        seed = req.get("seed")
        if isinstance(seed, int):
            call_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipe(**call_kwargs)
        images = getattr(result, "images", None) or []
        if not images:
            raise RuntimeError("Diffusers pipeline returned no images")

        encoded_images: list[str] = []
        for img in images[: params["num_images"]]:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded_images.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        return {
            "model": self.model_id,
            "architecture": self.architecture,
            "quality": params["quality"],
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "seed": seed if isinstance(seed, int) else None,
            "b64_png": encoded_images[0],
            "b64_pngs": encoded_images,
        }

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
