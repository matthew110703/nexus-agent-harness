"""Dynamic hardware discovery for Colab and local GPU/CPU runtimes."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

GIB = 1024 ** 3
L4_BASELINE_VRAM_GIB = 22.0


@dataclass
class GpuDeviceInfo:
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: str
    bf16_supported: bool = False


@dataclass
class HardwareSnapshot:
    platform: str
    python_version: str
    cpu_count: int
    system_ram_total_bytes: int
    system_ram_available_bytes: int
    local_disk_path: str
    local_disk_total_bytes: int
    local_disk_free_bytes: int
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    total_vram_bytes: int = 0
    free_vram_bytes: int = 0
    gpus: list[GpuDeviceInfo] = field(default_factory=list)
    tier: str = "cpu_only"
    package_versions: dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _system_memory() -> tuple[int, int]:
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        return 0, 0


def classify_hardware_tier(cuda_available: bool, total_vram_bytes: int) -> str:
    if not cuda_available or total_vram_bytes <= 0:
        return "cpu_only"
    gib = total_vram_bytes / GIB
    if gib >= 70.0:
        return "high_vram_datacenter"  # A100-80GB / H100 / multi-GPU
    if gib >= 36.0:
        return "mid_vram_datacenter"  # A100-40GB / L40S
    if gib >= L4_BASELINE_VRAM_GIB:
        return "l4_baseline"  # L4 24GB / RTX 3090/4090
    return "sub_baseline_gpu"


def detect_hardware(disk_path: Optional[Path] = None) -> HardwareSnapshot:
    """Detect live GPU, CUDA, RAM, disk, and ML runtime versions without assuming L4."""
    check_path = disk_path or Path("/content" if Path("/content").exists() else ".")
    try:
        usage = shutil.disk_usage(check_path)
        disk_total, disk_free = int(usage.total), int(usage.free)
    except OSError:
        disk_total, disk_free = 0, 0

    ram_total, ram_avail = _system_memory()

    cuda_available = False
    cuda_version: Optional[str] = None
    gpus: list[GpuDeviceInfo] = []
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                try:
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                except Exception:
                    free_b, total_b = 0, int(getattr(props, "total_memory", 0))
                gpus.append(
                    GpuDeviceInfo(
                        index=idx,
                        name=str(getattr(props, "name", f"GPU-{idx}")),
                        total_vram_bytes=int(total_b),
                        free_vram_bytes=int(free_b),
                        compute_capability=f"{getattr(props, 'major', 0)}.{getattr(props, 'minor', 0)}",
                        bf16_supported=bf16_ok,
                    )
                )
    except Exception:
        pass

    total_vram = sum(g.total_vram_bytes for g in gpus)
    free_vram = sum(g.free_vram_bytes for g in gpus)
    tier = classify_hardware_tier(cuda_available, total_vram)

    pkgs = {
        pkg: _pkg_version(pkg)
        for pkg in ("torch", "transformers", "diffusers", "accelerate", "bitsandbytes", "fastapi", "uvicorn")
    }

    return HardwareSnapshot(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        system_ram_total_bytes=ram_total,
        system_ram_available_bytes=ram_avail,
        local_disk_path=str(check_path),
        local_disk_total_bytes=disk_total,
        local_disk_free_bytes=disk_free,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_count=len(gpus),
        total_vram_bytes=total_vram,
        free_vram_bytes=free_vram,
        gpus=gpus,
        tier=tier,
        package_versions=pkgs,
    )
