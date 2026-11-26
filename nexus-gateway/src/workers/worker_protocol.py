"""Typed protocol messages between Nexus Gateway API and the isolated GPU model worker (v1.1.0)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

WorkerKind = Literal["language", "image"]
ImageQualityProfile = Literal["normal", "detailed", "maximum"]


@dataclass
class WorkerLoadRequest:
    kind: WorkerKind
    model_id: str
    model_path: str
    implementation: str
    architecture: str = "unknown"
    precision: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerGenerateRequest:
    model_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = "auto"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    reasoning_effort: Optional[str] = None
    enable_thinking: Optional[bool] = None
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerImageRequest:
    model_id: str
    prompt: str
    operation_id: str = ""
    quality: ImageQualityProfile = "detailed"
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    num_images: int = 1
    output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerStatus:
    pid: int
    alive: bool
    active_kind: Optional[WorkerKind] = None
    active_model_id: Optional[str] = None
    cuda_allocated_bytes: int = 0
    cuda_reserved_bytes: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
