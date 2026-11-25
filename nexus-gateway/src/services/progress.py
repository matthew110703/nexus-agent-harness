"""First-class runtime progress tracking for discovery, staging, loading, and denoising."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

OperationKind = Literal[
    "chat_model_load",
    "chat_generation",
    "diffusion_pipeline_load",
    "image_generation",
    "model_staging",
]
OperationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
OperationPhase = Literal[
    "resolving",
    "discovery",
    "staging",
    "checkpoint_validation",
    "processor_init",
    "weight_loading",
    "placement",
    "ready",
    "generating",
    "denoising",
    "saving",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProgressRecord:
    operation_id: str
    kind: OperationKind
    model: str
    status: OperationStatus
    phase: OperationPhase
    progress: float = 0.0
    indeterminate: bool = False
    bytes_copied: int = 0
    total_bytes: int = 0
    source: str = ""
    destination: str = ""
    step: int = 0
    steps_total: int = 0
    message: str = ""
    started_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressService:
    """Thread-safe bounded registry of active and recent runtime operations."""

    def __init__(self, max_records: int = 64) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, ProgressRecord] = {}
        self._max_records = max_records

    def start(
        self,
        kind: OperationKind,
        model: str,
        phase: OperationPhase = "discovery",
        *,
        operation_id: Optional[str] = None,
        indeterminate: bool = False,
        source: str = "",
        destination: str = "",
        message: str = "",
    ) -> ProgressRecord:
        now = _utc_now_iso()
        op_id = operation_id or f"op_{uuid.uuid4().hex[:12]}"
        record = ProgressRecord(
            operation_id=op_id,
            kind=kind,
            model=model,
            status="running",
            phase=phase,
            indeterminate=indeterminate,
            source=source,
            destination=destination,
            message=message,
            started_at=now,
            updated_at=now,
        )
        with self._lock:
            self._records[op_id] = record
            while len(self._records) > self._max_records:
                del self._records[next(iter(self._records))]
        return record

    def update(
        self,
        operation_id: str,
        *,
        status: Optional[OperationStatus] = None,
        phase: Optional[OperationPhase] = None,
        progress: Optional[float] = None,
        indeterminate: Optional[bool] = None,
        bytes_copied: Optional[int] = None,
        total_bytes: Optional[int] = None,
        step: Optional[int] = None,
        steps_total: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[ProgressRecord]:
        with self._lock:
            record = self._records.get(operation_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if phase is not None:
                record.phase = phase
            if indeterminate is not None:
                record.indeterminate = indeterminate
            if bytes_copied is not None:
                record.bytes_copied = bytes_copied
            if total_bytes is not None:
                record.total_bytes = total_bytes
            if step is not None:
                record.step = step
            if steps_total is not None:
                record.steps_total = steps_total
            if progress is not None:
                record.progress = max(0.0, min(100.0, float(progress)))
            elif record.total_bytes > 0 and record.phase == "staging":
                record.progress = min(100.0, round(record.bytes_copied / record.total_bytes * 100.0, 2))
            elif record.steps_total > 0 and record.phase == "denoising":
                record.progress = min(100.0, round(record.step / record.steps_total * 100.0, 2))
            if message is not None:
                record.message = message
            if error is not None:
                record.error = error
            record.updated_at = _utc_now_iso()
            return record

    def complete(self, operation_id: str, phase: OperationPhase = "ready", message: str = "Ready") -> Optional[ProgressRecord]:
        return self.update(operation_id, status="completed", phase=phase, progress=100.0, indeterminate=False, message=message)

    def fail(self, operation_id: str, error: str) -> Optional[ProgressRecord]:
        return self.update(operation_id, status="failed", indeterminate=False, error=error, message=error)

    def find_active(self, kind: OperationKind, model: str) -> Optional[ProgressRecord]:
        with self._lock:
            for record in reversed(list(self._records.values())):
                if record.kind == kind and record.model == model and record.status in ("queued", "running"):
                    return record
        return None

    def get(self, operation_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            rec = self._records.get(operation_id)
            return rec.to_dict() if rec is not None else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = [r.to_dict() for r in self._records.values()]
        latest_chat = next(
            (r for r in reversed(records) if r["kind"] in ("chat_model_load", "chat_generation", "model_staging")),
            None,
        )
        latest_diffusion = next(
            (r for r in reversed(records) if r["kind"] in ("diffusion_pipeline_load", "image_generation")),
            None,
        )
        return {"chat": latest_chat, "diffusion": latest_diffusion, "operations": records[-16:]}
