"""Single-active-model GPU resource manager with live worker-state reconciliation."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional

ActiveKind = Literal["chat", "diffusion"]


@dataclass
class ActiveHandle:
    kind: ActiveKind
    model_id: str
    instance: Any
    unload: Optional[Callable[[Any], None]] = None


class ResourceManager:
    """
    Ensures at most one chat model or diffusion pipeline is resident on the GPU at a time.
    Uses a non-thread-owned `threading.Lock` so FastAPI/Starlette sync generators that
    acquire on Thread A and finish on Thread B never raise `cannot release un-acquired lock`.
    Reconciles against live worker status so a crashed/restarted worker never leaves a stale handle.
    """

    def __init__(self) -> None:
        self._gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: Optional[ActiveHandle] = None
        self.unload_history: list[str] = []

    @property
    def is_busy(self) -> bool:
        return self._gate.locked()

    @property
    def active_model_id(self) -> Optional[str]:
        with self._state_lock:
            return self._active.model_id if self._active is not None else None

    @property
    def active_kind(self) -> Optional[ActiveKind]:
        with self._state_lock:
            return self._active.kind if self._active is not None else None

    def invalidate_if_mismatched(
        self,
        *,
        worker_alive: bool,
        worker_active_kind: Optional[str],
        worker_active_model_id: Optional[str],
    ) -> bool:
        """Clear stale cached handle if the worker died, restarted, or holds a different model."""
        with self._state_lock:
            if self._active is None:
                return False
            if (
                not worker_alive
                or worker_active_kind != self._active.kind
                or worker_active_model_id != self._active.model_id
            ):
                self._active = None
                return True
            return False

    def _unload_unlocked(self) -> None:
        with self._state_lock:
            handle = self._active
            self._active = None
        if handle is None:
            return
        try:
            if handle.unload is not None:
                handle.unload(handle.instance)
        finally:
            with self._state_lock:
                self.unload_history.append(handle.model_id)
            gc.collect()
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    def unload_active(self) -> None:
        with self._gate:
            self._unload_unlocked()

    @contextmanager
    def acquire(
        self,
        kind: ActiveKind,
        model_id: str,
        loader: Callable[[], Any],
        unload: Optional[Callable[[Any], None]] = None,
        verify_live: Optional[Callable[[], tuple[bool, Optional[str], Optional[str]]]] = None,
    ) -> Iterator[Any]:
        self._gate.acquire()
        try:
            if verify_live is not None:
                alive, w_kind, w_model = verify_live()
                self.invalidate_if_mismatched(
                    worker_alive=alive,
                    worker_active_kind=w_kind,
                    worker_active_model_id=w_model,
                )
            with self._state_lock:
                current = self._active
            if current is not None and (current.kind != kind or current.model_id != model_id):
                self._unload_unlocked()
            with self._state_lock:
                current = self._active
            if current is None:
                instance = loader()
                with self._state_lock:
                    self._active = ActiveHandle(kind=kind, model_id=model_id, instance=instance, unload=unload)
                    current = self._active
            yield current.instance
        finally:
            self._gate.release()
