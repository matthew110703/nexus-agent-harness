"""Single-active-model resource manager serializing chat and diffusion workloads."""

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
    """Ensures at most one chat model or diffusion pipeline is resident at a time."""

    def __init__(self) -> None:
        self._gate = threading.RLock()
        self._active: Optional[ActiveHandle] = None
        self.unload_history: list[str] = []

    @property
    def active_model_id(self) -> Optional[str]:
        with self._gate:
            return self._active.model_id if self._active is not None else None

    @property
    def active_kind(self) -> Optional[ActiveKind]:
        with self._gate:
            return self._active.kind if self._active is not None else None

    def unload_active(self) -> None:
        with self._gate:
            if self._active is None:
                return
            handle = self._active
            self._active = None
            try:
                if handle.unload is not None:
                    handle.unload(handle.instance)
            finally:
                self.unload_history.append(handle.model_id)
                gc.collect()
                try:
                    import torch  # type: ignore[import-not-found]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    @contextmanager
    def acquire(
        self,
        kind: ActiveKind,
        model_id: str,
        loader: Callable[[], Any],
        unload: Optional[Callable[[Any], None]] = None,
    ) -> Iterator[Any]:
        with self._gate:
            if self._active is not None and (self._active.kind != kind or self._active.model_id != model_id):
                self.unload_active()
            if self._active is None:
                instance = loader()
                self._active = ActiveHandle(kind=kind, model_id=model_id, instance=instance, unload=unload)
            yield self._active.instance
