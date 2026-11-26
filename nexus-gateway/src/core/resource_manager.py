"""Exclusive single-GPU lock manager for serialized inference and language/image swaps."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class ResourceManager:
    """
    Serializes every GPU lifecycle and inference operation across FastAPI worker threads.
    Uses a non-thread-owned `threading.Lock` so sync streaming generators that acquire on
    Thread A and finish on Thread B release cleanly without thread-ownership errors.
    """

    def __init__(self) -> None:
        self._gate = threading.Lock()
        self._swapping = False

    @property
    def is_busy(self) -> bool:
        return self._gate.locked()

    @property
    def is_swapping(self) -> bool:
        return self._swapping

    @contextmanager
    def exclusive(self, *, swapping: bool = False) -> Iterator[None]:
        self._gate.acquire()
        if swapping:
            self._swapping = True
        try:
            yield
        finally:
            if swapping:
                self._swapping = False
            self._gate.release()
