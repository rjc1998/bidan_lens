from __future__ import annotations

import queue
from contextlib import suppress


class LatestQueue[T]:
    """A one-value queue that discards stale work in interactive pipelines."""

    def __init__(self) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=1)

    def put(self, value: T) -> None:
        try:
            self._queue.put_nowait(value)
            return
        except queue.Full:
            pass
        with suppress(queue.Empty):
            self._queue.get_nowait()
        self._queue.put_nowait(value)

    def get(self, timeout: float | None = None) -> T:
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()
