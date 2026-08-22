from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.latest_queue import LatestQueue
from bidan_lens.models import HoverTarget, PopupResult
from bidan_lens.ocr.base import OcrEngine
from bidan_lens.pipeline.hit_test import hit_test


@dataclass(frozen=True, slots=True)
class FrameRequest:
    image: Image.Image
    origin: tuple[int, int]
    pointer: tuple[int, int]
    requested_at: float = field(default_factory=time.monotonic)


class PipelineCoordinator:
    """Latest-frame asynchronous OCR pipeline; stale pointer frames are discarded."""

    def __init__(
        self,
        ocr: OcrEngine,
        analyzer: KoreanAnalyzer,
        on_result: Callable[[PopupResult | None], None],
    ) -> None:
        self.ocr = ocr
        self.analyzer = analyzer
        self.on_result = on_result
        self._queue: LatestQueue[tuple[int, FrameRequest] | None] = LatestQueue()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._generation_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="bidan-ocr", daemon=True)
        self._thread.start()

    def submit(self, request: FrameRequest) -> None:
        with self._generation_lock:
            self._generation += 1
            generation = self._generation
        self._queue.put((generation, request))

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            queued = self._queue.get()
            if queued is None:
                return
            generation, request = queued
            try:
                document = self.ocr.recognize(request.image, origin=request.origin)
                target = hit_test(document, *request.pointer)
                with self._generation_lock:
                    current = generation == self._generation
                if current:
                    self.on_result(self._resolve(target, request.requested_at) if target else None)
            except Exception:
                # The GUI logs only exception types; never OCR text or screenshot pixels.
                with self._generation_lock:
                    current = generation == self._generation
                if current:
                    self.on_result(None)

    def _resolve(self, target: HoverTarget, requested_at: float) -> PopupResult:
        candidates = self.analyzer.analyze(
            target.sentence,
            (target.sentence_start, target.sentence_end),
        )
        return PopupResult(target, candidates, requested_at=requested_at)
