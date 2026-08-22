import threading
import time

from PIL import Image

from bidan_lens.models import (
    AnalysisCandidate,
    BoundingBox,
    OcrDocument,
    PopupResult,
)
from bidan_lens.ocr.base import OcrEngine
from bidan_lens.ocr.hangul import make_line
from bidan_lens.pipeline.coordinator import FrameRequest, PipelineCoordinator


class FakeOcr(OcrEngine):
    def recognize(self, image, *, origin=(0, 0)):
        line = make_line("어디에서 먹어요", BoundingBox(0, 0, 140, 20), 0.99)
        return OcrDocument((line,), time.monotonic(), *origin)


class DelayedOcr(OcrEngine):
    def recognize(self, image, *, origin=(0, 0)):
        if image.getpixel((0, 0))[0] == 1:
            time.sleep(0.05)
        line = make_line("어디에서", BoundingBox(0, 0, 100, 20), 0.99)
        return OcrDocument((line,), time.monotonic(), *origin)


class FakeAnalyzer:
    def analyze(self, sentence, target_span, max_candidates=5):
        surface = sentence[slice(*target_span)]
        return (AnalysisCandidate(surface, "어디", 1.0),)


def test_mocked_ocr_to_popup_pipeline() -> None:
    results: list[PopupResult | None] = []
    ready = threading.Event()

    def receive(result):
        results.append(result)
        ready.set()

    coordinator = PipelineCoordinator(FakeOcr(), FakeAnalyzer(), receive)
    coordinator.start()
    line = make_line("어디에서 먹어요", BoundingBox(0, 0, 140, 20), 0.99)
    x, y = line.eojeols[0].box.center
    requested_at = time.monotonic()
    coordinator.submit(
        FrameRequest(
            Image.new("RGB", (140, 20)),
            (50, 70),
            (int(x + 50), int(y + 70)),
            requested_at=requested_at,
        )
    )
    assert ready.wait(1)
    coordinator.stop()
    assert results[0] is not None
    assert results[0].target.surface == "어디에서"
    assert results[0].selected.lemma == "어디"
    assert results[0].requested_at == requested_at


def test_in_flight_stale_result_is_not_delivered() -> None:
    results = []
    ready = threading.Event()

    def receive(result):
        results.append(result)
        ready.set()

    coordinator = PipelineCoordinator(DelayedOcr(), FakeAnalyzer(), receive)
    coordinator.start()
    slow = Image.new("RGB", (100, 20), (1, 0, 0))
    fast = Image.new("RGB", (100, 20), (2, 0, 0))
    coordinator.submit(FrameRequest(slow, (0, 0), (10, 10)))
    time.sleep(0.01)
    coordinator.submit(FrameRequest(fast, (0, 0), (10, 10)))
    assert ready.wait(1)
    coordinator.stop()
    assert len(results) == 1
