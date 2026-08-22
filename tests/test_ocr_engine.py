import numpy as np
import pytest
from PIL import Image

from bidan_lens.models import BoundingBox
from bidan_lens.ocr.base import DetectedRegion, RecognizedText
from bidan_lens.ocr.paddle import PaddleOcrEngine, PaddleRecognizer, _normalize


class Detector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(10, 5, 110, 35), 0.9),)


class RetryingRecognizer:
    def __init__(self):
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        if self.calls == 1:
            return RecognizedText("어디에서", 0.5)
        return RecognizedText("어디에서", 0.96)


def test_engine_retries_once_and_returns_hangul_document() -> None:
    recognizer = RetryingRecognizer()
    engine = PaddleOcrEngine(Detector(), recognizer)
    document = engine.recognize(Image.new("RGB", (160, 60)), origin=(200, 300))
    assert recognizer.calls == 2
    assert document.origin_x == 200 and document.origin_y == 300
    assert document.lines[0].eojeols[0].text == "어디에서"
    assert document.lines[0].confidence == 0.96


class LatinRecognizer:
    def recognize(self, _image):
        return RecognizedText("English", 0.99)


def test_engine_rejects_non_hangul_text() -> None:
    engine = PaddleOcrEngine(Detector(), LatinRecognizer())
    assert not engine.recognize(Image.new("RGB", (160, 60))).lines


def test_detector_normalization_converts_rgb_to_bgr() -> None:
    tensor = _normalize(Image.new("RGB", (1, 1), (255, 0, 0)))

    assert tensor[0, 0, 0, 0] == pytest.approx(-0.485 / 0.229)
    assert tensor[0, 2, 0, 0] == pytest.approx((1.0 - 0.406) / 0.225)


class RecognitionSession:
    def __init__(self) -> None:
        self.tensor = None

    def get_inputs(self):
        return [type("Input", (), {"name": "x"})()]

    def run(self, _outputs, inputs):
        self.tensor = inputs["x"]
        return [np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)]


def test_recognizer_uses_fixed_zero_padded_bgr_tensor(tmp_path) -> None:
    characters = tmp_path / "characters.txt"
    characters.write_text("가\n", encoding="utf-8")
    session = RecognitionSession()
    recognizer = PaddleRecognizer(tmp_path / "unused.onnx", characters, session=session)

    result = recognizer.recognize(Image.new("RGB", (10, 10), (255, 0, 0)))

    assert result.text == "가"
    assert result.confidence == pytest.approx(1.0)
    assert session.tensor.shape == (1, 3, 48, 320)
    assert session.tensor[0, 0, 0, 0] == pytest.approx(-1.0)
    assert session.tensor[0, 2, 0, 0] == pytest.approx(1.0)
    assert np.all(session.tensor[:, :, :, 48:] == 0)
