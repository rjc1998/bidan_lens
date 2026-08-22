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


class WordSegmentingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def word_boxes(self, _image):
        return ((0, 40), (50, 100))

    def recognize(self, _image):
        values = ('\uc624\ub298', '\ub9cc\ub098\uc694')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_reconstructs_sentence_and_exact_word_geometry() -> None:
    recognizer = WordSegmentingRecognizer()
    engine = PaddleOcrEngine(Detector(), recognizer)

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert recognizer.calls == 2
    assert document.lines[0].text == '\uc624\ub298 \ub9cc\ub098\uc694'
    assert [word.text for word in document.lines[0].eojeols] == [
        '\uc624\ub298',
        '\ub9cc\ub098\uc694',
    ]
    assert document.lines[0].eojeols[0].box == BoundingBox(10, 5, 50, 35)
    assert document.lines[0].eojeols[1].box == BoundingBox(60, 5, 110, 35)


class ContextSegmentingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def word_boxes(self, _image):
        return ((0, 25), (30, 65), (70, 100))

    def recognize(self, _image):
        values = ('\uc624\ub298', 'K-2026/v1', '\ub9cc\ub098\uc694')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_retains_structured_ascii_context_without_a_hover_target() -> None:
    engine = PaddleOcrEngine(Detector(), ContextSegmentingRecognizer())

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert document.lines[0].text == '\uc624\ub298 K-2026/v1 \ub9cc\ub098\uc694'
    assert [word.text for word in document.lines[0].eojeols] == [
        '\uc624\ub298',
        '\ub9cc\ub098\uc694',
    ]


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


class SegmentingSession:
    def get_inputs(self):
        return [type('Input', (), {'name': 'x'})()]

    def run(self, _outputs, _inputs):
        output = np.zeros((1, 10, 3), dtype=np.float32)
        output[0, :, 0] = 1.0
        for timestep in (1, 4, 8):
            output[0, timestep] = (0.0, 1.0, 0.0)
        for timestep in (3, 6):
            output[0, timestep] = (0.0, 0.0, 1.0)
        return [output]


def test_recognizer_uses_ctc_space_probabilities_for_word_boxes(tmp_path) -> None:
    characters = tmp_path / 'characters.txt'
    characters.write_text('\uac00\n', encoding='utf-8')
    recognizer = PaddleRecognizer(
        tmp_path / 'unused.onnx', characters, session=SegmentingSession()
    )

    boxes = recognizer.word_boxes(Image.new('RGB', (100, 20), (255, 0, 0)))

    assert len(boxes) == 3
    assert boxes[0][0] == 0
    assert boxes[-1][1] == 100
    assert all(left < right for left, right in boxes)


def test_recognizer_splits_a_ctc_segment_at_a_wide_visual_gap(tmp_path) -> None:
    characters = tmp_path / 'characters.txt'
    characters.write_text('\uac00\n', encoding='utf-8')
    recognizer = PaddleRecognizer(
        tmp_path / 'unused.onnx', characters, session=RecognitionSession()
    )
    image = Image.new('RGB', (100, 20), (255, 255, 255))
    image.paste((0, 0, 0), (5, 4, 35, 16))
    image.paste((0, 0, 0), (50, 4, 95, 16))

    boxes = recognizer.word_boxes(image)

    assert len(boxes) == 2
    assert boxes[0][1] <= 35
    assert boxes[1][0] >= 50


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


def test_recognizer_expands_tensor_for_wide_text_lines(tmp_path) -> None:
    characters = tmp_path / 'characters.txt'
    characters.write_text('\uac00\n', encoding='utf-8')
    session = RecognitionSession()
    recognizer = PaddleRecognizer(tmp_path / 'unused.onnx', characters, session=session)

    recognizer.recognize(Image.new('RGB', (1000, 20), (255, 0, 0)))

    assert session.tensor.shape == (1, 3, 48, 2400)
