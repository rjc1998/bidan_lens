import numpy as np
import pytest
from PIL import Image

from bidan_lens.models import BoundingBox, OcrEojeol, OcrLine
from bidan_lens.ocr.base import DetectedRegion, RecognizedText
from bidan_lens.ocr.paddle import (
    PaddleOcrEngine,
    PaddleRecognizer,
    _normalize,
    _recover_isolated_close_word_pairs,
    _recover_isolated_overlapping_word_pairs,
    _recover_overlapping_word_triplets,
    _recover_terminal_overlapping_word_pair,
    _remove_tiny_contained_fragments,
)


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


class SupplementalContextRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def word_boxes(self, _image):
        return ((0, 15), (18, 25), (28, 38), (41, 48), (47, 70), (75, 100))

    def recognize(self, _image):
        values = (
            ('\uc624\ub298', 0.99),
            ('10', 0.99),
            ('EC', 0.99),
            ('K', 0.99),
            ('-2024/v1', 0.99),
            ('\ub9cc\ub098\uc694', 0.99),
        )
        text, confidence = values[self.calls]
        self.calls += 1
        return RecognizedText(text, confidence)


def test_engine_retains_numeric_abbreviation_and_joined_version_context() -> None:
    engine = PaddleOcrEngine(Detector(), SupplementalContextRecognizer())

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert document.lines[0].text == '\uc624\ub298 10 EC K-2024/v1 \ub9cc\ub098\uc694'
    assert [word.text for word in document.lines[0].eojeols] == [
        '\uc624\ub298',
        '\ub9cc\ub098\uc694',
    ]


class CollinearDetector:
    def detect(self, _image):
        return (
            DetectedRegion(BoundingBox(10, 5, 50, 25), 0.9),
            DetectedRegion(BoundingBox(55, 6, 110, 26), 0.9),
        )


class SequentialRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ('\uc624\ub298', '\ub9cc\ub098\uc694')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_reconstructs_collinear_detector_fragments_as_one_sentence() -> None:
    engine = PaddleOcrEngine(CollinearDetector(), SequentialRecognizer())

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert len(document.lines) == 1
    assert document.lines[0].text == '\uc624\ub298 \ub9cc\ub098\uc694'
    assert [
        (word.text, word.sentence_start, word.sentence_end)
        for word in document.lines[0].eojeols
    ] == [
        ('\uc624\ub298', 0, 2),
        ('\ub9cc\ub098\uc694', 3, 6),
    ]


class StructuredContextRegionRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ('\uc624\ub298', 'K-2026/v1')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_merges_separate_structured_context_without_hover_target() -> None:
    engine = PaddleOcrEngine(
        CollinearDetector(),
        StructuredContextRegionRecognizer(),
    )

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert document.lines[0].text == '\uc624\ub298 K-2026/v1'
    assert [word.text for word in document.lines[0].eojeols] == ['\uc624\ub298']


class OverlappingSequentialRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ('\uc624\ub298 \ud559\uad50\uc5d0', '\ud559\uad50\uc5d0 \ub9cc\ub098\uc694')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


class OverlappingDetector:
    def detect(self, _image):
        return (
            DetectedRegion(BoundingBox(10, 5, 60, 25), 0.9),
            DetectedRegion(BoundingBox(55, 6, 110, 26), 0.9),
        )


def test_engine_deduplicates_sentence_text_but_keeps_overlapping_word_geometry() -> None:
    engine = PaddleOcrEngine(OverlappingDetector(), OverlappingSequentialRecognizer())

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert document.lines[0].text == '\uc624\ub298 \ud559\uad50\uc5d0 \ub9cc\ub098\uc694'
    school_words = [
        word for word in document.lines[0].eojeols if word.text == '\ud559\uad50\uc5d0'
    ]
    assert len(school_words) == 2
    assert {
        (word.sentence_start, word.sentence_end) for word in school_words
    } == {(3, 6)}


class PunctuatedOverlapRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ('\ub610', '\ub610, \ub9cc\ub098\uc694')
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_deduplicates_punctuation_variants_only_for_overlapping_regions() -> None:
    engine = PaddleOcrEngine(OverlappingDetector(), PunctuatedOverlapRecognizer())

    document = engine.recognize(Image.new('RGB', (160, 60)))

    assert document.lines[0].text == '\ub610 \ub9cc\ub098\uc694'


def test_tiny_contained_fragment_is_removed_and_spans_are_repaired() -> None:
    line = OcrLine(
        '\ud559\uad50 \uc544 \ud559\uad50\uc5d0\uc11c',
        BoundingBox(0, 0, 120, 20),
        0.9,
        (
            OcrEojeol('\ud559\uad50', BoundingBox(0, 0, 30, 20), 0.9, 0, 2),
            OcrEojeol('\uc544', BoundingBox(50, 0, 55, 20), 0.8, 3, 4),
            OcrEojeol('\ud559\uad50\uc5d0\uc11c', BoundingBox(20, 0, 100, 20), 0.9, 5, 9),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == '\ud559\uad50 \ud559\uad50\uc5d0\uc11c'
    assert [(item.sentence_start, item.sentence_end) for item in cleaned.eojeols] == [
        (0, 2),
        (3, 7),
    ]


def test_contained_fragment_at_width_threshold_is_preserved() -> None:
    line = OcrLine(
        '\uc544 \ud559\uad50\uc5d0\uc11c',
        BoundingBox(0, 0, 100, 20),
        0.9,
        (
            OcrEojeol('\uc544', BoundingBox(20, 0, 29, 20), 0.8, 0, 1),
            OcrEojeol('\ud559\uad50\uc5d0\uc11c', BoundingBox(0, 0, 80, 20), 0.9, 2, 6),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


class TripletRecognizer:
    def recognize(self, _image):
        return RecognizedText('\ud559\uad50\uc5d0\uc11c\ub294', 0.999)


def test_overlapping_word_triplet_discards_only_confirmed_leading_sliver() -> None:
    words = [
        ('\uac00', BoundingBox(10, 0, 16, 20), 0.78),
        ('\ud559\uad50\uc5d0\uc11c', BoundingBox(15, 0, 75, 20), 0.999),
        ('\ub294', BoundingBox(74, 0, 94, 20), 0.93),
    ]

    recovered = _recover_overlapping_word_triplets(
        words,
        Image.new('RGB', (100, 20)),
        BoundingBox(0, 0, 100, 20),
        TripletRecognizer(),
    )

    assert recovered == [
        ('\ud559\uad50\uc5d0\uc11c\ub294', BoundingBox(10, 0, 94, 20), 0.93),
    ]


def test_overlapping_word_triplet_preserves_wider_leading_word() -> None:
    words = [
        ('\uac00', BoundingBox(5, 0, 15, 20), 0.78),
        ('\ud559\uad50\uc5d0\uc11c', BoundingBox(14, 0, 74, 20), 0.999),
        ('\ub294', BoundingBox(73, 0, 93, 20), 0.93),
    ]

    assert _recover_overlapping_word_triplets(
        words,
        Image.new('RGB', (100, 20)),
        BoundingBox(0, 0, 100, 20),
        TripletRecognizer(),
    ) == words


class ClosePairRecognizer:
    def recognize(self, _image):
        return RecognizedText('\ud559\uad50\uc5d0\uc11c\ub294', 0.9985)


class FinalSyllablePairRecognizer:
    def recognize(self, _image):
        return RecognizedText('\uc0ac\uc6a9\ud558\ub294', 0.9998)


def test_isolated_close_pair_merges_only_with_matching_pitch_and_wide_neighbors() -> None:
    words = [
        ('\uc624\ub298', BoundingBox(0, 0, 50, 30), 0.999),
        ('\ud559\uad50', BoundingBox(65, 0, 113, 30), 0.9998),
        ('\uc5d0\uc11c\ub294', BoundingBox(119, 0, 191, 30), 0.9994),
        ('\uac11\ub2c8\ub2e4', BoundingBox(206, 0, 278, 30), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (300, 30)),
        BoundingBox(0, 0, 300, 30),
        ClosePairRecognizer(),
    )

    assert recovered == [
        words[0],
        ('\ud559\uad50\uc5d0\uc11c\ub294', BoundingBox(65, 0, 191, 30), 0.9985),
        words[3],
    ]


def test_isolated_close_pair_preserves_pair_without_wide_following_gap() -> None:
    words = [
        ('\uc624\ub298', BoundingBox(0, 0, 50, 30), 0.999),
        ('\ud559\uad50', BoundingBox(65, 0, 113, 30), 0.9998),
        ('\uc5d0\uc11c\ub294', BoundingBox(119, 0, 191, 30), 0.9994),
        ('\uac11\ub2c8\ub2e4', BoundingBox(200, 0, 272, 30), 0.999),
    ]

    assert _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (300, 30)),
        BoundingBox(0, 0, 300, 30),
        ClosePairRecognizer(),
    ) == words


def test_isolated_close_pair_recovers_high_confidence_final_syllable() -> None:
    words = [
        ('\uc774\uac83', BoundingBox(0, 0, 32, 20), 0.9999),
        ('\uc0ac\uc6a9\ud558', BoundingBox(39, 0, 87, 20), 0.9999),
        ('\ub294', BoundingBox(90.4, 0, 104.4, 20), 0.9998),
        ('\ubc29\ubc95', BoundingBox(112.4, 0, 144.4, 20), 0.9999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (160, 20)),
        BoundingBox(0, 0, 160, 20),
        FinalSyllablePairRecognizer(),
    )

    assert recovered == [
        words[0],
        ('\uc0ac\uc6a9\ud558\ub294', BoundingBox(39, 0, 104.4, 20), 0.9998),
        words[3],
    ]


def test_isolated_close_pair_preserves_final_syllable_near_following_word() -> None:
    words = [
        ('\uc774\uac83', BoundingBox(0, 0, 32, 20), 0.9999),
        ('\uc0ac\uc6a9\ud558', BoundingBox(39, 0, 87, 20), 0.9999),
        ('\ub294', BoundingBox(90.4, 0, 104.4, 20), 0.9998),
        ('\ubc29\ubc95', BoundingBox(110.4, 0, 142.4, 20), 0.9999),
    ]

    assert _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (160, 20)),
        BoundingBox(0, 0, 160, 20),
        FinalSyllablePairRecognizer(),
    ) == words


def test_isolated_close_pair_preserves_lower_confidence_final_syllable() -> None:
    words = [
        ('\uc774\uac83', BoundingBox(0, 0, 32, 20), 0.9999),
        ('\uc0ac\uc6a9\ud558', BoundingBox(39, 0, 87, 20), 0.9999),
        ('\ub294', BoundingBox(90.4, 0, 104.4, 20), 0.9993),
        ('\ubc29\ubc95', BoundingBox(112.4, 0, 144.4, 20), 0.9999),
    ]

    assert _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (160, 20)),
        BoundingBox(0, 0, 160, 20),
        FinalSyllablePairRecognizer(),
    ) == words


class TerminalOverlapRecognizer:
    def recognize(self, _image):
        return RecognizedText('\uc9c0\uc6d0\ud55c\ub2e4', 0.99995)


def test_terminal_overlapping_pair_merges_with_exact_combined_recognition() -> None:
    words = [
        ('\uc774\ub97c', BoundingBox(0, 0, 60, 30), 0.9998),
        ('\uc9c0\uc6d0', BoundingBox(76, 0, 136, 30), 0.9997),
        ('\ud55c\ub2e4', BoundingBox(135, 0, 191, 30), 0.99995),
    ]

    recovered = _recover_terminal_overlapping_word_pair(
        words,
        Image.new('RGB', (200, 30)),
        BoundingBox(0, 0, 200, 30),
        TerminalOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ('\uc9c0\uc6d0\ud55c\ub2e4', BoundingBox(76, 0, 191, 30), 0.9997),
    ]


def test_terminal_overlapping_pair_preserves_internal_pair() -> None:
    words = [
        ('\uc774\ub97c', BoundingBox(0, 0, 60, 30), 0.9998),
        ('\uc9c0\uc6d0', BoundingBox(76, 0, 136, 30), 0.9997),
        ('\ud55c\ub2e4', BoundingBox(135, 0, 191, 30), 0.99995),
        ('\uc624\ub298', BoundingBox(210, 0, 270, 30), 0.9999),
    ]

    assert _recover_terminal_overlapping_word_pair(
        words,
        Image.new('RGB', (280, 30)),
        BoundingBox(0, 0, 280, 30),
        TerminalOverlapRecognizer(),
    ) == words


class IsolatedOverlapRecognizer:
    def recognize(self, _image):
        return RecognizedText('\uc9c0\uc6d0\ud558\ub294\ub2e4', 0.9958)


def test_isolated_overlapping_pair_merges_exact_two_plus_three_surface() -> None:
    words = [
        ('\uc774\ub97c', BoundingBox(0, 0, 30, 20), 0.999),
        ('\uc9c0\uc6d0', BoundingBox(39, 0, 68, 20), 0.9989),
        ('\ud558\ub294\ub2e4', BoundingBox(67, 0, 106, 20), 0.9994),
        ('\uc624\ub298', BoundingBox(114, 0, 144, 20), 0.999),
    ]

    recovered = _recover_isolated_overlapping_word_pairs(
        words,
        Image.new('RGB', (150, 20)),
        BoundingBox(0, 0, 150, 20),
        IsolatedOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ('\uc9c0\uc6d0\ud558\ub294\ub2e4', BoundingBox(39, 0, 106, 20), 0.9958),
        words[3],
    ]


def test_isolated_overlapping_pair_preserves_pair_near_following_word() -> None:
    words = [
        ('\uc774\ub97c', BoundingBox(0, 0, 30, 20), 0.999),
        ('\uc9c0\uc6d0', BoundingBox(39, 0, 68, 20), 0.9989),
        ('\ud558\ub294\ub2e4', BoundingBox(67, 0, 106, 20), 0.9994),
        ('\uc624\ub298', BoundingBox(112, 0, 142, 20), 0.999),
    ]

    assert _recover_isolated_overlapping_word_pairs(
        words,
        Image.new('RGB', (150, 20)),
        BoundingBox(0, 0, 150, 20),
        IsolatedOverlapRecognizer(),
    ) == words


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
