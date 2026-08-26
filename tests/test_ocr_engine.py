import numpy as np
import pytest
from PIL import Image

from bidan_lens.models import BoundingBox, OcrEojeol, OcrLine
from bidan_lens.ocr.base import DetectedRegion, RecognizedText
from bidan_lens.ocr.hangul import make_line
from bidan_lens.ocr.paddle import (
    PaddleOcrEngine,
    PaddleRecognizer,
    _discard_confirmed_overlapping_character_duplicates,
    _merge_line_group,
    _normalize,
    _recover_confirmed_three_plus_five_splits,
    _recover_confirmed_three_plus_three_splits,
    _recover_initial_overlapping_word_pair,
    _recover_isolated_close_word_pairs,
    _recover_isolated_overlapping_word_pairs,
    _recover_overlapping_suffix_pairs,
    _recover_overlapping_word_triplets,
    _recover_relative_gap_two_plus_two_pairs,
    _recover_terminal_overlapping_word_pair,
    _recover_word_boundaries,
    _remove_tiny_contained_fragments,
    _split_mandatory_auxiliary_spacing,
    _split_punctuation_wrapped_word,
    _split_trailing_punctuation_boundary,
)


class Detector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(10, 5, 110, 35), 0.9),)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/\uc2e0\uc784/\uace0", ["/\uc2e0\uc784/", "\uace0"]),
        (
            "\uc5f0\u2014\uc74c\uc2dd\uc810\u2014\uc788\uc5b4",
            ["\uc5f0", "\u2014\uc74c\uc2dd\uc810\u2014", "\uc788\uc5b4"],
        ),
        (
            "\ubbfc\uc8fc\ub2f9\uc5d0\uc120\u2014\ub610\u2014\uc720\ud544\uc6b0",
            ["\ubbfc\uc8fc\ub2f9\uc5d0\uc120", "\u2014\ub610\u2014", "\uc720\ud544\uc6b0"],
        ),
        (
            "\u201c\uc218\ub2e8\uc774\u201d\ud30c\uad34\ub41c",
            ["\u201c\uc218\ub2e8\uc774\u201d", "\ud30c\uad34\ub41c"],
        ),
        (
            "\ubc1c\uba85\uacfc\u2018\uc9c0\ub09c\ub0a0\uc758\u2019\ud559\uc2dd\uc790",
            ["\ubc1c\uba85\uacfc", "\u2018\uc9c0\ub09c\ub0a0\uc758\u2019", "\ud559\uc2dd\uc790"],
        ),
    ],
)
def test_paired_punctuation_recovers_wrapped_word_boundaries(
    text: str,
    expected: list[str],
) -> None:
    box = BoundingBox(10, 5, 10 + len(text) * 20, 35)

    parts = _split_punctuation_wrapped_word(text, box, 0.91)

    assert [part[0] for part in parts] == expected
    assert parts[0][1].left == box.left
    assert parts[-1][1].right == box.right
    assert all(part[2] == 0.91 for part in parts)


def test_unpaired_punctuation_does_not_split_word() -> None:
    box = BoundingBox(10, 5, 110, 35)

    assert _split_punctuation_wrapped_word("\ud55c-\uad6d", box, 0.91) == [
        ("\ud55c-\uad6d", box, 0.91)
    ]


def test_wrapper_does_not_split_single_syllable_trailing_particle() -> None:
    text = '\u201c\uc720\ub2c8\uc628\ud504\ub808\uc2a4\u201d\ub294'
    box = BoundingBox(10, 5, 210, 35)

    assert _split_punctuation_wrapped_word(text, box, 0.91) == [
        (text, box, 0.91)
    ]


def test_slash_wrapper_can_split_single_syllable_following_word() -> None:
    text = '/\uc2e0\uc784/\uace0'
    box = BoundingBox(10, 5, 130, 35)

    assert [
        part[0] for part in _split_punctuation_wrapped_word(text, box, 0.91)
    ] == ['/\uc2e0\uc784/', '\uace0']


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\uc568\ubc94:\ucee4\ubc84", ["\uc568\ubc94:", "\ucee4\ubc84"]),
        ("\uadfc\uac70\ub85c?\ud55c", ["\uadfc\uac70\ub85c?", "\ud55c"]),
        ("\ud588\ub2e4!!\uadf8\ub7ec\ub098", ["\ud588\ub2e4!!", "\uadf8\ub7ec\ub098"]),
    ],
)
def test_terminal_punctuation_recovers_following_word_boundary(
    text: str,
    expected: list[str],
) -> None:
    box = BoundingBox(10, 5, 10 + len(text) * 20, 35)

    parts = _split_trailing_punctuation_boundary(text, box, 0.91)

    assert [part[0] for part in parts] == expected
    assert parts[0][1].left == box.left
    assert parts[-1][1].right == box.right


def test_internal_colon_without_following_hangul_does_not_split() -> None:
    box = BoundingBox(10, 5, 110, 35)

    assert _split_trailing_punctuation_boundary("K:2026", box, 0.91) == [("K:2026", box, 0.91)]


def test_mandatory_auxiliary_spacing_is_recovered_with_terminal_punctuation() -> None:
    text = "\ub450\uc5b4\uc57c\ud588\ub2e4!"
    box = BoundingBox(10, 5, 130, 35)

    parts = _split_mandatory_auxiliary_spacing(text, box, 0.91)

    assert [part[0] for part in parts] == ["\ub450\uc5b4\uc57c", "\ud588\ub2e4!"]
    assert parts[0][1] == BoundingBox(10, 5, 70, 35)
    assert parts[1][1] == BoundingBox(70, 5, 130, 35)


@pytest.mark.parametrize("text", ["\uc57c\ud588\ub2e4", "\uc774\uc57c\uae30\ud588\ub2e4"])
def test_auxiliary_spacing_does_not_split_unrelated_words(text: str) -> None:
    box = BoundingBox(10, 5, 110, 35)

    assert _split_mandatory_auxiliary_spacing(text, box, 0.91) == [(text, box, 0.91)]


class RetryingRecognizer:
    def __init__(self):
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        if self.calls == 1:
            return RecognizedText("어디에서", 0.5)
        return RecognizedText("어디에서", 0.96)


class SingleSegmentAuxiliaryRecognizer:
    def word_boxes(self, _image):
        return ((0, 100),)

    def recognize(self, _image):
        return RecognizedText("\ub450\uc5b4\uc57c\ud588\ub2e4!", 0.99)


def test_engine_retries_once_and_returns_hangul_document() -> None:
    recognizer = RetryingRecognizer()
    engine = PaddleOcrEngine(Detector(), recognizer)
    document = engine.recognize(Image.new("RGB", (160, 60)), origin=(200, 300))
    assert recognizer.calls == 2
    assert document.origin_x == 200 and document.origin_y == 300
    assert document.lines[0].eojeols[0].text == "어디에서"
    assert document.lines[0].confidence == 0.96


def test_engine_recovers_auxiliary_spacing_on_single_segment_line() -> None:
    engine = PaddleOcrEngine(Detector(), SingleSegmentAuxiliaryRecognizer())

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\ub450\uc5b4\uc57c \ud588\ub2e4!"
    assert [word.text for word in document.lines[0].eojeols] == [
        "\ub450\uc5b4\uc57c",
        "\ud588\ub2e4",
    ]


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
        values = ("\uc624\ub298", "\ub9cc\ub098\uc694")
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_reconstructs_sentence_and_exact_word_geometry() -> None:
    recognizer = WordSegmentingRecognizer()
    engine = PaddleOcrEngine(Detector(), recognizer)

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert recognizer.calls == 2
    assert document.lines[0].text == "\uc624\ub298 \ub9cc\ub098\uc694"
    assert [word.text for word in document.lines[0].eojeols] == [
        "\uc624\ub298",
        "\ub9cc\ub098\uc694",
    ]
    assert document.lines[0].eojeols[0].box == BoundingBox(10, 5, 50, 35)
    assert document.lines[0].eojeols[1].box == BoundingBox(60, 5, 110, 35)


class ContextSegmentingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def word_boxes(self, _image):
        return ((0, 25), (30, 65), (70, 100))

    def recognize(self, _image):
        values = ("\uc624\ub298", "K-2026/v1", "\ub9cc\ub098\uc694")
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_retains_structured_ascii_context_without_a_hover_target() -> None:
    engine = PaddleOcrEngine(Detector(), ContextSegmentingRecognizer())

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\uc624\ub298 K-2026/v1 \ub9cc\ub098\uc694"
    assert [word.text for word in document.lines[0].eojeols] == [
        "\uc624\ub298",
        "\ub9cc\ub098\uc694",
    ]


class SupplementalContextRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def word_boxes(self, _image):
        return ((0, 15), (18, 25), (28, 38), (41, 48), (47, 70), (75, 100))

    def recognize(self, _image):
        values = (
            ("\uc624\ub298", 0.99),
            ("10", 0.99),
            ("EC", 0.99),
            ("K", 0.99),
            ("-2024/v1", 0.99),
            ("\ub9cc\ub098\uc694", 0.99),
        )
        text, confidence = values[self.calls]
        self.calls += 1
        return RecognizedText(text, confidence)


def test_engine_retains_numeric_abbreviation_and_joined_version_context() -> None:
    engine = PaddleOcrEngine(Detector(), SupplementalContextRecognizer())

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\uc624\ub298 10 EC K-2024/v1 \ub9cc\ub098\uc694"
    assert [word.text for word in document.lines[0].eojeols] == [
        "\uc624\ub298",
        "\ub9cc\ub098\uc694",
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
        values = ("\uc624\ub298", "\ub9cc\ub098\uc694")
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_reconstructs_collinear_detector_fragments_as_one_sentence() -> None:
    engine = PaddleOcrEngine(CollinearDetector(), SequentialRecognizer())

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert len(document.lines) == 1
    assert document.lines[0].text == "\uc624\ub298 \ub9cc\ub098\uc694"
    assert [
        (word.text, word.sentence_start, word.sentence_end) for word in document.lines[0].eojeols
    ] == [
        ("\uc624\ub298", 0, 2),
        ("\ub9cc\ub098\uc694", 3, 6),
    ]


class StructuredContextRegionRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ("\uc624\ub298", "K-2026/v1")
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_merges_separate_structured_context_without_hover_target() -> None:
    engine = PaddleOcrEngine(
        CollinearDetector(),
        StructuredContextRegionRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\uc624\ub298 K-2026/v1"
    assert [word.text for word in document.lines[0].eojeols] == ["\uc624\ub298"]


class OverlappingSequentialRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ("\uc624\ub298 \ud559\uad50\uc5d0", "\ud559\uad50\uc5d0 \ub9cc\ub098\uc694")
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

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\uc624\ub298 \ud559\uad50\uc5d0 \ub9cc\ub098\uc694"
    school_words = [word for word in document.lines[0].eojeols if word.text == "\ud559\uad50\uc5d0"]
    assert len(school_words) == 2
    assert {(word.sentence_start, word.sentence_end) for word in school_words} == {(3, 6)}


class PunctuatedOverlapRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        values = ("\ub610", "\ub610, \ub9cc\ub098\uc694")
        value = values[self.calls]
        self.calls += 1
        return RecognizedText(value, 0.99)


def test_engine_deduplicates_punctuation_variants_only_for_overlapping_regions() -> None:
    engine = PaddleOcrEngine(OverlappingDetector(), PunctuatedOverlapRecognizer())

    document = engine.recognize(Image.new("RGB", (160, 60)))

    assert document.lines[0].text == "\ub610 \ub9cc\ub098\uc694"


def test_exact_spatial_duplicate_line_reuses_existing_sentence_span() -> None:
    complete = make_line(
        '\uac00 \ub098\ub2e4',
        BoundingBox(0, 0, 80, 20),
        0.99,
        (
            BoundingBox(0, 0, 20, 20),
            BoundingBox(20, 0, 30, 20),
            BoundingBox(30, 0, 50, 20),
            BoundingBox(50, 0, 70, 20),
        ),
    )
    duplicate = make_line(
        '\uac00',
        BoundingBox(2, 1, 21, 19),
        0.99,
        (BoundingBox(2, 1, 21, 19),),
    )

    merged = _merge_line_group([complete, duplicate])

    assert merged.text == '\uac00 \ub098\ub2e4'
    assert [item.text for item in merged.eojeols] == ['\uac00', '\ub098\ub2e4', '\uac00']
    assert [(item.sentence_start, item.sentence_end) for item in merged.eojeols] == [
        (0, 1),
        (2, 4),
        (0, 1),
    ]


def test_spatial_duplicate_requires_strong_horizontal_overlap() -> None:
    complete = make_line(
        '\uac00 \ub098\ub2e4',
        BoundingBox(0, 0, 80, 20),
        0.99,
        (
            BoundingBox(0, 0, 20, 20),
            BoundingBox(20, 0, 30, 20),
            BoundingBox(30, 0, 50, 20),
            BoundingBox(50, 0, 70, 20),
        ),
    )
    separate = make_line(
        '\uac00',
        BoundingBox(15, 1, 34, 19),
        0.99,
        (BoundingBox(15, 1, 34, 19),),
    )

    merged = _merge_line_group([complete, separate])

    assert merged.text == '\uac00 \ub098\ub2e4 \uac00'
    assert merged.eojeols[-1].sentence_start == 5
    assert merged.eojeols[-1].sentence_end == 6


def test_structured_suffix_and_leading_word_artifacts_are_removed_together() -> None:
    complete = OcrLine(
        "\uac00 K-2025/v0 \ud6c4",
        BoundingBox(0, 0, 130, 20),
        0.99,
        (
            OcrEojeol("\uac00", BoundingBox(0, 0, 20, 20), 0.999, 0, 1),
            OcrEojeol("\ud6c4", BoundingBox(100, 0, 120, 20), 0.999, 12, 13),
        ),
    )
    overlapping = OcrLine(
        "0 \ud6c4\ubcf4\ub77c \ub098\ub2e4",
        BoundingBox(90, 0, 210, 20),
        0.99,
        (
            OcrEojeol("\ud6c4\ubcf4\ub77c", BoundingBox(100.5, 0, 160, 20), 0.999, 2, 5),
            OcrEojeol("\ub098\ub2e4", BoundingBox(170, 0, 210, 20), 0.999, 6, 8),
        ),
    )

    merged = _merge_line_group([complete, overlapping])

    assert merged.text == "\uac00 K-2025/v0 \ud6c4\ubcf4\ub77c \ub098\ub2e4"
    assert [item.text for item in merged.eojeols] == [
        "\uac00",
        "\ud6c4\ubcf4\ub77c",
        "\ub098\ub2e4",
    ]
    assert [(item.sentence_start, item.sentence_end) for item in merged.eojeols] == [
        (0, 1),
        (12, 15),
        (16, 18),
    ]


@pytest.mark.parametrize(
    ("structured", "fragment_box", "fragment_confidence"),
    [
        ("K-2025/v1", BoundingBox(100, 0, 120, 20), 0.999),
        ("K-2025/v0", BoundingBox(98.9, 0, 118.9, 20), 0.999),
        ("K-2025/v0", BoundingBox(100, 0, 120, 20), 0.989),
    ],
)
def test_structured_overlap_artifacts_require_exact_suffix_and_geometry(
    structured: str,
    fragment_box: BoundingBox,
    fragment_confidence: float,
) -> None:
    complete_text = f"\uac00 {structured} \ud6c4"
    complete = OcrLine(
        complete_text,
        BoundingBox(0, 0, 130, 20),
        0.99,
        (
            OcrEojeol("\uac00", BoundingBox(0, 0, 20, 20), 0.999, 0, 1),
            OcrEojeol(
                "\ud6c4",
                fragment_box,
                fragment_confidence,
                len(complete_text) - 1,
                len(complete_text),
            ),
        ),
    )
    overlapping = OcrLine(
        "0 \ud6c4\ubcf4\ub77c \ub098\ub2e4",
        BoundingBox(90, 0, 210, 20),
        0.99,
        (
            OcrEojeol("\ud6c4\ubcf4\ub77c", BoundingBox(100.5, 0, 160, 20), 0.999, 2, 5),
            OcrEojeol("\ub098\ub2e4", BoundingBox(170, 0, 210, 20), 0.999, 6, 8),
        ),
    )

    merged = _merge_line_group([complete, overlapping])

    assert merged.text == f"{complete_text} 0 \ud6c4\ubcf4\ub77c \ub098\ub2e4"


def test_tiny_contained_fragment_is_removed_and_spans_are_repaired() -> None:
    line = OcrLine(
        "\ud559\uad50 \uc544 \ud559\uad50\uc5d0\uc11c",
        BoundingBox(0, 0, 120, 20),
        0.9,
        (
            OcrEojeol("\ud559\uad50", BoundingBox(0, 0, 30, 20), 0.9, 0, 2),
            OcrEojeol("\uc544", BoundingBox(50, 0, 55, 20), 0.8, 3, 4),
            OcrEojeol("\ud559\uad50\uc5d0\uc11c", BoundingBox(20, 0, 100, 20), 0.9, 5, 9),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == "\ud559\uad50 \ud559\uad50\uc5d0\uc11c"
    assert [(item.sentence_start, item.sentence_end) for item in cleaned.eojeols] == [
        (0, 2),
        (3, 7),
    ]


def test_matching_character_fragment_with_normal_pitch_is_removed() -> None:
    line = OcrLine(
        '학교 학 학교에서',
        BoundingBox(0, 0, 120, 20),
        0.9,
        (
            OcrEojeol('학교', BoundingBox(0, 0, 30, 20), 0.9, 0, 2),
            OcrEojeol('학', BoundingBox(35, 0, 55, 20), 0.99, 3, 4),
            OcrEojeol('학교에서', BoundingBox(35, 0, 115, 20), 0.99, 5, 9),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == '학교 학교에서'
    assert [item.text for item in cleaned.eojeols] == ['학교', '학교에서']
    assert [(item.sentence_start, item.sentence_end) for item in cleaned.eojeols] == [
        (0, 2),
        (3, 7),
    ]


def test_contained_suffix_fragment_of_short_word_is_removed() -> None:
    line = OcrLine(
        '기원 원 수세기',
        BoundingBox(0, 0, 120, 20),
        0.9,
        (
            OcrEojeol('기원', BoundingBox(0, 0, 40, 20), 0.99, 0, 2),
            OcrEojeol('원', BoundingBox(30, 0, 41, 20), 0.99, 3, 4),
            OcrEojeol('수세기', BoundingBox(50, 0, 110, 20), 0.99, 5, 8),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == '기원 수세기'
    assert [item.text for item in cleaned.eojeols] == ['기원', '수세기']
    assert [(item.sentence_start, item.sentence_end) for item in cleaned.eojeols] == [
        (0, 2),
        (3, 6),
    ]


def test_unrelated_contained_character_of_short_word_is_preserved() -> None:
    line = OcrLine(
        '기원 가 수세기',
        BoundingBox(0, 0, 120, 20),
        0.9,
        (
            OcrEojeol('기원', BoundingBox(0, 0, 40, 20), 0.99, 0, 2),
            OcrEojeol('가', BoundingBox(30, 0, 41, 20), 0.99, 3, 4),
            OcrEojeol('수세기', BoundingBox(50, 0, 110, 20), 0.99, 5, 8),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


def test_contained_fragment_at_width_threshold_is_preserved() -> None:
    line = OcrLine(
        "\uc544 \ud559\uad50\uc5d0\uc11c",
        BoundingBox(0, 0, 100, 20),
        0.9,
        (
            OcrEojeol("\uc544", BoundingBox(20, 0, 29, 20), 0.8, 0, 1),
            OcrEojeol("\ud559\uad50\uc5d0\uc11c", BoundingBox(0, 0, 80, 20), 0.9, 2, 6),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


def test_low_confidence_contained_fragment_is_removed() -> None:
    line = OcrLine(
        "\ub77c \uac00\ub098\ub2e4",
        BoundingBox(0, 0, 150, 20),
        0.9,
        (
            OcrEojeol("\ub77c", BoundingBox(30, 0, 52, 20), 0.53, 0, 1),
            OcrEojeol("\uac00\ub098\ub2e4", BoundingBox(0, 0, 150, 20), 0.999, 2, 5),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == "\uac00\ub098\ub2e4"
    assert [item.text for item in cleaned.eojeols] == ["\uac00\ub098\ub2e4"]
    assert cleaned.eojeols[0].sentence_start == 0
    assert cleaned.eojeols[0].sentence_end == 3


def test_high_confidence_contained_fragment_is_preserved() -> None:
    line = OcrLine(
        "\ub77c \uac00\ub098\ub2e4",
        BoundingBox(0, 0, 150, 20),
        0.9,
        (
            OcrEojeol("\ub77c", BoundingBox(30, 0, 52, 20), 0.6, 0, 1),
            OcrEojeol("\uac00\ub098\ub2e4", BoundingBox(0, 0, 150, 20), 0.999, 2, 5),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


@pytest.mark.parametrize(
    ("fragment_box", "word_text", "word_box", "word_confidence"),
    [
        (BoundingBox(30, 0, 55, 20), "\uac00\ub098\ub2e4", BoundingBox(0, 0, 150, 20), 0.999),
        (BoundingBox(-1, 0, 21, 20), "\uac00\ub098\ub2e4", BoundingBox(0, 0, 150, 20), 0.999),
        (BoundingBox(30, 0, 52, 20), "\uac00\ub098", BoundingBox(0, 0, 150, 20), 0.999),
        (BoundingBox(30, 0, 52, 20), "\uac00\ub098\ub2e4", BoundingBox(0, 0, 150, 20), 0.989),
    ],
)
def test_low_confidence_contained_fragment_requires_all_bounds(
    fragment_box: BoundingBox,
    word_text: str,
    word_box: BoundingBox,
    word_confidence: float,
) -> None:
    line = OcrLine(
        f"\ub77c {word_text}",
        BoundingBox(-1, 0, 150, 20),
        0.9,
        (
            OcrEojeol("\ub77c", fragment_box, 0.53, 0, 1),
            OcrEojeol(word_text, word_box, word_confidence, 2, 2 + len(word_text)),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


def test_low_confidence_leading_fragment_of_short_word_is_removed() -> None:
    line = OcrLine(
        "\ub77c \uac00\ub098",
        BoundingBox(9.5, 0, 110, 20),
        0.9,
        (
            OcrEojeol("\ub77c", BoundingBox(9.5, 0, 33.5, 20), 0.49, 0, 1),
            OcrEojeol("\uac00\ub098", BoundingBox(10, 0, 110, 20), 0.999, 2, 4),
        ),
    )

    cleaned = _remove_tiny_contained_fragments(line)

    assert cleaned.text == "\uac00\ub098"
    assert [item.text for item in cleaned.eojeols] == ["\uac00\ub098"]
    assert cleaned.eojeols[0].sentence_start == 0
    assert cleaned.eojeols[0].sentence_end == 2


@pytest.mark.parametrize(
    ("fragment_box", "fragment_confidence", "word_confidence"),
    [
        (BoundingBox(9.5, 0, 35, 20), 0.49, 0.999),
        (BoundingBox(8.9, 0, 32.9, 20), 0.49, 0.999),
        (BoundingBox(9.5, 0, 33.5, 20), 0.5, 0.999),
        (BoundingBox(9.5, 0, 33.5, 20), 0.49, 0.9989),
    ],
)
def test_low_confidence_leading_fragment_requires_all_bounds(
    fragment_box: BoundingBox,
    fragment_confidence: float,
    word_confidence: float,
) -> None:
    line = OcrLine(
        "\ub77c \uac00\ub098",
        BoundingBox(8.9, 0, 110, 20),
        0.9,
        (
            OcrEojeol("\ub77c", fragment_box, fragment_confidence, 0, 1),
            OcrEojeol("\uac00\ub098", BoundingBox(10, 0, 110, 20), word_confidence, 2, 4),
        ),
    )

    assert _remove_tiny_contained_fragments(line) == line


class DuplicateCharacterRecognizer:
    def __init__(self, text: str = '학교에서') -> None:
        self.text = text

    def recognize(self, _image):
        return RecognizedText(self.text, 0.9997)


def test_confirmed_overlapping_character_duplicate_is_discarded() -> None:
    words = [
        ('학', BoundingBox(30, 0, 37, 20), 0.776),
        ('학교에서', BoundingBox(36, 0, 91, 20), 0.986),
    ]

    recovered = _discard_confirmed_overlapping_character_duplicates(
        words,
        Image.new('RGB', (100, 20)),
        BoundingBox(0, 0, 100, 20),
        DuplicateCharacterRecognizer(),
    )

    assert recovered == [
        ('학교에서', BoundingBox(30, 0, 91, 20), 0.986),
    ]


def test_overlapping_character_is_preserved_without_exact_confirmation() -> None:
    words = [
        ('학', BoundingBox(30, 0, 37, 20), 0.776),
        ('학교에서', BoundingBox(36, 0, 91, 20), 0.986),
    ]

    assert (
        _discard_confirmed_overlapping_character_duplicates(
            words,
            Image.new('RGB', (100, 20)),
            BoundingBox(0, 0, 100, 20),
            DuplicateCharacterRecognizer('학생학교에서'),
        )
        == words
    )


def test_high_confidence_overlapping_character_is_preserved_despite_exact_confirmation() -> None:
    words = [
        ('학', BoundingBox(30, 0, 36, 20), 0.982),
        ('학교에서', BoundingBox(35, 0, 93, 20), 0.999),
    ]

    assert (
        _discard_confirmed_overlapping_character_duplicates(
            words,
            Image.new('RGB', (100, 20)),
            BoundingBox(0, 0, 100, 20),
            DuplicateCharacterRecognizer(),
        )
        == words
    )


def test_exactly_confirmed_touching_character_duplicate_is_discarded() -> None:
    words = [
        ('학', BoundingBox(30, 0, 39, 20), 0.33),
        ('학교에서', BoundingBox(39, 0, 99, 20), 0.998),
    ]

    recovered = _discard_confirmed_overlapping_character_duplicates(
        words,
        Image.new('RGB', (110, 20)),
        BoundingBox(0, 0, 110, 20),
        DuplicateCharacterRecognizer(),
    )

    assert recovered == [
        ('학교에서', BoundingBox(30, 0, 99, 20), 0.998),
    ]


def test_one_pixel_duplicate_overlap_is_allowed_on_small_line() -> None:
    words = [
        ('학', BoundingBox(30, 0, 37, 14), 0.55),
        ('학교에서', BoundingBox(36, 0, 92, 14), 0.991),
    ]

    recovered = _discard_confirmed_overlapping_character_duplicates(
        words,
        Image.new('RGB', (100, 14)),
        BoundingBox(0, 0, 100, 14),
        DuplicateCharacterRecognizer(),
    )

    assert recovered == [
        ('학교에서', BoundingBox(30, 0, 92, 14), 0.991),
    ]


def test_exactly_confirmed_overlapping_digit_artifact_is_discarded() -> None:
    words = [
        ('9', BoundingBox(30, 0, 40, 30), 0.915),
        ('약소국', BoundingBox(39, 0, 93, 30), 0.995),
    ]

    recovered = _discard_confirmed_overlapping_character_duplicates(
        words,
        Image.new('RGB', (100, 30)),
        BoundingBox(0, 0, 100, 30),
        DuplicateCharacterRecognizer('약소국'),
    )

    assert recovered == [
        ('약소국', BoundingBox(30, 0, 93, 30), 0.995),
    ]


def test_overlapping_digit_is_preserved_without_exact_confirmation() -> None:
    words = [
        ('9', BoundingBox(30, 0, 40, 30), 0.915),
        ('약소국', BoundingBox(39, 0, 93, 30), 0.995),
    ]

    assert (
        _discard_confirmed_overlapping_character_duplicates(
            words,
            Image.new('RGB', (100, 30)),
            BoundingBox(0, 0, 100, 30),
            DuplicateCharacterRecognizer('9약소국'),
        )
        == words
    )


def test_normal_width_character_duplicate_requires_exact_confirmation() -> None:
    words = [
        ('학', BoundingBox(30, 0, 43, 20), 0.33),
        ('학교에서', BoundingBox(42, 0, 102, 20), 0.998),
    ]

    assert (
        _discard_confirmed_overlapping_character_duplicates(
            words,
            Image.new('RGB', (110, 20)),
            BoundingBox(0, 0, 110, 20),
            DuplicateCharacterRecognizer('학생학교에서'),
        )
        == words
    )


class TripletRecognizer:
    def recognize(self, _image):
        return RecognizedText("\ud559\uad50\uc5d0\uc11c\ub294", 0.999)


def test_overlapping_word_triplet_discards_only_confirmed_leading_sliver() -> None:
    words = [
        ("\uac00", BoundingBox(10, 0, 16, 20), 0.78),
        ("\ud559\uad50\uc5d0\uc11c", BoundingBox(15, 0, 75, 20), 0.999),
        ("\ub294", BoundingBox(74, 0, 94, 20), 0.93),
    ]

    recovered = _recover_overlapping_word_triplets(
        words,
        Image.new("RGB", (100, 20)),
        BoundingBox(0, 0, 100, 20),
        TripletRecognizer(),
    )

    assert recovered == [
        ("\ud559\uad50\uc5d0\uc11c\ub294", BoundingBox(10, 0, 94, 20), 0.93),
    ]


def test_triplet_recovery_precedes_pair_duplicate_cleanup() -> None:
    words = [
        ('가', BoundingBox(10, 0, 16, 20), 0.78),
        ('학교에서', BoundingBox(15, 0, 75, 20), 0.999),
        ('는', BoundingBox(74, 0, 94, 20), 0.93),
    ]

    recovered = _recover_word_boundaries(
        words,
        Image.new('RGB', (100, 20)),
        BoundingBox(0, 0, 100, 20),
        TripletRecognizer(),
    )

    assert recovered == [
        ('학교에서는', BoundingBox(10, 0, 94, 20), 0.93),
    ]


class OverlappingSuffixRecognizer:
    def __init__(self, text: str = '의도와는') -> None:
        self.text = text

    def recognize(self, _image):
        return RecognizedText(self.text, 0.9994)


def test_overlapping_suffix_pair_merges_with_exact_combined_recognition() -> None:
    words = [
        ('의도', BoundingBox(10, 0, 50, 30), 0.603),
        ('도와는', BoundingBox(49, 0, 100, 30), 0.978),
    ]

    recovered = _recover_overlapping_suffix_pairs(
        words,
        Image.new('RGB', (110, 30)),
        BoundingBox(0, 0, 110, 30),
        OverlappingSuffixRecognizer(),
    )

    assert recovered == [
        ('의도와는', BoundingBox(10, 0, 100, 30), 0.603),
    ]


def test_overlapping_suffix_pair_requires_exact_combined_recognition() -> None:
    words = [
        ('의도', BoundingBox(10, 0, 50, 30), 0.603),
        ('도와는', BoundingBox(49, 0, 100, 30), 0.978),
    ]

    assert (
        _recover_overlapping_suffix_pairs(
            words,
            Image.new('RGB', (110, 30)),
            BoundingBox(0, 0, 110, 30),
            OverlappingSuffixRecognizer('의도 도와는'),
        )
        == words
    )


def test_high_confidence_repeated_boundary_is_preserved() -> None:
    words = [
        ('학교', BoundingBox(10, 0, 50, 30), 0.999),
        ('교실', BoundingBox(49, 0, 90, 30), 0.998),
    ]

    assert (
        _recover_overlapping_suffix_pairs(
            words,
            Image.new('RGB', (100, 30)),
            BoundingBox(0, 0, 100, 30),
            OverlappingSuffixRecognizer('학교실'),
        )
        == words
    )


def test_overlapping_word_triplet_preserves_wider_leading_word() -> None:
    words = [
        ("\uac00", BoundingBox(5, 0, 15, 20), 0.78),
        ("\ud559\uad50\uc5d0\uc11c", BoundingBox(14, 0, 74, 20), 0.999),
        ("\ub294", BoundingBox(73, 0, 93, 20), 0.93),
    ]

    assert (
        _recover_overlapping_word_triplets(
            words,
            Image.new("RGB", (100, 20)),
            BoundingBox(0, 0, 100, 20),
            TripletRecognizer(),
        )
        == words
    )


class ClosePairRecognizer:
    def recognize(self, _image):
        return RecognizedText("\ud559\uad50\uc5d0\uc11c\ub294", 0.9985)


class FinalSyllablePairRecognizer:
    def recognize(self, _image):
        return RecognizedText("\uc0ac\uc6a9\ud558\ub294", 0.9998)


class ShortPairRecognizer:
    def __init__(self, confidence: float = 0.99995) -> None:
        self.confidence = confidence

    def recognize(self, _image):
        return RecognizedText("\ucd08\uae30", self.confidence)


def test_isolated_close_pair_merges_only_with_matching_pitch_and_wide_neighbors() -> None:
    words = [
        ("\uc624\ub298", BoundingBox(0, 0, 50, 30), 0.999),
        ("\ud559\uad50", BoundingBox(65, 0, 113, 30), 0.9998),
        ("\uc5d0\uc11c\ub294", BoundingBox(119, 0, 191, 30), 0.9994),
        ("\uac11\ub2c8\ub2e4", BoundingBox(206, 0, 278, 30), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (300, 30)),
        BoundingBox(0, 0, 300, 30),
        ClosePairRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\ud559\uad50\uc5d0\uc11c\ub294", BoundingBox(65, 0, 191, 30), 0.9985),
        words[3],
    ]


def test_isolated_close_pair_preserves_pair_without_wide_following_gap() -> None:
    words = [
        ("\uc624\ub298", BoundingBox(0, 0, 50, 30), 0.999),
        ("\ud559\uad50", BoundingBox(65, 0, 113, 30), 0.9998),
        ("\uc5d0\uc11c\ub294", BoundingBox(119, 0, 191, 30), 0.9994),
        ("\uac11\ub2c8\ub2e4", BoundingBox(200, 0, 272, 30), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (300, 30)),
            BoundingBox(0, 0, 300, 30),
            ClosePairRecognizer(),
        )
        == words
    )


def test_isolated_close_pair_recovers_high_confidence_final_syllable() -> None:
    words = [
        ("\uc774\uac83", BoundingBox(0, 0, 32, 20), 0.9999),
        ("\uc0ac\uc6a9\ud558", BoundingBox(39, 0, 87, 20), 0.9999),
        ("\ub294", BoundingBox(90.4, 0, 104.4, 20), 0.9998),
        ("\ubc29\ubc95", BoundingBox(112.4, 0, 144.4, 20), 0.9999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (160, 20)),
        BoundingBox(0, 0, 160, 20),
        FinalSyllablePairRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uc0ac\uc6a9\ud558\ub294", BoundingBox(39, 0, 104.4, 20), 0.9998),
        words[3],
    ]


def test_isolated_close_pair_preserves_final_syllable_near_following_word() -> None:
    words = [
        ("\uc774\uac83", BoundingBox(0, 0, 32, 20), 0.9999),
        ("\uc0ac\uc6a9\ud558", BoundingBox(39, 0, 87, 20), 0.9999),
        ("\ub294", BoundingBox(90.4, 0, 104.4, 20), 0.9998),
        ("\ubc29\ubc95", BoundingBox(110.4, 0, 142.4, 20), 0.9999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (160, 20)),
            BoundingBox(0, 0, 160, 20),
            FinalSyllablePairRecognizer(),
        )
        == words
    )


def test_isolated_close_pair_preserves_lower_confidence_final_syllable() -> None:
    words = [
        ("\uc774\uac83", BoundingBox(0, 0, 32, 20), 0.9999),
        ("\uc0ac\uc6a9\ud558", BoundingBox(39, 0, 87, 20), 0.9999),
        ("\ub294", BoundingBox(90.4, 0, 104.4, 20), 0.9993),
        ("\ubc29\ubc95", BoundingBox(112.4, 0, 144.4, 20), 0.9999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (160, 20)),
            BoundingBox(0, 0, 160, 20),
            FinalSyllablePairRecognizer(),
        )
        == words
    )


def test_isolated_touching_syllables_merge_with_wide_neighbors() -> None:
    words = [
        ("\uc774\ubbf8", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ucd08", BoundingBox(50, 0, 69, 20), 0.99),
        ("\uae30", BoundingBox(69, 0, 81, 20), 0.997),
        ("\uc790\ubcf8\uc8fc\uc758", BoundingBox(90, 0, 170, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (180, 20)),
        BoundingBox(0, 0, 180, 20),
        ShortPairRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\ucd08\uae30", BoundingBox(50, 0, 81, 20), 0.99),
        words[3],
    ]


def test_isolated_touching_syllables_require_wide_following_gap() -> None:
    words = [
        ("\uc774\ubbf8", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ucd08", BoundingBox(50, 0, 69, 20), 0.99),
        ("\uae30", BoundingBox(69, 0, 81, 20), 0.997),
        ("\uc790\ubcf8\uc8fc\uc758", BoundingBox(89.5, 0, 169.5, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (180, 20)),
            BoundingBox(0, 0, 180, 20),
            ShortPairRecognizer(),
        )
        == words
    )


def test_isolated_touching_syllables_require_near_certain_combined_text() -> None:
    words = [
        ("\uc774\ubbf8", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ucd08", BoundingBox(50, 0, 69, 20), 0.99),
        ("\uae30", BoundingBox(69, 0, 81, 20), 0.997),
        ("\uc790\ubcf8\uc8fc\uc758", BoundingBox(90, 0, 170, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (180, 20)),
            BoundingBox(0, 0, 180, 20),
            ShortPairRecognizer(0.9998),
        )
        == words
    )


class IsolatedOnePlusTwoRecognizer:
    def __init__(self, confidence: float = 0.99991) -> None:
        self.confidence = confidence

    def recognize(self, _image):
        return RecognizedText("\uac1c\uc778\uc744", self.confidence)


def test_isolated_one_plus_two_pair_merges_with_exact_recognition() -> None:
    words = [
        ("\uc9d1\ub2e8\ub9cc", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uac1c", BoundingBox(46, 0, 63, 20), 0.9989),
        ("\uc778\uc744", BoundingBox(67.2, 0, 108.2, 20), 0.99985),
        ("\ubd80\uac01", BoundingBox(114.6, 0, 146.6, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (160, 20)),
        BoundingBox(0, 0, 160, 20),
        IsolatedOnePlusTwoRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uac1c\uc778\uc744", BoundingBox(46, 0, 108.2, 20), 0.9989),
        words[3],
    ]


def test_isolated_one_plus_two_pair_requires_wide_following_gap() -> None:
    words = [
        ("\uc9d1\ub2e8\ub9cc", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uac1c", BoundingBox(46, 0, 63, 20), 0.9989),
        ("\uc778\uc744", BoundingBox(67.2, 0, 108.2, 20), 0.99985),
        ("\ubd80\uac01", BoundingBox(114.5, 0, 146.5, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (160, 20)),
            BoundingBox(0, 0, 160, 20),
            IsolatedOnePlusTwoRecognizer(),
        )
        == words
    )


class CompetitiveOnePlusTwoRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        *,
        competing_confidence: float = 0.7,
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.competing_confidence = competing_confidence

    def recognize(self, image):
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, 0.9998)
        return RecognizedText("competing", self.competing_confidence)


def test_line_initial_one_plus_two_pair_merges_with_weak_competing_union() -> None:
    words = [
        ("\uc774", BoundingBox(0, 0, 15, 20), 0.9993),
        ("\ub7ec\ud55c", BoundingBox(22.2, 0, 60.2, 20), 0.9987),
        ("\uc77c\ubcf8\uc778\ub4e4\uc758", BoundingBox(72.4, 0, 140, 20), 0.9993),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (150, 20)),
        BoundingBox(0, 0, 150, 20),
        CompetitiveOnePlusTwoRecognizer(61, "\uc774\ub7ec\ud55c"),
    )

    assert recovered == [
        ("\uc774\ub7ec\ud55c", BoundingBox(0, 0, 60.2, 20), 0.9987),
        words[2],
    ]


def test_line_initial_one_plus_two_pair_rejects_strong_competing_union() -> None:
    words = [
        ("\uc774", BoundingBox(0, 0, 15, 20), 0.9993),
        ("\ub7ec\ud55c", BoundingBox(22.2, 0, 60.2, 20), 0.9987),
        ("\uc77c\ubcf8\uc778\ub4e4\uc758", BoundingBox(72.4, 0, 140, 20), 0.9993),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (150, 20)),
            BoundingBox(0, 0, 150, 20),
            CompetitiveOnePlusTwoRecognizer(
                61,
                "\uc774\ub7ec\ud55c",
                competing_confidence=0.9,
            ),
        )
        == words
    )


def test_touching_following_one_plus_two_pair_merges_with_weak_competing_union() -> None:
    words = [
        ("\ubcf4\ub2c8", BoundingBox(161.64, 0, 184.64, 15.85), 0.9953),
        ("\uac1c", BoundingBox(190.64, 0, 201.64, 15.85), 0.99995),
        ("\uc778\uc744", BoundingBox(202.64, 0, 225.64, 15.85), 0.9994),
        (
            "\ubd80\uac01\uc2dc\ud0a4\uae30\ub97c",
            BoundingBox(225.64, 0, 302.64, 15.85),
            0.999,
        ),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (543, 16)),
        BoundingBox(85.64, 0, 627.36, 15.85),
        CompetitiveOnePlusTwoRecognizer(35, "\uac1c\uc778\uc744"),
    )

    assert recovered == [
        words[0],
        ("\uac1c\uc778\uc744", BoundingBox(190.64, 0, 225.64, 15.85), 0.9994),
        words[3],
    ]


def test_touching_following_one_plus_two_pair_rejects_strong_competing_union() -> None:
    words = [
        ("\ubcf4\ub2c8", BoundingBox(161.64, 0, 184.64, 15.85), 0.9953),
        ("\uac1c", BoundingBox(190.64, 0, 201.64, 15.85), 0.99995),
        ("\uc778\uc744", BoundingBox(202.64, 0, 225.64, 15.85), 0.9994),
        (
            "\ubd80\uac01\uc2dc\ud0a4\uae30\ub97c",
            BoundingBox(225.64, 0, 302.64, 15.85),
            0.999,
        ),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (543, 16)),
            BoundingBox(85.64, 0, 627.36, 15.85),
            CompetitiveOnePlusTwoRecognizer(
                35,
                "\uac1c\uc778\uc744",
                competing_confidence=0.9,
            ),
        )
        == words
    )


def test_low_confidence_overlapping_one_plus_two_pair_is_not_recovered() -> None:
    words = [
        ("\ub9d0\uc740", BoundingBox(0, 0, 26, 18), 0.9999),
        ("\uc2dc", BoundingBox(25, 0, 42, 18), 0.923),
        ("\uc6d0\uc740", BoundingBox(45, 0, 71, 18), 0.9997),
        ("\ud558\uc9c0\ub9cc", BoundingBox(70, 0, 117, 18), 0.9866),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (120, 18)),
            BoundingBox(0, 0, 120, 18),
            CompetitiveOnePlusTwoRecognizer(1, "unused"),
        )
        == words
    )


class CompetitiveThreePlusThreeRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        *,
        strong_competitor_width: int | None = None,
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.strong_competitor_width = strong_competitor_width

    def recognize(self, image):
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, 0.9994)
        confidence = 0.995 if image.width == self.strong_competitor_width else 0.98
        return RecognizedText("competing", confidence)


def test_line_initial_three_plus_three_pair_merges_with_weak_competitor() -> None:
    words = [
        ("\ubc1c\uc804\uc2dc", BoundingBox(0, 0, 60, 20), 0.9967),
        ("\ud0a8\ub2e4\ub294", BoundingBox(65.2, 0, 125.2, 20), 0.99995),
        ("\uac83\uc774", BoundingBox(136.1, 0, 176.1, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (190, 20)),
        BoundingBox(0, 0, 190, 20),
        CompetitiveThreePlusThreeRecognizer(126, "\ubc1c\uc804\uc2dc\ud0a8\ub2e4\ub294"),
    )

    assert recovered == [
        (
            "\ubc1c\uc804\uc2dc\ud0a8\ub2e4\ub294",
            BoundingBox(0, 0, 125.2, 20),
            0.9967,
        ),
        words[2],
    ]


def test_isolated_three_plus_three_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\ub2f9\uc2dc", BoundingBox(0, 0, 60, 20), 0.999),
        ("\uc720\ub7fd\uc2dc", BoundingBox(72.2, 0, 132.2, 20), 0.9995),
        ("\ubbfc\ub4e4\uc740", BoundingBox(139.2, 0, 199.2, 20), 0.9969),
        ("\ud604\uc2e4\uc5d0", BoundingBox(208, 0, 268, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (280, 20)),
        BoundingBox(0, 0, 280, 20),
        CompetitiveThreePlusThreeRecognizer(128, "\uc720\ub7fd\uc2dc\ubbfc\ub4e4\uc740"),
    )

    assert recovered == [
        words[0],
        (
            "\uc720\ub7fd\uc2dc\ubbfc\ub4e4\uc740",
            BoundingBox(72.2, 0, 199.2, 20),
            0.9969,
        ),
        words[3],
    ]


def test_isolated_three_plus_three_pair_rejects_strong_competitor() -> None:
    words = [
        ("\ub2f9\uc2dc", BoundingBox(0, 0, 60, 20), 0.999),
        ("\uc720\ub7fd\uc2dc", BoundingBox(72.2, 0, 132.2, 20), 0.9995),
        ("\ubbfc\ub4e4\uc740", BoundingBox(139.2, 0, 199.2, 20), 0.9969),
        ("\ud604\uc2e4\uc5d0", BoundingBox(208, 0, 268, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (280, 20)),
            BoundingBox(0, 0, 280, 20),
            CompetitiveThreePlusThreeRecognizer(
                128,
                "\uc720\ub7fd\uc2dc\ubbfc\ub4e4\uc740",
                strong_competitor_width=129,
            ),
        )
        == words
    )


class CompetitiveThreePlusTwoRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        *,
        strong_competitor_width: int | None = None,
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.strong_competitor_width = strong_competitor_width
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, 0.9994)
        confidence = 0.995 if image.width == self.strong_competitor_width else 0.98
        return RecognizedText("competing", confidence)


def test_narrow_gap_three_plus_two_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uace0\ud310\ubcf8", BoundingBox(44, 0, 104, 20), 0.9988),
        ("\uc5d0\uc11c", BoundingBox(105, 0, 145, 20), 0.9996),
        ("\ub2e4\uc74c", BoundingBox(150, 0, 190, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (200, 20)),
        BoundingBox(0, 0, 200, 20),
        CompetitiveThreePlusTwoRecognizer(101, "\uace0\ud310\ubcf8\uc5d0\uc11c"),
    )

    assert recovered == [
        words[0],
        ("\uace0\ud310\ubcf8\uc5d0\uc11c", BoundingBox(44, 0, 145, 20), 0.9988),
        words[3],
    ]


def test_isolated_wide_three_plus_two_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ub514\uc790\uc774", BoundingBox(52.2, 0, 112.2, 20), 0.9982),
        ("\ub108\uc758", BoundingBox(119.42, 0, 159.42, 20), 0.9995),
        ("\ub2e4\uc74c", BoundingBox(171.64, 0, 211.64, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (220, 20)),
        BoundingBox(0, 0, 220, 20),
        CompetitiveThreePlusTwoRecognizer(108, "\ub514\uc790\uc774\ub108\uc758"),
    )

    assert recovered == [
        words[0],
        ("\ub514\uc790\uc774\ub108\uc758", BoundingBox(52.2, 0, 159.42, 20), 0.9982),
        words[3],
    ]


def test_narrow_gap_three_plus_two_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uace0\ud310\ubcf8", BoundingBox(44, 0, 104, 20), 0.9988),
        ("\uc5d0\uc11c", BoundingBox(105, 0, 145, 20), 0.9996),
        ("\ub2e4\uc74c", BoundingBox(150, 0, 190, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (200, 20)),
            BoundingBox(0, 0, 200, 20),
            CompetitiveThreePlusTwoRecognizer(
                101,
                "\uace0\ud310\ubcf8\uc5d0\uc11c",
                strong_competitor_width=104,
            ),
        )
        == words
    )


def test_three_plus_two_spacing_control_is_not_recovered() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ub9cc\ub4e4\uc5b4", BoundingBox(50.2, 0, 110.2, 20), 0.9838),
        ("\ub0b4\ub294", BoundingBox(115.88, 0, 155.88, 20), 0.9999),
        ("\ub2e4\uc74c", BoundingBox(166.1, 0, 206.1, 20), 0.999),
    ]
    recognizer = CompetitiveThreePlusTwoRecognizer(106, "unused")

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (215, 20)),
            BoundingBox(0, 0, 215, 20),
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class CompetitiveFourPlusTwoRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        candidate_confidence: float,
        *,
        strong_competitor_width: int | None = None,
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.candidate_confidence = candidate_confidence
        self.strong_competitor_width = strong_competitor_width

    def recognize(self, image):
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, self.candidate_confidence)
        confidence = 0.99 if image.width == self.strong_competitor_width else 0.98
        return RecognizedText("competing", confidence)


def test_positive_gap_four_plus_two_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uae40\ub300\uc911\uc528", BoundingBox(50.2, 0, 130.2, 20), 0.9988),
        ("\uc870\ucc28", BoundingBox(134.74, 0, 174.74, 20), 0.9998),
        ("\ub2e4\uc74c", BoundingBox(183.74, 0, 223.74, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (230, 20)),
        BoundingBox(0, 0, 230, 20),
        CompetitiveFourPlusTwoRecognizer(
            125,
            "\uae40\ub300\uc911\uc528\uc870\ucc28",
            0.9971,
        ),
    )

    assert recovered == [
        words[0],
        (
            "\uae40\ub300\uc911\uc528\uc870\ucc28",
            BoundingBox(50.2, 0, 174.74, 20),
            0.9971,
        ),
        words[3],
    ]


def test_overlapping_four_plus_two_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uc911\uc18c\uae30\uc5c5", BoundingBox(47.2, 0, 127.2, 20), 0.99895),
        ("\uc5d0\uc11c", BoundingBox(126.16, 0, 166.16, 20), 0.9607),
        ("\ub2e4\uc74c", BoundingBox(174.36, 0, 214.36, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (220, 20)),
        BoundingBox(0, 0, 220, 20),
        CompetitiveFourPlusTwoRecognizer(
            120,
            "\uc911\uc18c\uae30\uc5c5\uc5d0\uc11c",
            0.9994,
        ),
    )

    assert recovered == [
        words[0],
        (
            "\uc911\uc18c\uae30\uc5c5\uc5d0\uc11c",
            BoundingBox(47.2, 0, 166.16, 20),
            0.9607,
        ),
        words[3],
    ]


def test_positive_gap_four_plus_two_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uae40\ub300\uc911\uc528", BoundingBox(50.2, 0, 130.2, 20), 0.9988),
        ("\uc870\ucc28", BoundingBox(134.74, 0, 174.74, 20), 0.9998),
        ("\ub2e4\uc74c", BoundingBox(183.74, 0, 223.74, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (230, 20)),
            BoundingBox(0, 0, 230, 20),
            CompetitiveFourPlusTwoRecognizer(
                125,
                "\uae40\ub300\uc911\uc528\uc870\ucc28",
                0.9971,
                strong_competitor_width=131,
            ),
        )
        == words
    )


def test_overlapping_four_plus_two_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uc911\uc18c\uae30\uc5c5", BoundingBox(47.2, 0, 127.2, 20), 0.99895),
        ("\uc5d0\uc11c", BoundingBox(126.16, 0, 166.16, 20), 0.9607),
        ("\ub2e4\uc74c", BoundingBox(174.36, 0, 214.36, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (220, 20)),
            BoundingBox(0, 0, 220, 20),
            CompetitiveFourPlusTwoRecognizer(
                120,
                "\uc911\uc18c\uae30\uc5c5\uc5d0\uc11c",
                0.9994,
                strong_competitor_width=89,
            ),
        )
        == words
    )


class NarrowThreePlusTwoRecognizer:
    def recognize(self, _image):
        return RecognizedText("이론에서는", 0.99985)


def test_narrow_three_plus_two_pair_merges_with_exact_recognition() -> None:
    words = [
        ("상대성", BoundingBox(0, 0, 47, 20), 0.9995),
        ("이론에", BoundingBox(52, 0, 89, 20), 0.99975),
        ("서는", BoundingBox(91, 0, 116, 20), 0.99985),
        ("사건이라는", BoundingBox(115, 0, 184, 20), 0.9942),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (190, 20)),
        BoundingBox(0, 0, 190, 20),
        NarrowThreePlusTwoRecognizer(),
    )

    assert recovered == [
        words[0],
        ("이론에서는", BoundingBox(52, 0, 116, 20), 0.99975),
        words[3],
    ]


def test_narrow_three_plus_two_pair_requires_following_overlap() -> None:
    words = [
        ("상대성", BoundingBox(0, 0, 47, 20), 0.9995),
        ("이론에", BoundingBox(52, 0, 89, 20), 0.99975),
        ("서는", BoundingBox(91, 0, 116, 20), 0.99985),
        ("사건이라는", BoundingBox(116, 0, 185, 20), 0.9942),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (190, 20)),
            BoundingBox(0, 0, 190, 20),
            NarrowThreePlusTwoRecognizer(),
        )
        == words
    )


class TerminalOverlapRecognizer:
    def recognize(self, _image):
        return RecognizedText("\uc9c0\uc6d0\ud55c\ub2e4", 0.99995)


def test_terminal_overlapping_pair_merges_with_exact_combined_recognition() -> None:
    words = [
        ("\uc774\ub97c", BoundingBox(0, 0, 60, 30), 0.9998),
        ("\uc9c0\uc6d0", BoundingBox(76, 0, 136, 30), 0.9997),
        ("\ud55c\ub2e4", BoundingBox(135, 0, 191, 30), 0.99995),
    ]

    recovered = _recover_terminal_overlapping_word_pair(
        words,
        Image.new("RGB", (200, 30)),
        BoundingBox(0, 0, 200, 30),
        TerminalOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uc9c0\uc6d0\ud55c\ub2e4", BoundingBox(76, 0, 191, 30), 0.9997),
    ]


def test_terminal_overlapping_pair_preserves_internal_pair() -> None:
    words = [
        ("\uc774\ub97c", BoundingBox(0, 0, 60, 30), 0.9998),
        ("\uc9c0\uc6d0", BoundingBox(76, 0, 136, 30), 0.9997),
        ("\ud55c\ub2e4", BoundingBox(135, 0, 191, 30), 0.99995),
        ("\uc624\ub298", BoundingBox(210, 0, 270, 30), 0.9999),
    ]

    assert (
        _recover_terminal_overlapping_word_pair(
            words,
            Image.new("RGB", (280, 30)),
            BoundingBox(0, 0, 280, 30),
            TerminalOverlapRecognizer(),
        )
        == words
    )


class IsolatedOverlapRecognizer:
    def recognize(self, _image):
        return RecognizedText("\uc9c0\uc6d0\ud558\ub294\ub2e4", 0.9958)


def test_isolated_overlapping_pair_merges_exact_two_plus_three_surface() -> None:
    words = [
        ("\uc774\ub97c", BoundingBox(0, 0, 30, 20), 0.999),
        ("\uc9c0\uc6d0", BoundingBox(39, 0, 68, 20), 0.9989),
        ("\ud558\ub294\ub2e4", BoundingBox(67, 0, 106, 20), 0.9994),
        ("\uc624\ub298", BoundingBox(114, 0, 144, 20), 0.999),
    ]

    recovered = _recover_isolated_overlapping_word_pairs(
        words,
        Image.new("RGB", (150, 20)),
        BoundingBox(0, 0, 150, 20),
        IsolatedOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uc9c0\uc6d0\ud558\ub294\ub2e4", BoundingBox(39, 0, 106, 20), 0.9958),
        words[3],
    ]


def test_isolated_overlapping_pair_preserves_pair_near_following_word() -> None:
    words = [
        ("\uc774\ub97c", BoundingBox(0, 0, 30, 20), 0.999),
        ("\uc9c0\uc6d0", BoundingBox(39, 0, 68, 20), 0.9989),
        ("\ud558\ub294\ub2e4", BoundingBox(67, 0, 106, 20), 0.9994),
        ("\uc624\ub298", BoundingBox(112, 0, 142, 20), 0.999),
    ]

    assert (
        _recover_isolated_overlapping_word_pairs(
            words,
            Image.new("RGB", (150, 20)),
            BoundingBox(0, 0, 150, 20),
            IsolatedOverlapRecognizer(),
        )
        == words
    )


class IsolatedFinalOverlapRecognizer:
    def recognize(self, _image):
        return RecognizedText("\uae30\uc5c5\uc740", 0.9998)


def test_isolated_overlapping_pair_merges_exact_final_syllable() -> None:
    words = [
        ("\ud604\uc0c1\uc740", BoundingBox(0, 0, 44, 20), 0.999),
        ("\uae30\uc5c5", BoundingBox(49, 0, 89, 20), 0.9975),
        ("\uc740", BoundingBox(88.1, 0, 102.1, 20), 0.8),
        ("\uc9d1\uc911", BoundingBox(109, 0, 139, 20), 0.999),
    ]

    recovered = _recover_isolated_overlapping_word_pairs(
        words,
        Image.new("RGB", (150, 20)),
        BoundingBox(0, 0, 150, 20),
        IsolatedFinalOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uae30\uc5c5\uc740", BoundingBox(49, 0, 102.1, 20), 0.8),
        words[3],
    ]


def test_isolated_overlapping_final_syllable_requires_wide_following_gap() -> None:
    words = [
        ("\ud604\uc0c1\uc740", BoundingBox(0, 0, 44, 20), 0.999),
        ("\uae30\uc5c5", BoundingBox(49, 0, 89, 20), 0.9975),
        ("\uc740", BoundingBox(88.1, 0, 102.1, 20), 0.8),
        ("\uc9d1\uc911", BoundingBox(108.8, 0, 138.8, 20), 0.999),
    ]

    assert (
        _recover_isolated_overlapping_word_pairs(
            words,
            Image.new("RGB", (150, 20)),
            BoundingBox(0, 0, 150, 20),
            IsolatedFinalOverlapRecognizer(),
        )
        == words
    )


def test_isolated_overlapping_final_syllable_accepts_pitch_roundoff() -> None:
    words = [
        ("다국적", BoundingBox(121.84, 149.67, 189.84, 176.09), 0.9998),
        ("기업", BoundingBox(200.84, 149.67, 242.84, 176.09), 0.9999),
        ("은", BoundingBox(241.84, 149.67, 271.84, 176.09), 0.9945),
        ("이러한", BoundingBox(280.84, 149.67, 350.84, 176.09), 0.9999),
    ]

    recovered = _recover_isolated_overlapping_word_pairs(
        words,
        Image.new("RGB", (1000, 27)),
        BoundingBox(66.84, 149.67, 939.16, 176.09),
        IsolatedFinalOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ("기업은", BoundingBox(200.84, 149.67, 271.84, 176.09), 0.9945),
        words[3],
    ]


class IsolatedLeadingOverlapRecognizer:
    def __init__(self, confidence: float = 0.9976) -> None:
        self.confidence = confidence

    def recognize(self, _image):
        return RecognizedText("\ube44\ubc00\ud65c\ub3d9\uc744", self.confidence)


def test_isolated_overlapping_pair_merges_exact_leading_syllable() -> None:
    words = [
        ("\ubbfc\uc911\uc6b4\ub3d9\uc740", BoundingBox(0, 0, 77, 20), 0.978),
        ("\ube44", BoundingBox(86, 0, 103, 20), 0.83),
        ("\ubc00\ud65c\ub3d9\uc744", BoundingBox(102, 0, 163, 20), 0.9976),
        ("\uc911\uc2ec\uc73c\ub85c", BoundingBox(171, 0, 233, 20), 0.997),
    ]

    recovered = _recover_isolated_overlapping_word_pairs(
        words,
        Image.new("RGB", (240, 20)),
        BoundingBox(0, 0, 240, 20),
        IsolatedLeadingOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\ube44\ubc00\ud65c\ub3d9\uc744", BoundingBox(86, 0, 163, 20), 0.83),
        words[3],
    ]


def test_isolated_overlapping_leading_syllable_requires_exact_confidence() -> None:
    words = [
        ("\ubbfc\uc911\uc6b4\ub3d9\uc740", BoundingBox(0, 0, 77, 20), 0.978),
        ("\ube44", BoundingBox(86, 0, 103, 20), 0.83),
        ("\ubc00\ud65c\ub3d9\uc744", BoundingBox(102, 0, 163, 20), 0.9976),
        ("\uc911\uc2ec\uc73c\ub85c", BoundingBox(171, 0, 233, 20), 0.997),
    ]

    assert (
        _recover_isolated_overlapping_word_pairs(
            words,
            Image.new("RGB", (240, 20)),
            BoundingBox(0, 0, 240, 20),
            IsolatedLeadingOverlapRecognizer(0.9974),
        )
        == words
    )


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
        return [type("Input", (), {"name": "x"})()]

    def run(self, _outputs, _inputs):
        output = np.zeros((1, 10, 3), dtype=np.float32)
        output[0, :, 0] = 1.0
        for timestep in (1, 4, 8):
            output[0, timestep] = (0.0, 1.0, 0.0)
        for timestep in (3, 6):
            output[0, timestep] = (0.0, 0.0, 1.0)
        return [output]


def test_recognizer_uses_ctc_space_probabilities_for_word_boxes(tmp_path) -> None:
    characters = tmp_path / "characters.txt"
    characters.write_text("\uac00\n", encoding="utf-8")
    recognizer = PaddleRecognizer(tmp_path / "unused.onnx", characters, session=SegmentingSession())

    boxes = recognizer.word_boxes(Image.new("RGB", (100, 20), (255, 0, 0)))

    assert len(boxes) == 3
    assert boxes[0][0] == 0
    assert boxes[-1][1] == 100
    assert all(left < right for left, right in boxes)


def test_recognizer_splits_a_ctc_segment_at_a_wide_visual_gap(tmp_path) -> None:
    characters = tmp_path / "characters.txt"
    characters.write_text("\uac00\n", encoding="utf-8")
    recognizer = PaddleRecognizer(
        tmp_path / "unused.onnx", characters, session=RecognitionSession()
    )
    image = Image.new("RGB", (100, 20), (255, 255, 255))
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
    characters = tmp_path / "characters.txt"
    characters.write_text("\uac00\n", encoding="utf-8")
    session = RecognitionSession()
    recognizer = PaddleRecognizer(tmp_path / "unused.onnx", characters, session=session)

    recognizer.recognize(Image.new("RGB", (1000, 20), (255, 0, 0)))

    assert session.tensor.shape == (1, 3, 48, 2400)


class InitialFinalOverlapRecognizer:
    def __init__(self, confidence: float = 0.99961, text: str = "신기술") -> None:
        self.confidence = confidence
        self.text = text

    def recognize(self, _image):
        return RecognizedText(self.text, self.confidence)


def test_initial_overlapping_pair_merges_exact_final_syllable() -> None:
    words = [
        ("신기", BoundingBox(0, 0, 30, 20), 0.9988),
        ("술", BoundingBox(28.9, 0, 40.9, 20), 0.98),
        ("개발의", BoundingBox(44.31, 0, 86.31, 20), 0.9999),
    ]
    recovered = _recover_initial_overlapping_word_pair(
        words,
        Image.new("RGB", (100, 20)),
        BoundingBox(0, 0, 100, 20),
        InitialFinalOverlapRecognizer(),
    )
    assert recovered == [
        ("신기술", BoundingBox(0, 0, 40.9, 20), 0.98),
        words[2],
    ]


def test_initial_overlapping_pair_requires_following_gap() -> None:
    words = [
        ("신기", BoundingBox(0, 0, 30, 20), 0.9988),
        ("술", BoundingBox(28.9, 0, 40.9, 20), 0.98),
        ("개발의", BoundingBox(44.2, 0, 86.2, 20), 0.9999),
    ]
    assert (
        _recover_initial_overlapping_word_pair(
            words,
            Image.new("RGB", (100, 20)),
            BoundingBox(0, 0, 100, 20),
            InitialFinalOverlapRecognizer(),
        )
        == words
    )


def test_initial_overlapping_pair_requires_exact_combined_recognition() -> None:
    words = [
        ("신기", BoundingBox(0, 0, 30, 20), 0.9988),
        ("술", BoundingBox(28.9, 0, 40.9, 20), 0.98),
        ("개발의", BoundingBox(44.31, 0, 86.31, 20), 0.9999),
    ]
    assert (
        _recover_initial_overlapping_word_pair(
            words,
            Image.new("RGB", (100, 20)),
            BoundingBox(0, 0, 100, 20),
            InitialFinalOverlapRecognizer(text="신기업"),
        )
        == words
    )


class ConfirmedThreePlusThreeRecognizer:
    def __init__(self, second_text: str = "백화점") -> None:
        self.second_text = second_text
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.01
        return ((0, 49), (54, 103))

    def recognize(self, _image):
        values = (
            RecognizedText("신세계", 0.9992),
            RecognizedText(self.second_text, 0.9998),
        )
        result = values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_three_plus_three_split_recovers_low_ctc_space() -> None:
    words = [
        ("앞말", BoundingBox(0, 0, 15, 17), 0.999),
        ("신세계백화점", BoundingBox(20, 0, 123, 17), 0.9993),
        ("뒷말", BoundingBox(130, 0, 150, 17), 0.999),
    ]

    recovered = _recover_confirmed_three_plus_three_splits(
        words,
        Image.new("RGB", (160, 17)),
        BoundingBox(0, 0, 160, 17),
        ConfirmedThreePlusThreeRecognizer(),
    )

    assert recovered == [
        words[0],
        ("신세계", BoundingBox(20, 0, 69, 17), 0.9992),
        ("백화점", BoundingBox(74, 0, 123, 17), 0.9993),
        words[2],
    ]


def test_confirmed_three_plus_three_split_requires_exact_parts() -> None:
    words = [
        ("신세계백화점", BoundingBox(20, 0, 123, 17), 0.9993),
    ]

    assert (
        _recover_confirmed_three_plus_three_splits(
            words,
            Image.new("RGB", (140, 17)),
            BoundingBox(0, 0, 140, 17),
            ConfirmedThreePlusThreeRecognizer("문화점"),
        )
        == words
    )


class ConfirmedThreePlusFiveRecognizer:
    def __init__(self, second_text: str = "서적상에게") -> None:
        self.second_text = second_text
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.02
        return ((0, 49), (54, 137))

    def recognize(self, _image):
        values = (
            RecognizedText("필사해", 0.9997),
            RecognizedText(self.second_text, 0.9989),
        )
        result = values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_three_plus_five_split_recovers_low_ctc_space() -> None:
    words = [
        ("앞말", BoundingBox(0, 0, 15, 16), 0.999),
        ("필사해서적상에게", BoundingBox(20, 0, 157, 16), 0.9965),
        ("뒷말", BoundingBox(164, 0, 184, 16), 0.999),
    ]

    recovered = _recover_confirmed_three_plus_five_splits(
        words,
        Image.new("RGB", (190, 16)),
        BoundingBox(0, 0, 190, 16),
        ConfirmedThreePlusFiveRecognizer(),
    )

    assert recovered == [
        words[0],
        ("필사해", BoundingBox(20, 0, 69, 16), 0.9965),
        ("서적상에게", BoundingBox(74, 0, 157, 16), 0.9965),
        words[2],
    ]


def test_confirmed_three_plus_five_split_requires_exact_parts() -> None:
    words = [
        ("필사해서적상에게", BoundingBox(20, 0, 157, 16), 0.9965),
    ]

    assert (
        _recover_confirmed_three_plus_five_splits(
            words,
            Image.new("RGB", (175, 16)),
            BoundingBox(0, 0, 175, 16),
            ConfirmedThreePlusFiveRecognizer("문학상에게"),
        )
        == words
    )


class RelativeGapTwoPlusTwoRecognizer:
    def __init__(self, confidence: float = 0.999) -> None:
        self.confidence = confidence

    def recognize(self, _image):
        return RecognizedText("한계성에", self.confidence)


def test_relative_gap_two_plus_two_pair_merges_exact_union() -> None:
    words = [
        ("앞말", BoundingBox(0, 0, 30, 20), 0.999),
        ("한계", BoundingBox(40, 0, 72, 20), 0.9994),
        ("성에", BoundingBox(75.2, 0, 108, 20), 0.9998),
        ("뒷말", BoundingBox(115.2, 0, 147.2, 20), 0.999),
    ]

    recovered = _recover_relative_gap_two_plus_two_pairs(
        words,
        Image.new("RGB", (160, 20)),
        BoundingBox(0, 0, 160, 20),
        RelativeGapTwoPlusTwoRecognizer(),
    )

    assert recovered == [
        words[0],
        ("한계성에", BoundingBox(40, 0, 108, 20), 0.999),
        words[3],
    ]


def test_relative_gap_two_plus_two_pair_requires_wider_following_gap() -> None:
    words = [
        ("한계", BoundingBox(0, 0, 32, 20), 0.9994),
        ("성에", BoundingBox(35.2, 0, 68, 20), 0.9998),
        ("뒷말", BoundingBox(73.1, 0, 105.1, 20), 0.999),
    ]

    assert (
        _recover_relative_gap_two_plus_two_pairs(
            words,
            Image.new("RGB", (120, 20)),
            BoundingBox(0, 0, 120, 20),
            RelativeGapTwoPlusTwoRecognizer(),
        )
        == words
    )


def test_relative_gap_two_plus_two_pair_rejects_ordinary_space() -> None:
    words = [
        ("앞말", BoundingBox(0, 0, 32, 20), 0.999),
        ("없다", BoundingBox(40, 0, 65, 20), 0.9992),
        ("보니", BoundingBox(70, 0, 95, 20), 0.9991),
        ("뒷말", BoundingBox(102, 0, 134, 20), 0.999),
    ]

    assert (
        _recover_relative_gap_two_plus_two_pairs(
            words,
            Image.new("RGB", (145, 20)),
            BoundingBox(0, 0, 145, 20),
            RelativeGapTwoPlusTwoRecognizer(),
        )
        == words
    )
