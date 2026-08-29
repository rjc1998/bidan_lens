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
    _recover_confirmed_central_paired_wrapped_two_split,
    _recover_confirmed_direct_retry_regression,
    _recover_confirmed_enhanced_two_substitution,
    _recover_confirmed_enhanced_wrapped_four_substitution,
    _recover_confirmed_five_plus_three_prefix_split,
    _recover_confirmed_four_plus_four_split,
    _recover_confirmed_isolated_paired_wrapped_two_plus_two_split,
    _recover_confirmed_leading_punctuated_single_split,
    _recover_confirmed_leading_three_plus_six_punctuated_split,
    _recover_confirmed_low_confidence_three_plus_five_split,
    _recover_confirmed_mismatched_wrapped_three_plus_one_split,
    _recover_confirmed_numeric_ellipsis_tail_split,
    _recover_confirmed_one_plus_one_split,
    _recover_confirmed_paired_wrapped_four_plus_two_split,
    _recover_confirmed_paired_wrapped_three_plus_three_split,
    _recover_confirmed_paired_wrapper_four_substitution,
    _recover_confirmed_punctuated_three_plus_three_plus_one_split,
    _recover_confirmed_punctuated_three_plus_three_split,
    _recover_confirmed_punctuation_trimmed_single,
    _recover_confirmed_right_wrapper_five_substitution,
    _recover_confirmed_seven_character_splits,
    _recover_confirmed_substitution_readings,
    _recover_confirmed_terminal_punctuated_overlap_pair,
    _recover_confirmed_terminal_three_substitution,
    _recover_confirmed_terminal_wrapped_four_substitution,
    _recover_confirmed_three_plus_five_splits,
    _recover_confirmed_three_plus_three_splits,
    _recover_confirmed_three_plus_two_prefix_split,
    _recover_confirmed_three_plus_two_split,
    _recover_confirmed_three_plus_two_terminal_punctuation_split,
    _recover_confirmed_two_plus_four_splits,
    _recover_confirmed_two_plus_punctuated_two_split,
    _recover_confirmed_two_plus_two_split,
    _recover_confirmed_wrapped_five_plus_four_split,
    _recover_confirmed_wrapped_four_syllable_triplet,
    _recover_confirmed_wrapped_single_geometry,
    _recover_initial_overlapping_word_pair,
    _recover_isolated_close_word_pairs,
    _recover_isolated_overlapping_word_pairs,
    _recover_overlapping_suffix_pairs,
    _recover_overlapping_word_triplets,
    _recover_relative_gap_two_plus_two_pairs,
    _recover_terminal_digit_hangul_pair,
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


class TwoPlusTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(10, 5, 340, 82.48), 0.9),)


class SingleSegmentTwoPlusTwoRecognizer:
    def __init__(self) -> None:
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.01:
            return ((0, 165), (164, 330))
        return ((0, 330),)

    def recognize(self, _image):
        values = (
            RecognizedText("\uac00\ub098\ub2e4\ub77c", 0.99987),
            RecognizedText("\uac00\ub098", 0.99994),
            RecognizedText("\ub2e4\ub77c", 0.99995),
        )
        result = values[self.recognition_calls]
        self.recognition_calls += 1
        return result


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


def test_engine_recovers_confirmed_two_plus_two_single_segment_line() -> None:
    engine = PaddleOcrEngine(
        TwoPlusTwoDetector(),
        SingleSegmentTwoPlusTwoRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (360, 100)))

    assert document.lines[0].text == "\uac00\ub098 \ub2e4\ub77c"
    assert [word.text for word in document.lines[0].eojeols] == [
        "\uac00\ub098",
        "\ub2e4\ub77c",
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


class ReviewedThreePlusOneRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        confidence: float,
        *,
        strong_competitor_width: int | None = None,
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.confidence = confidence
        self.strong_competitor_width = strong_competitor_width

    def recognize(self, image):
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, self.confidence)
        competitor = 0.99 if image.width == self.strong_competitor_width else 0.98
        return RecognizedText("competing", competitor)


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


def test_isolated_three_plus_one_pair_merges_with_exact_weakly_competed_union() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ube44\ud310\ud574", BoundingBox(49.1, 0, 109.1, 20), 0.9999),
        ("\uc57c", BoundingBox(114.78, 0, 131.18, 20), 0.9989),
        ("\ub2e4\uc74c", BoundingBox(142.4, 0, 182.4, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (195, 20)),
        BoundingBox(0, 0, 195, 20),
        ReviewedThreePlusOneRecognizer(83, "\ube44\ud310\ud574\uc57c", 0.99985),
    )

    assert recovered == [
        words[0],
        ("\ube44\ud310\ud574\uc57c", BoundingBox(49.1, 0, 131.18, 20), 0.9989),
        words[3],
    ]


def test_isolated_three_plus_one_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ube44\ud310\ud574", BoundingBox(49.1, 0, 109.1, 20), 0.9999),
        ("\uc57c", BoundingBox(114.78, 0, 131.18, 20), 0.9989),
        ("\ub2e4\uc74c", BoundingBox(142.4, 0, 182.4, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (195, 20)),
            BoundingBox(0, 0, 195, 20),
            ReviewedThreePlusOneRecognizer(
                83,
                "\ube44\ud310\ud574\uc57c",
                0.99985,
                strong_competitor_width=69,
            ),
        )
        == words
    )


def test_isolated_wide_three_plus_one_pair_merges_exact_union() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ube44\ud310\ud574", BoundingBox(50.32, 0, 108.32, 20), 0.99975),
        ("\uc57c", BoundingBox(115.55, 0, 132.55, 20), 0.9992),
        ("\ub2e4\uc74c", BoundingBox(143.91, 0, 183.91, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (195, 20)),
        BoundingBox(0, 0, 195, 20),
        ReviewedThreePlusOneRecognizer(83, "\ube44\ud310\ud574\uc57c", 0.99975),
    )

    assert recovered == [
        words[0],
        ("\ube44\ud310\ud574\uc57c", BoundingBox(50.32, 0, 132.55, 20), 0.9992),
        words[3],
    ]


def test_isolated_wide_three_plus_one_pair_rejects_weak_union() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ube44\ud310\ud574", BoundingBox(50.32, 0, 108.32, 20), 0.99975),
        ("\uc57c", BoundingBox(115.55, 0, 132.55, 20), 0.9992),
        ("\ub2e4\uc74c", BoundingBox(143.91, 0, 183.91, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (195, 20)),
            BoundingBox(0, 0, 195, 20),
            ReviewedThreePlusOneRecognizer(83, "\ube44\ud310\ud574\uc57c", 0.99969),
        )
        == words
    )


def test_isolated_wide_three_plus_one_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\ube44\ud310\ud574", BoundingBox(50.32, 0, 108.32, 20), 0.99975),
        ("\uc57c", BoundingBox(115.55, 0, 132.55, 20), 0.9992),
        ("\ub2e4\uc74c", BoundingBox(143.91, 0, 183.91, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (195, 20)),
            BoundingBox(0, 0, 195, 20),
            ReviewedThreePlusOneRecognizer(
                83,
                "\ube44\ud310\ud574\uc57c",
                0.99975,
                strong_competitor_width=69,
            ),
        )
        == words
    )


def test_overlapping_three_plus_one_pair_accepts_one_union_substitution() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 37.6, 20), 0.999),
        ("\ube44\ud310\ud788", BoundingBox(50.1, 0, 110.1, 20), 0.9986),
        ("\uc57c", BoundingBox(108.96, 0, 120.76, 20), 0.912),
        ("\ub2e4\uc74c", BoundingBox(126.46, 0, 166.46, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (180, 20)),
        BoundingBox(0, 0, 180, 20),
        ReviewedThreePlusOneRecognizer(71, "\ube44\ud310\ud574\uc57c", 0.9996),
    )

    assert recovered == [
        words[0],
        ("\ube44\ud310\ud574\uc57c", BoundingBox(50.1, 0, 120.76, 20), 0.912),
        words[3],
    ]


def test_overlapping_three_plus_one_pair_rejects_two_union_substitutions() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 37.6, 20), 0.999),
        ("\ube44\ud310\ud788", BoundingBox(50.1, 0, 110.1, 20), 0.9986),
        ("\uc57c", BoundingBox(108.96, 0, 120.76, 20), 0.912),
        ("\ub2e4\uc74c", BoundingBox(126.46, 0, 166.46, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (180, 20)),
            BoundingBox(0, 0, 180, 20),
            ReviewedThreePlusOneRecognizer(71, "\ube44\ud3c9\ud574\uc57c", 0.9996),
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


class ReviewedWideOnePlusTwoRecognizer:
    def __init__(self, *, competing_confidence: float = 0.97) -> None:
        self.competing_confidence = competing_confidence

    def recognize(self, image):
        if image.width == 60:
            return RecognizedText("\uac1c\uc778\uc744", 0.99982)
        return RecognizedText("competing", self.competing_confidence)


def test_isolated_wide_one_plus_two_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uac1c", BoundingBox(55.5, 0, 69.5, 20), 0.836),
        ("\uc778\uc744", BoundingBox(76.73, 0, 114.73, 20), 0.9989),
        ("\ub2e4\uc74c", BoundingBox(126.95, 0, 166.95, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (180, 20)),
        BoundingBox(0, 0, 180, 20),
        ReviewedWideOnePlusTwoRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uac1c\uc778\uc744", BoundingBox(55.5, 0, 114.73, 20), 0.836),
        words[3],
    ]


def test_isolated_wide_one_plus_two_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("\uac1c", BoundingBox(55.5, 0, 69.5, 20), 0.836),
        ("\uc778\uc744", BoundingBox(76.73, 0, 114.73, 20), 0.9989),
        ("\ub2e4\uc74c", BoundingBox(126.95, 0, 166.95, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (180, 20)),
            BoundingBox(0, 0, 180, 20),
            ReviewedWideOnePlusTwoRecognizer(competing_confidence=0.98),
        )
        == words
    )


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


def test_overlapping_four_plus_one_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 30, 20), 0.999),
        ("\uac00\ub098\ub2e4\ub77c", BoundingBox(36, 0, 107, 20), 0.9997),
        ("\ub9c8", BoundingBox(106.05, 0, 128.05, 20), 0.9141),
        ("\ub2e4\uc74c", BoundingBox(136, 0, 196, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (205, 20)),
        BoundingBox(0, 0, 205, 20),
        CompetitiveOnePlusTwoRecognizer(
            93,
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
        ),
    )

    assert recovered == [
        words[0],
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(36, 0, 128.05, 20),
            0.9141,
        ),
        words[3],
    ]


def test_overlapping_four_plus_one_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 30, 20), 0.999),
        ("\uac00\ub098\ub2e4\ub77c", BoundingBox(36, 0, 107, 20), 0.9997),
        ("\ub9c8", BoundingBox(106.05, 0, 128.05, 20), 0.9141),
        ("\ub2e4\uc74c", BoundingBox(136, 0, 196, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (205, 20)),
            BoundingBox(0, 0, 205, 20),
            CompetitiveOnePlusTwoRecognizer(
                93,
                "\uac00\ub098\ub2e4\ub77c\ub9c8",
                competing_confidence=0.98,
            ),
        )
        == words
    )


def test_isolated_one_plus_four_pair_merges_with_weak_competitors() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 80, 20), 0.999),
        ("\uac00", BoundingBox(91, 0, 109, 20), 0.9985),
        ("\ub098\ub2e4\ub77c\ub9c8", BoundingBox(116, 0, 196, 20), 0.9998),
        ("\ub2e4\uc74c", BoundingBox(210, 0, 250, 20), 0.999),
    ]

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (260, 20)),
        BoundingBox(0, 0, 260, 20),
        CompetitiveOnePlusTwoRecognizer(
            105,
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
        ),
    )

    assert recovered == [
        words[0],
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(91, 0, 196, 20),
            0.9985,
        ),
        words[3],
    ]


def test_isolated_one_plus_four_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 80, 20), 0.999),
        ("\uac00", BoundingBox(91, 0, 109, 20), 0.9985),
        ("\ub098\ub2e4\ub77c\ub9c8", BoundingBox(116, 0, 196, 20), 0.9998),
        ("\ub2e4\uc74c", BoundingBox(210, 0, 250, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (260, 20)),
            BoundingBox(0, 0, 260, 20),
            CompetitiveOnePlusTwoRecognizer(
                105,
                "\uac00\ub098\ub2e4\ub77c\ub9c8",
                competing_confidence=0.998,
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


class TerminalDigitHangulRecognizer:
    def __init__(self, *, competing_confidence: float = 0.5) -> None:
        self.competing_confidence = competing_confidence

    def recognize(self, image):
        if image.width == 88:
            return RecognizedText("12\ub144\ub9d0", 0.99925)
        return RecognizedText("competing", self.competing_confidence)


def test_terminal_digit_hangul_pair_merges_with_weak_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("12\ub144", BoundingBox(52.5, 0, 112.5, 20), 0.9962),
        ("\ub9d0", BoundingBox(119.53, 0, 139.53, 20), 0.9997),
    ]

    assert _recover_terminal_digit_hangul_pair(
        words,
        Image.new("RGB", (150, 20)),
        BoundingBox(0, 0, 150, 20),
        TerminalDigitHangulRecognizer(),
    ) == [
        words[0],
        ("12\ub144\ub9d0", BoundingBox(52.5, 0, 139.53, 20), 0.9962),
    ]


def test_terminal_digit_hangul_pair_rejects_strong_competitor() -> None:
    words = [
        ("\uc774\uc804", BoundingBox(0, 0, 40, 20), 0.999),
        ("12\ub144", BoundingBox(52.5, 0, 112.5, 20), 0.9962),
        ("\ub9d0", BoundingBox(119.53, 0, 139.53, 20), 0.9997),
    ]

    assert (
        _recover_terminal_digit_hangul_pair(
            words,
            Image.new("RGB", (150, 20)),
            BoundingBox(0, 0, 150, 20),
            TerminalDigitHangulRecognizer(competing_confidence=0.99),
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


class ConfirmedFourPlusFourRecognizer:
    def __init__(
        self,
        second_text: str = '가리기에',
        second_confidence: float = 0.9997,
        segments: tuple[tuple[int, int], ...] = ((0, 60), (64, 124)),
    ) -> None:
        self.values = (
            RecognizedText('정당성을', 0.99965),
            RecognizedText(second_text, second_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.04
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_four_plus_four_split_recovers_reviewed_profile() -> None:
    height = 14.08
    words = [
        ('앞말', BoundingBox(0, 0, 15, height), 0.999),
        ('정당성을가리기에', BoundingBox(20, 0, 144, height), 0.9981),
        ('뒷말', BoundingBox(150, 0, 170, height), 0.999),
    ]

    recovered = _recover_confirmed_four_plus_four_split(
        words,
        Image.new('RGB', (180, 15)),
        BoundingBox(0, 0, 180, height),
        ConfirmedFourPlusFourRecognizer(),
    )

    assert recovered == [
        words[0],
        ('정당성을', BoundingBox(20, 0, 80, height), 0.9981),
        ('가리기에', BoundingBox(84, 0, 144, height), 0.9981),
        words[2],
    ]


@pytest.mark.parametrize(
    'recognizer',
    [
        ConfirmedFourPlusFourRecognizer(second_text='가리기도'),
        ConfirmedFourPlusFourRecognizer(second_confidence=0.99959),
        ConfirmedFourPlusFourRecognizer(segments=((0, 60), (65, 124))),
        ConfirmedFourPlusFourRecognizer(segments=((0, 59), (63, 124))),
    ],
)
def test_confirmed_four_plus_four_split_requires_exact_reviewed_profile(
    recognizer,
) -> None:
    words = [
        ('정당성을가리기에', BoundingBox(20, 0, 144, 15), 0.9981),
    ]

    assert (
        _recover_confirmed_four_plus_four_split(
            words,
            Image.new('RGB', (160, 15)),
            BoundingBox(0, 0, 160, 15),
            recognizer,
        )
        == words
    )


def test_confirmed_four_plus_four_split_requires_word_confidence() -> None:
    words = [
        ('정당성을가리기에', BoundingBox(20, 0, 144, 15), 0.9979),
    ]

    assert (
        _recover_confirmed_four_plus_four_split(
            words,
            Image.new('RGB', (160, 15)),
            BoundingBox(0, 0, 160, 15),
            ConfirmedFourPlusFourRecognizer(),
        )
        == words
    )


class ConfirmedNumericEllipsisTailRecognizer:
    def __init__(
        self,
        *,
        padded_text: str = "\uac00\ub098\ub2e4",
        padded_confidence: float = 0.9892,
        first_part_text: str = "\uac00\ub098\ub2e4\u2026",
        last_part_text: str = "1",
        segments: tuple[tuple[int, int], ...] = ((0, 72), (71, 81)),
    ) -> None:
        self.values = (
            RecognizedText("\uac00\ub098\ub2e4", 0.9902),
            RecognizedText(padded_text, padded_confidence),
            RecognizedText(first_part_text, 0.9958),
            RecognizedText(last_part_text, 0.9978),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.04
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_numeric_ellipsis_tail_recovers_word_boundary() -> None:
    height = 17.6
    words = [
        ("\uac00\ub098\ub2e41\u2026", BoundingBox(20, 0, 101, height), 0.9947),
    ]

    assert _recover_confirmed_numeric_ellipsis_tail_split(
        words,
        Image.new("RGB", (121, 19)),
        BoundingBox(0, 0, 121, height),
        ConfirmedNumericEllipsisTailRecognizer(),
    ) == [
        ("\uac00\ub098\ub2e4\u2026", BoundingBox(20, 0, 92, height), 0.9892),
        ("1", BoundingBox(91, 0, 101, height), 0.9892),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedNumericEllipsisTailRecognizer(
            padded_text="\uac00\ub098\ub77c"
        ),
        ConfirmedNumericEllipsisTailRecognizer(padded_confidence=0.9889),
        ConfirmedNumericEllipsisTailRecognizer(
            first_part_text="\uac00\ub098\ub2e41"
        ),
        ConfirmedNumericEllipsisTailRecognizer(last_part_text="2"),
        ConfirmedNumericEllipsisTailRecognizer(
            segments=((0, 72), (70, 81))
        ),
        ConfirmedNumericEllipsisTailRecognizer(
            segments=((0, 71), (71, 81))
        ),
        ConfirmedNumericEllipsisTailRecognizer(
            segments=((2, 72), (71, 79))
        ),
    ],
)
def test_confirmed_numeric_ellipsis_tail_requires_exact_profile(
    recognizer,
) -> None:
    words = [
        ("\uac00\ub098\ub2e41\u2026", BoundingBox(20, 0, 101, 17.6), 0.9947),
    ]

    assert (
        _recover_confirmed_numeric_ellipsis_tail_split(
            words,
            Image.new("RGB", (121, 19)),
            BoundingBox(0, 0, 121, 17.6),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        [("\uac00\ub098\ub2e4A\u2026", BoundingBox(20, 0, 101, 17.6), 0.9947)],
        [("\uac00\ub098\ub2e41!", BoundingBox(20, 0, 101, 17.6), 0.9947)],
        [("\uac00\ub098\ub2e41\u2026", BoundingBox(20, 0, 101, 17.6), 0.9939)],
    ],
)
def test_confirmed_numeric_ellipsis_tail_requires_word_shape(words) -> None:
    assert (
        _recover_confirmed_numeric_ellipsis_tail_split(
            words,
            Image.new("RGB", (121, 19)),
            BoundingBox(0, 0, 121, 17.6),
            ConfirmedNumericEllipsisTailRecognizer(),
        )
        == words
    )


class ConfirmedOnePlusOneRecognizer:
    def __init__(
        self,
        *,
        first_part_text: str = "\uac00",
        last_part_text: str = "\ub098",
        first_part_confidence: float = 0.99996,
        variant_text: str = "\uac00",
        variant_confidence: float = 0.99981,
        segments: tuple[tuple[int, int], ...] = ((0, 27), (37, 69)),
    ) -> None:
        self.values = (
            RecognizedText(first_part_text, first_part_confidence),
            RecognizedText(last_part_text, 0.99998),
            RecognizedText(variant_text, variant_confidence),
            RecognizedText("\ub098", 0.99998),
            RecognizedText("\uac00", 0.9999),
            RecognizedText("\ub098", 0.99998),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_one_plus_one_recovers_word_boundary() -> None:
    height = 69 / 2.175
    words = [("\uac00\ub098", BoundingBox(20, 0, 89, height), 0.99991)]

    assert _recover_confirmed_one_plus_one_split(
        words,
        Image.new("RGB", (109, 32)),
        BoundingBox(0, 0, 109, height),
        ConfirmedOnePlusOneRecognizer(),
    ) == [
        ("\uac00", BoundingBox(20, 0, 47, height), 0.99981),
        ("\ub098", BoundingBox(57, 0, 89, height), 0.99981),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedOnePlusOneRecognizer(first_part_text="\ub2e4"),
        ConfirmedOnePlusOneRecognizer(first_part_confidence=0.99994),
        ConfirmedOnePlusOneRecognizer(variant_text="\ub2e4"),
        ConfirmedOnePlusOneRecognizer(variant_confidence=0.99979),
        ConfirmedOnePlusOneRecognizer(segments=((0, 27), (36, 69))),
        ConfirmedOnePlusOneRecognizer(segments=((0, 26), (37, 69))),
        ConfirmedOnePlusOneRecognizer(segments=((2, 27), (37, 67))),
    ],
)
def test_confirmed_one_plus_one_requires_exact_profile(recognizer) -> None:
    height = 69 / 2.175
    words = [("\uac00\ub098", BoundingBox(20, 0, 89, height), 0.99991)]

    assert (
        _recover_confirmed_one_plus_one_split(
            words,
            Image.new("RGB", (109, 32)),
            BoundingBox(0, 0, 109, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "height"),
    [
        (
            [("\uac00A", BoundingBox(20, 0, 89, 69 / 2.175), 0.99991)],
            69 / 2.175,
        ),
        (
            [("\uac00\ub098", BoundingBox(20, 0, 89, 69 / 2.175), 0.99989)],
            69 / 2.175,
        ),
        (
            [("\uac00\ub098", BoundingBox(20, 0, 89, 69 / 2.175), 0.99993)],
            69 / 2.175,
        ),
        (
            [("\uac00\ub098", BoundingBox(20, 0, 89, 31), 0.99991)],
            31,
        ),
    ],
)
def test_confirmed_one_plus_one_requires_word_profile(words, height) -> None:
    assert (
        _recover_confirmed_one_plus_one_split(
            words,
            Image.new("RGB", (109, 32)),
            BoundingBox(0, 0, 109, height),
            ConfirmedOnePlusOneRecognizer(),
        )
        == words
    )


class ConfirmedThreePlusTwoPrefixRecognizer:
    def __init__(
        self,
        *,
        first_ctc_text: str = "\uac00\ub098\ub2e4\u2026",
        last_ctc_text: str = "\ub77c\ub9c8",
        first_ctc_confidence: float = 0.55138,
        last_ctc_confidence: float = 0.9904,
        variant_first_text: str = "\uac00\ub098\ub2e4",
        variant_last_text: str = "\ub77c\ub9c8",
        variant_confidence: float = 0.99893,
        segments: tuple[tuple[int, int], ...] = ((0, 72), (71, 105)),
    ) -> None:
        self.values = (
            RecognizedText(first_ctc_text, first_ctc_confidence),
            RecognizedText(last_ctc_text, last_ctc_confidence),
            RecognizedText(variant_first_text, 0.99953),
            RecognizedText(variant_last_text, 0.99949),
            RecognizedText(variant_first_text, 0.99932),
            RecognizedText(variant_last_text, variant_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.04
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_three_plus_two_prefix_recovers_word_boundary() -> None:
    height = 105 / 5.965
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(20, 0, 125, height),
            0.99905,
        )
    ]

    assert _recover_confirmed_three_plus_two_prefix_split(
        words,
        Image.new("RGB", (145, 19)),
        BoundingBox(0, 0, 145, height),
        ConfirmedThreePlusTwoPrefixRecognizer(),
    ) == [
        (
            "\uac00\ub098\ub2e4",
            BoundingBox(20, 0, 71, height),
            0.99893,
        ),
        (
            "\ub77c\ub9c8",
            BoundingBox(91, 0, 125, height),
            0.9904,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedThreePlusTwoPrefixRecognizer(
            first_ctc_text="\uac00\ub098\ub2e4\ub77c"
        ),
        ConfirmedThreePlusTwoPrefixRecognizer(first_ctc_confidence=0.5499),
        ConfirmedThreePlusTwoPrefixRecognizer(last_ctc_text="\ub77c\ubc14"),
        ConfirmedThreePlusTwoPrefixRecognizer(last_ctc_confidence=0.9899),
        ConfirmedThreePlusTwoPrefixRecognizer(variant_first_text="\uac00\ub098\ub77c"),
        ConfirmedThreePlusTwoPrefixRecognizer(variant_last_text="\ub77c\ubc14"),
        ConfirmedThreePlusTwoPrefixRecognizer(variant_confidence=0.9988),
        ConfirmedThreePlusTwoPrefixRecognizer(segments=((0, 72), (70, 105))),
        ConfirmedThreePlusTwoPrefixRecognizer(segments=((0, 71), (70, 105))),
        ConfirmedThreePlusTwoPrefixRecognizer(segments=((2, 72), (71, 103))),
    ],
)
def test_confirmed_three_plus_two_prefix_requires_exact_profile(
    recognizer,
) -> None:
    height = 105 / 5.965
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(20, 0, 125, height),
            0.99905,
        )
    ]

    assert (
        _recover_confirmed_three_plus_two_prefix_split(
            words,
            Image.new("RGB", (145, 19)),
            BoundingBox(0, 0, 145, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "height"),
    [
        (
            [
                (
                    "\uac00\ub098\ub2e4\ub77cA",
                    BoundingBox(20, 0, 125, 105 / 5.965),
                    0.99905,
                )
            ],
            105 / 5.965,
        ),
        (
            [
                (
                    "\uac00\ub098\ub2e4\ub77c\ub9c8",
                    BoundingBox(20, 0, 125, 105 / 5.965),
                    0.9989,
                )
            ],
            105 / 5.965,
        ),
        (
            [
                (
                    "\uac00\ub098\ub2e4\ub77c\ub9c8",
                    BoundingBox(20, 0, 125, 105 / 5.965),
                    0.99911,
                )
            ],
            105 / 5.965,
        ),
        (
            [
                (
                    "\uac00\ub098\ub2e4\ub77c\ub9c8",
                    BoundingBox(20, 0, 125, 17.5),
                    0.99905,
                )
            ],
            17.5,
        ),
    ],
)
def test_confirmed_three_plus_two_prefix_requires_word_profile(
    words,
    height,
) -> None:
    assert (
        _recover_confirmed_three_plus_two_prefix_split(
            words,
            Image.new("RGB", (145, 19)),
            BoundingBox(0, 0, 145, height),
            ConfirmedThreePlusTwoPrefixRecognizer(),
        )
        == words
    )

class ConfirmedThreePlusTwoTerminalPunctuationRecognizer:
    def __init__(
        self,
        *,
        first_text: str = "가나다",
        first_confidence: float = 0.99993,
        middle_text: str = "-",
        middle_confidence: float = 0.4966,
        last_text: str = "라마.",
        last_confidence: float = 0.9773,
        target_variant_text: str = "가나다",
        target_variant_confidence: float = 0.99994,
        punctuated_target_variant_text: str = "가나다…",
        punctuated_target_variant_confidence: float = 0.99994,
        second_punctuated_target_variant_text: str | None = None,
        suffix_variant_text: str = "라마.",
        suffix_variant_confidence: float = 0.9881,
        segments: tuple[tuple[int, int], ...] = (
            (0, 53),
            (52, 70),
            (76, 113),
        ),
    ) -> None:
        self.values = (
            RecognizedText(first_text, first_confidence),
            RecognizedText(middle_text, middle_confidence),
            RecognizedText(last_text, last_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.99995),
            RecognizedText(
                punctuated_target_variant_text,
                punctuated_target_variant_confidence,
            ),
            RecognizedText(
                second_punctuated_target_variant_text
                or punctuated_target_variant_text,
                0.99995,
            ),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.996),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.002
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SegmentedThreePlusTwoTerminalPunctuationRecognizer(
    ConfirmedThreePlusTwoTerminalPunctuationRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("가", 0.9999),
            RecognizedText("가나다라마.", 0.99175),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((0, 15), (20, 133))
        return super().word_boxes(_image, space_threshold)


class ThreePlusTwoTerminalPunctuationDetector:
    def detect(self, _image):
        height = 113 / 6.417
        return (DetectedRegion(BoundingBox(10.4, 5, 143.4, 5 + height), 0.9),)


def test_engine_recovers_three_plus_two_terminal_punctuation_line() -> None:
    engine = PaddleOcrEngine(
        ThreePlusTwoTerminalPunctuationDetector(),
        SegmentedThreePlusTwoTerminalPunctuationRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (160, 40)))

    assert document.lines[0].text == "가 가나다… 라마."
    assert [word.text for word in document.lines[0].eojeols] == [
        "가",
        "가나다",
        "라마",
    ]


def test_confirmed_three_plus_two_terminal_punctuation_recovers() -> None:
    height = 113 / 6.417
    words = [
        ("가나다라마.", BoundingBox(20, 0, 133, height), 0.99175)
    ]

    assert _recover_confirmed_three_plus_two_terminal_punctuation_split(
        words,
        Image.new("RGB", (153, 19)),
        BoundingBox(0, 0, 153, height),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(),
    ) == [
        ("가나다…", BoundingBox(20, 0, 73, height), 0.99175),
        ("라마.", BoundingBox(96, 0, 133, height), 0.9773),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(first_text="가나마"),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(first_confidence=0.9998),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(middle_text="가"),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(middle_confidence=0.4959),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(last_text="라바."),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(last_confidence=0.9769),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            target_variant_text="가나마"
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            target_variant_confidence=0.9998
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            punctuated_target_variant_text="가나다."
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            punctuated_target_variant_confidence=0.9998
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            second_punctuated_target_variant_text="가나다:"
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            suffix_variant_text="라바."
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            suffix_variant_confidence=0.9879
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            segments=((2, 53), (52, 70), (76, 113))
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            segments=((0, 53), (53, 70), (76, 113))
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            segments=((0, 53), (52, 70), (75, 113))
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            segments=((0, 50), (49, 70), (76, 113))
        ),
        ConfirmedThreePlusTwoTerminalPunctuationRecognizer(
            segments=((0, 53), (52, 70), (76, 111))
        ),
    ],
)
def test_confirmed_three_plus_two_terminal_punctuation_requires_profile(
    recognizer,
) -> None:
    height = 113 / 6.417
    words = [
        ("가나다라마.", BoundingBox(20, 0, 133, height), 0.99175)
    ]

    assert (
        _recover_confirmed_three_plus_two_terminal_punctuation_split(
            words,
            Image.new("RGB", (153, 19)),
            BoundingBox(0, 0, 153, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("text", "confidence", "height"),
    [
        ("가나다라A.", 0.99175, 113 / 6.417),
        ("가나다라마A", 0.99175, 113 / 6.417),
        ("가나다라마.", 0.99169, 113 / 6.417),
        ("가나다라마.", 0.99181, 113 / 6.417),
        ("가나다라마.", 0.99175, 113 / 6.405),
    ],
)
def test_confirmed_three_plus_two_terminal_punctuation_requires_word_shape(
    text,
    confidence,
    height,
) -> None:
    words = [(text, BoundingBox(20, 0, 133, height), confidence)]

    assert (
        _recover_confirmed_three_plus_two_terminal_punctuation_split(
            words,
            Image.new("RGB", (153, 19)),
            BoundingBox(0, 0, 153, height),
            ConfirmedThreePlusTwoTerminalPunctuationRecognizer(),
        )
        == words
    )

class ConfirmedFivePlusThreePrefixRecognizer:
    def __init__(
        self,
        *,
        first_ctc_text: str = "\uac00\ub098\ub2e4\ub77c\ub9c8\u2026",
        last_ctc_text: str = "\ubc14\uc0ac\uc544",
        first_ctc_confidence: float = 0.8784,
        last_ctc_confidence: float = 0.99991,
        variant_first_text: str = "\uac00\ub098\ub2e4\ub77c\ub9c8",
        variant_last_text: str = "\ubc14\uc0ac\uc544",
        variant_confidence: float = 0.9967,
        segments: tuple[tuple[int, int], ...] = ((0, 82), (87, 128)),
    ) -> None:
        self.values = (
            RecognizedText(first_ctc_text, first_ctc_confidence),
            RecognizedText(last_ctc_text, last_ctc_confidence),
            RecognizedText(variant_first_text, 0.9996),
            RecognizedText(variant_last_text, variant_confidence),
            RecognizedText(variant_first_text, 0.9994),
            RecognizedText(variant_last_text, 0.9988),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.04
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_five_plus_three_prefix_recovers_word_boundary() -> None:
    height = 128 / 6.61
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac\uc544",
            BoundingBox(20, 0, 148, height),
            0.99961,
        )
    ]

    assert _recover_confirmed_five_plus_three_prefix_split(
        words,
        Image.new("RGB", (168, 20)),
        BoundingBox(0, 0, 168, height),
        ConfirmedFivePlusThreePrefixRecognizer(),
    ) == [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(20, 0, 83, height),
            0.9967,
        ),
        (
            "\ubc14\uc0ac\uc544",
            BoundingBox(107, 0, 148, height),
            0.9967,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedFivePlusThreePrefixRecognizer(
            first_ctc_text="\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14"
        ),
        ConfirmedFivePlusThreePrefixRecognizer(first_ctc_confidence=0.8779),
        ConfirmedFivePlusThreePrefixRecognizer(last_ctc_text="\ubc14\uc0ac\uc790"),
        ConfirmedFivePlusThreePrefixRecognizer(last_ctc_confidence=0.99989),
        ConfirmedFivePlusThreePrefixRecognizer(variant_first_text="\uac00\ub098\ub2e4\ub77c\ubc14"),
        ConfirmedFivePlusThreePrefixRecognizer(variant_last_text="\ubc14\uc0ac\uc790"),
        ConfirmedFivePlusThreePrefixRecognizer(variant_confidence=0.9959),
        ConfirmedFivePlusThreePrefixRecognizer(segments=((0, 82), (86, 128))),
        ConfirmedFivePlusThreePrefixRecognizer(segments=((0, 81), (87, 128))),
        ConfirmedFivePlusThreePrefixRecognizer(segments=((2, 82), (87, 126))),
    ],
)
def test_confirmed_five_plus_three_prefix_requires_exact_profile(
    recognizer,
) -> None:
    height = 128 / 6.61
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac\uc544",
            BoundingBox(20, 0, 148, height),
            0.99961,
        )
    ]

    assert (
        _recover_confirmed_five_plus_three_prefix_split(
            words,
            Image.new("RGB", (168, 20)),
            BoundingBox(0, 0, 168, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("text", "confidence", "height"),
    [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0acA",
            0.99961,
            128 / 6.61,
        ),
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac\uc544",
            0.99959,
            128 / 6.61,
        ),
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac\uc544",
            0.99963,
            128 / 6.61,
        ),
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac\uc544",
            0.99961,
            19.2,
        ),
    ],
)
def test_confirmed_five_plus_three_prefix_requires_word_profile(
    text,
    confidence,
    height,
) -> None:
    words = [(text, BoundingBox(20, 0, 148, height), confidence)]

    assert (
        _recover_confirmed_five_plus_three_prefix_split(
            words,
            Image.new("RGB", (168, 20)),
            BoundingBox(0, 0, 168, height),
            ConfirmedFivePlusThreePrefixRecognizer(),
        )
        == words
    )


_CENTRAL_PREFIX = "".join(map(chr, (0xAC00, 0xB098, 0xB2E4, 0xB77C)))
_CENTRAL_TARGET = "".join(map(chr, (0xB9C8, 0xBC14)))
_CENTRAL_SUFFIX = "".join(map(chr, (0xC0AC, 0xC544, 0xC790, 0xCC28)))
_CENTRAL_FIRST = _CENTRAL_PREFIX + chr(0xB9C8)
_CENTRAL_FOLLOWING = "".join(map(chr, (0xAC00, 0xB098)))
_CENTRAL_TRAILING = "".join(map(chr, (0xB2E4, 0xB77C, 0xB9C8, 0xBC14)))
_CENTRAL_RAW = (
    _CENTRAL_PREFIX
    + chr(0x201C)
    + _CENTRAL_TARGET
    + chr(0x2019)
    + _CENTRAL_SUFFIX
)
_CENTRAL_MIDDLE = chr(0x2018) + _CENTRAL_TARGET
_CENTRAL_WRAPPER = chr(0x201C) + _CENTRAL_TARGET + chr(0x201D)


class ConfirmedCentralPairedWrappedTwoRecognizer:
    def __init__(
        self,
        *,
        prefix_text: str = _CENTRAL_PREFIX,
        prefix_confidence: float = 0.999253,
        middle_text: str = _CENTRAL_MIDDLE,
        middle_confidence: float = 0.839075,
        closing_text: str = chr(0x2019),
        closing_confidence: float = 0.951414,
        suffix_text: str = _CENTRAL_SUFFIX,
        suffix_confidence: float = 0.998996,
        prefix_variant_text: str = _CENTRAL_PREFIX,
        prefix_variant_confidence: float = 0.995875,
        target_variant_text: str = _CENTRAL_TARGET,
        target_variant_confidence: float = 0.999595,
        wrapper_variant_text: str = _CENTRAL_WRAPPER,
        wrapper_variant_confidence: float = 0.58446,
        suffix_variant_text: str = _CENTRAL_SUFFIX,
        suffix_variant_confidence: float = 0.971297,
        segments: tuple[tuple[int, int], ...] = (
            (0, 107),
            (115, 176),
            (175, 190),
            (199, 303),
        ),
    ) -> None:
        self.values = (
            RecognizedText(prefix_text, prefix_confidence),
            RecognizedText(middle_text, middle_confidence),
            RecognizedText(closing_text, closing_confidence),
            RecognizedText(suffix_text, suffix_confidence),
            *(
                RecognizedText(prefix_variant_text, prefix_variant_confidence)
                for _ in range(5)
            ),
            *(
                RecognizedText(target_variant_text, target_variant_confidence)
                for _ in range(5)
            ),
            *(
                RecognizedText(wrapper_variant_text, wrapper_variant_confidence)
                for _ in range(6)
            ),
            *(
                RecognizedText(suffix_variant_text, suffix_variant_confidence)
                for _ in range(5)
            ),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.001:
            return self.segments
        return ((0, _image.width),)

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class CentralPairedWrappedTwoRecognizer(
    ConfirmedCentralPairedWrappedTwoRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText(_CENTRAL_FIRST, 0.999845),
            RecognizedText(_CENTRAL_RAW, 0.636831),
            RecognizedText(_CENTRAL_RAW, 0.766935),
            RecognizedText(_CENTRAL_FOLLOWING, 0.99977),
            RecognizedText(_CENTRAL_TRAILING, 0.999783),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((40, 170), (181, 484), (494, 548), (556, 662))
        return super().word_boxes(_image, space_threshold)


class CentralPairedWrappedTwoDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(81.52, 236.7391, 784.48, 263.1522),
                0.9,
            ),
        )


def central_paired_wrapped_two_words(
    *,
    first_text: str = _CENTRAL_FIRST,
    candidate_text: str = _CENTRAL_RAW,
    candidate_confidence: float = 0.766935,
    candidate_box: BoundingBox | None = None,
    following_text: str = _CENTRAL_FOLLOWING,
    following_confidence: float = 0.99977,
    trailing_text: str = _CENTRAL_TRAILING,
    trailing_confidence: float = 0.999783,
) -> list[tuple[str, BoundingBox, float]]:
    return [
        (first_text, BoundingBox(40, 0, 170, 26.4131), 0.999845),
        (
            candidate_text,
            candidate_box or BoundingBox(181, 0, 484, 26.4131),
            candidate_confidence,
        ),
        (
            following_text,
            BoundingBox(494, 0, 548, 26.4131),
            following_confidence,
        ),
        (
            trailing_text,
            BoundingBox(556, 0, 662, 26.4131),
            trailing_confidence,
        ),
    ]


def test_confirmed_central_paired_wrapped_two_recovers() -> None:
    recovered = _recover_confirmed_central_paired_wrapped_two_split(
        central_paired_wrapped_two_words(),
        Image.new("RGB", (704, 28)),
        BoundingBox(0, 0, 704, 26.4131),
        ConfirmedCentralPairedWrappedTwoRecognizer(),
    )

    assert recovered[1:] == [
        (
            _CENTRAL_PREFIX,
            BoundingBox(181, 0, 288, 26.4131),
            0.766935,
        ),
        (
            _CENTRAL_WRAPPER,
            BoundingBox(296, 0, 371, 26.4131),
            0.58446,
        ),
        (
            _CENTRAL_SUFFIX,
            BoundingBox(380, 0, 484, 26.4131),
            0.766935,
        ),
        *central_paired_wrapped_two_words()[2:],
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedCentralPairedWrappedTwoRecognizer(prefix_confidence=0.9991),
        ConfirmedCentralPairedWrappedTwoRecognizer(middle_confidence=0.8389),
        ConfirmedCentralPairedWrappedTwoRecognizer(closing_confidence=0.9509),
        ConfirmedCentralPairedWrappedTwoRecognizer(suffix_confidence=0.9988),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            prefix_variant_text=_CENTRAL_PREFIX[:-1] + chr(0xB9C8)
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            target_variant_confidence=0.9994
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            wrapper_variant_text=chr(0x201C) + _CENTRAL_TARGET + chr(0x2019)
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            wrapper_variant_confidence=0.5839
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            suffix_variant_confidence=0.9709
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            segments=((0, 107), (114, 176), (175, 190), (199, 303))
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            segments=((0, 107), (115, 176), (174, 190), (199, 303))
        ),
        ConfirmedCentralPairedWrappedTwoRecognizer(
            segments=((0, 107), (115, 176), (175, 190), (198, 303))
        ),
    ],
)
def test_confirmed_central_paired_wrapped_two_requires_crop_evidence(
    recognizer,
) -> None:
    words = central_paired_wrapped_two_words()

    assert (
        _recover_confirmed_central_paired_wrapped_two_split(
            words,
            Image.new("RGB", (704, 28)),
            BoundingBox(0, 0, 704, 26.4131),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        central_paired_wrapped_two_words(first_text=_CENTRAL_PREFIX),
        central_paired_wrapped_two_words(
            candidate_text=(
                _CENTRAL_PREFIX
                + chr(0x201C)
                + _CENTRAL_TARGET
                + chr(0x201D)
                + _CENTRAL_SUFFIX
            )
        ),
        central_paired_wrapped_two_words(candidate_confidence=0.7668),
        central_paired_wrapped_two_words(
            candidate_box=BoundingBox(181, 0, 485, 26.4131)
        ),
        central_paired_wrapped_two_words(following_text=chr(0xAC00)),
        central_paired_wrapped_two_words(following_confidence=0.9996),
        central_paired_wrapped_two_words(trailing_text=_CENTRAL_TRAILING[:-1]),
        central_paired_wrapped_two_words(trailing_confidence=0.9996),
        central_paired_wrapped_two_words()[:3],
    ],
)
def test_confirmed_central_paired_wrapped_two_requires_word_profile(
    words,
) -> None:
    assert (
        _recover_confirmed_central_paired_wrapped_two_split(
            words,
            Image.new("RGB", (704, 28)),
            BoundingBox(0, 0, 704, 26.4131),
            ConfirmedCentralPairedWrappedTwoRecognizer(),
        )
        == words
    )


def test_engine_recovers_central_paired_wrapped_two_segment() -> None:
    engine = PaddleOcrEngine(
        CentralPairedWrappedTwoDetector(),
        CentralPairedWrappedTwoRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (900, 350)))

    assert [word.text for word in document.lines[0].eojeols] == [
        _CENTRAL_FIRST,
        _CENTRAL_PREFIX,
        _CENTRAL_TARGET,
        _CENTRAL_SUFFIX,
        _CENTRAL_FOLLOWING,
        _CENTRAL_TRAILING,
    ]
    assert document.lines[0].eojeols[2].box == BoundingBox(
        396.27,
        236.7391,
        433.77,
        263.1522,
    )


class ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer:
    def __init__(
        self,
        *,
        direct_text: str = "/\uac00\ub098-\ub77c\ub9c8?",
        direct_confidence: float = 0.5355,
        opening_text: str = "/",
        opening_confidence: float = 0.4215,
        middle_text: str = "\uac00\ub098.",
        middle_confidence: float = 0.9235,
        suffix_text: str = "\ub77c\ub9c8?",
        suffix_confidence: float = 0.9938,
        target_variant_text: str = "\uac00\ub098",
        target_variant_confidence: float = 0.9995,
        wrapper_variant_text: str = "\u2014\uac00\ub098\u2014",
        wrapper_variant_confidence: float = 0.6218,
        suffix_variant_text: str = "\ub77c\ub9c8?",
        suffix_variant_confidence: float = 0.9935,
        default_segments: tuple[tuple[int, int], ...] = ((5, 221),),
        segments: tuple[tuple[int, int], ...] = (
            (0, 38),
            (37, 130),
            (141, 216),
        ),
    ) -> None:
        self.values = (
            RecognizedText(direct_text, direct_confidence),
            RecognizedText(opening_text, opening_confidence),
            RecognizedText(middle_text, middle_confidence),
            RecognizedText(suffix_text, suffix_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9996),
            RecognizedText(target_variant_text, 0.9997),
            RecognizedText(target_variant_text, 0.9998),
            RecognizedText(target_variant_text, 0.9999),
            RecognizedText(target_variant_text, 0.9998),
            RecognizedText(target_variant_text, 0.9999),
            RecognizedText(wrapper_variant_text, wrapper_variant_confidence),
            RecognizedText(wrapper_variant_text, 0.4807),
            RecognizedText(wrapper_variant_text, 0.5438),
            RecognizedText(wrapper_variant_text, 0.5048),
            RecognizedText(wrapper_variant_text, 0.3765),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.9927),
            RecognizedText(suffix_variant_text, 0.9926),
            RecognizedText(suffix_variant_text, 0.9908),
            RecognizedText(suffix_variant_text, 0.9938),
        )
        self.default_segments = default_segments
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return self.default_segments
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class IsolatedPairedWrappedTwoPlusTwoRecognizer(
    ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText(
                "/\uac00\ub098\u2014\ub77c\ub9c8?", 0.385632
            ),
            RecognizedText(
                "/\uac00\ub098\u2014\ub77c\ub9c8?", 0.80905
            ),
            *self.values,
        )


class IsolatedPairedWrappedTwoPlusTwoDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(115.08, 251.8043, 345.92, 285.2608),
                0.9,
            ),
        )


def isolated_paired_wrapped_two_plus_two_words(
    *,
    text: str = "/\uac00\ub098\u2014\ub77c\ub9c8?",
    confidence: float = 0.80905,
    box: BoundingBox | None = None,
) -> list[tuple[str, BoundingBox, float]]:
    return [
        (
            text,
            box or BoundingBox(0, 0, 230.84, 33.4565),
            confidence,
        )
    ]


def test_confirmed_isolated_paired_wrapped_two_plus_two_recovers() -> None:
    words = isolated_paired_wrapped_two_plus_two_words()

    recovered = _recover_confirmed_isolated_paired_wrapped_two_plus_two_split(
        words,
        Image.new("RGB", (231, 35)),
        BoundingBox(0, 0, 230.84, 33.4565),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(),
    )

    assert recovered == [
        (
            "\u2014\uac00\ub098\u2014",
            BoundingBox(5, 0, 135, 33.4565),
            0.3765,
        ),
        (
            "\ub77c\ub9c8?",
            BoundingBox(146, 0, 221, 33.4565),
            0.80905,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="A\uac00\ub098-\ub77c\ub9c8?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="/\uac00\ub2e4-\ub77c\ub9c8?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="/\uac00\ub098\u2014\ub77c\ub9c8?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="/\uac00\ub098A\ub77c\ub9c8?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="/\uac00\ub098-\ub77c\uc0ac?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_text="/\uac00\ub098/\ub77c\ub9c8?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            direct_confidence=0.5349
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            opening_text="-"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            opening_confidence=0.4209
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            middle_text="\uac00\ub2e4."
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            middle_text="\uac00\ub098A"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            middle_text="\uac00\ub098\u2014"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            middle_text="\uac00\ub098-"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            middle_confidence=0.9229
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            suffix_text="\ub77c\uc0ac?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            suffix_confidence=0.9929
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            target_variant_text="\uac00\ub2e4"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            target_variant_confidence=0.9993
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            wrapper_variant_text="/\uac00\ub098/"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            wrapper_variant_text="\u201c\uac00\ub098\u201d"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            wrapper_variant_text="\u2014\uac00\ub2e4\u2014"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            wrapper_variant_text="\u2014\uac00\ub098-"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            wrapper_variant_confidence=0.3759
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            suffix_variant_text="\ub77c\uc0ac?"
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            suffix_variant_confidence=0.9899
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            default_segments=((4, 220),)
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            default_segments=((5, 220),)
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            default_segments=((5, 222),)
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            default_segments=((5, 150), (160, 221))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((2, 38), (37, 130), (141, 216))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((0, 38), (35, 130), (141, 216))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((0, 37), (36, 130), (141, 216))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((0, 38), (37, 131), (141, 216))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((0, 38), (37, 130), (140, 216))
        ),
        ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(
            segments=((0, 38), (37, 130), (141, 214))
        ),
    ],
)
def test_confirmed_isolated_paired_wrapped_two_plus_two_requires_crop_evidence(
    recognizer,
) -> None:
    words = isolated_paired_wrapped_two_plus_two_words()

    assert (
        _recover_confirmed_isolated_paired_wrapped_two_plus_two_split(
            words,
            Image.new("RGB", (231, 35)),
            BoundingBox(0, 0, 230.84, 33.4565),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        isolated_paired_wrapped_two_plus_two_words(
            text="\u201c\uac00\ub098\u2014\ub77c\ub9c8?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="A\uac00\ub098\u2014\ub77c\ub9c8?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="/\uac00A\u2014\ub77c\ub9c8?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="/\uac00\ub098/\ub77c\ub9c8?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="/\uac00\ub098A\ub77c\ub9c8?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="/\uac00\ub098\u2014\ub77cA?"
        ),
        isolated_paired_wrapped_two_plus_two_words(
            text="/\uac00\ub098\u2014\ub77c\ub9c8A"
        ),
        isolated_paired_wrapped_two_plus_two_words(confidence=0.8089),
        isolated_paired_wrapped_two_plus_two_words(
            box=BoundingBox(0, 0, 230.4, 33.4565)
        ),
        isolated_paired_wrapped_two_plus_two_words() * 2,
    ],
)
def test_confirmed_isolated_paired_wrapped_two_plus_two_requires_word_profile(
    words,
) -> None:
    assert (
        _recover_confirmed_isolated_paired_wrapped_two_plus_two_split(
            words,
            Image.new("RGB", (231, 35)),
            BoundingBox(0, 0, 230.84, 33.4565),
            ConfirmedIsolatedPairedWrappedTwoPlusTwoRecognizer(),
        )
        == words
    )


def test_engine_recovers_isolated_paired_wrapped_two_plus_two_segment() -> None:
    engine = PaddleOcrEngine(
        IsolatedPairedWrappedTwoPlusTwoDetector(),
        IsolatedPairedWrappedTwoPlusTwoRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (400, 350)))

    assert [word.text for word in document.lines[0].eojeols] == [
        "\uac00\ub098",
        "\ub77c\ub9c8",
    ]
    assert document.lines[0].eojeols[0].box == BoundingBox(
        152.57999999999998,
        251.8043,
        217.57999999999998,
        285.2608,
    )


class ConfirmedPairedWrappedThreePlusThreeRecognizer:
    def __init__(
        self,
        *,
        wrapper_text: str = "\u2014\uac00\ub098\ub2e4\u2014",
        wrapper_confidence: float = 0.8285,
        artifact_text: str = "A",
        artifact_confidence: float = 0.2985,
        suffix_text: str = "\ubc14\uc0ac\uc544?",
        suffix_confidence: float = 0.5315,
        target_variant_text: str = "\uac00\ub098\ub2e4",
        target_variant_confidence: float = 0.9996,
        wrapper_variant_text: str = "\u2014\uac00\ub098\ub2e4\u2014",
        wrapper_variant_confidence: float = 0.7935,
        suffix_variant_text: str = "\ubc14\uc0ac\uc544?",
        suffix_variant_confidence: float = 0.9895,
        segments: tuple[tuple[int, int], ...] = (
            (0, 242),
            (241, 272),
            (290, 470),
        ),
    ) -> None:
        self.values = (
            RecognizedText(wrapper_text, wrapper_confidence),
            RecognizedText(artifact_text, artifact_confidence),
            RecognizedText(suffix_text, suffix_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9997),
            RecognizedText(target_variant_text, 0.9998),
            RecognizedText(target_variant_text, 0.9996),
            RecognizedText(target_variant_text, 0.9997),
            RecognizedText(wrapper_variant_text, wrapper_variant_confidence),
            RecognizedText(wrapper_variant_text, 0.8152),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.9915),
            RecognizedText(suffix_variant_text, 0.9931),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SegmentedPairedWrappedThreePlusThreeRecognizer(
    ConfirmedPairedWrappedThreePlusThreeRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("\uac00\ub098\ub2e4", 0.999952),
            RecognizedText("\ub77c\ub9c8", 0.999994),
            RecognizedText(
                "/\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544?", 0.680876
            ),
            RecognizedText(
                "/\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544?", 0.646285
            ),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((41, 208), (226, 336), (354, 824))
        return super().word_boxes(_image, space_threshold)


class PairedWrappedThreePlusThreeDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(10, 5, 850, 63.1087),
                0.9,
            ),
        )


_PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES = (
    BoundingBox(41, 0, 208, 58.1087),
    BoundingBox(226, 0, 336, 58.1087),
    BoundingBox(354, 0, 824, 58.1087),
)


def paired_wrapped_three_plus_three_words(
    *,
    text: str = "/\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544?",
    confidence: float = 0.680876,
    boxes: tuple[BoundingBox, ...] = _PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES,
    first_text: str = "\uac00\ub098\ub2e4",
    first_confidence: float = 0.999952,
    second_text: str = "\ub77c\ub9c8",
    second_confidence: float = 0.999994,
) -> list[tuple[str, BoundingBox, float]]:
    return [
        (first_text, boxes[0], first_confidence),
        (second_text, boxes[1], second_confidence),
        (text, boxes[2], confidence),
    ]


def test_confirmed_paired_wrapped_three_plus_three_recovers() -> None:
    words = paired_wrapped_three_plus_three_words()

    recovered = _recover_confirmed_paired_wrapped_three_plus_three_split(
        words,
        Image.new("RGB", (824, 59)),
        BoundingBox(0, 0, 824, 58.1087),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(),
    )

    assert recovered == [
        *words[:2],
        (
            "\u2014\uac00\ub098\ub2e4\u2014",
            BoundingBox(354, 0, 596, 58.1087),
            0.680876,
        ),
        (
            "\ubc14\uc0ac\uc544?",
            BoundingBox(644, 0, 824, 58.1087),
            0.5315,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_text="/\uac00\ub098\ub2e4\u2014"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_text="\u2014\uac00\ub098\ub77c\u2014"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_text="\u2014\uac00\ub098\ub2e4-"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_text="\u201c\uac00\ub098\ub2e4\u2014"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_confidence=0.8279
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(artifact_text="!"),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(artifact_text="AB"),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            artifact_confidence=0.2979
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            suffix_text="\ubc14\uc0ac\uc790?"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            suffix_confidence=0.5309
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            target_variant_text="\uac00\ub098\ub77c"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            target_variant_confidence=0.9994
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_variant_text="\u2014\uac00\ub098\ub77c\u2014"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            wrapper_variant_confidence=0.7929
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            suffix_variant_text="\ubc14\uc0ac\uc790?"
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            suffix_variant_confidence=0.9889
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((2, 242), (241, 272), (290, 470))
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((0, 242), (240, 272), (290, 470))
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((0, 241), (240, 271), (290, 470))
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((0, 242), (241, 274), (290, 470))
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((0, 242), (241, 272), (288, 470))
        ),
        ConfirmedPairedWrappedThreePlusThreeRecognizer(
            segments=((0, 242), (241, 272), (290, 468))
        ),
    ],
)
def test_confirmed_paired_wrapped_three_plus_three_requires_crop_evidence(
    recognizer,
) -> None:
    words = paired_wrapped_three_plus_three_words()

    assert (
        _recover_confirmed_paired_wrapped_three_plus_three_split(
            words,
            Image.new("RGB", (824, 59)),
            BoundingBox(0, 0, 824, 58.1087),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        paired_wrapped_three_plus_three_words(
            first_text="\uac00\ub098"
        ),
        paired_wrapped_three_plus_three_words(
            first_text="\uac00A\ub2e4"
        ),
        paired_wrapped_three_plus_three_words(first_confidence=0.9998),
        paired_wrapped_three_plus_three_words(second_text="\ub77c"),
        paired_wrapped_three_plus_three_words(second_text="\ub77cA"),
        paired_wrapped_three_plus_three_words(second_confidence=0.99998),
        paired_wrapped_three_plus_three_words(
            text="\u201c\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="A\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="/\uac00A\ub2e4\u2014\ubc14\uc0ac\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="/\uac00\ub098\ub2e4/\ubc14\uc0ac\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="/\uac00\ub098\ub2e4A\ubc14\uc0ac\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="/\uac00\ub098\ub2e4\u2014\ubc14A\uc544?"
        ),
        paired_wrapped_three_plus_three_words(
            text="/\uac00\ub098\ub2e4\u2014\ubc14\uc0ac\uc544A"
        ),
        paired_wrapped_three_plus_three_words(confidence=0.6807),
        paired_wrapped_three_plus_three_words(
            boxes=(
                BoundingBox(41, 0, 207, 58.1087),
                *_PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[1:],
            )
        ),
        paired_wrapped_three_plus_three_words(
            boxes=(
                _PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[0],
                BoundingBox(226, 0, 335, 58.1087),
                _PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[2],
            )
        ),
        paired_wrapped_three_plus_three_words(
            boxes=(
                _PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[0],
                BoundingBox(225, 0, 335, 58.1087),
                _PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[2],
            )
        ),
        paired_wrapped_three_plus_three_words(
            boxes=(
                *_PAIRED_WRAPPED_THREE_PLUS_THREE_BOXES[:2],
                BoundingBox(353, 0, 824, 58.1087),
            )
        ),
        paired_wrapped_three_plus_three_words()[:2],
    ],
)
def test_confirmed_paired_wrapped_three_plus_three_requires_word_profile(
    words,
) -> None:
    assert (
        _recover_confirmed_paired_wrapped_three_plus_three_split(
            words,
            Image.new("RGB", (824, 59)),
            BoundingBox(0, 0, 824, 58.1087),
            ConfirmedPairedWrappedThreePlusThreeRecognizer(),
        )
        == words
    )


def test_engine_recovers_paired_wrapped_three_plus_three_segment() -> None:
    engine = PaddleOcrEngine(
        PairedWrappedThreePlusThreeDetector(),
        SegmentedPairedWrappedThreePlusThreeRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (870, 80)))

    assert [word.text for word in document.lines[0].eojeols] == [
        "\uac00\ub098\ub2e4",
        "\ub77c\ub9c8",
        "\uac00\ub098\ub2e4",
        "\ubc14\uc0ac\uc544",
    ]
    assert document.lines[0].eojeols[2].box == BoundingBox(
        412.4,
        5,
        557.6,
        63.1087,
    )


class ConfirmedPairedWrappedFourPlusTwoRecognizer:
    def __init__(
        self,
        *,
        direct_text: str = "/\uac00\ub098\ub2e4\ub77c/\ub9c8\ubc14",
        direct_confidence: float = 0.5405,
        wrapper_text: str = "/\uac00\ub098\ub2e4\ub77c/",
        wrapper_confidence: float = 0.5185,
        suffix_text: str = "\ub9c8\ubc14",
        suffix_confidence: float = 0.9998,
        target_variant_text: str = "\uac00\ub098\ub2e4\ub77c",
        target_variant_confidence: float = 0.9989,
        wrapper_variant_text: str = "/\uac00\ub098\ub2e4\ub77c/",
        wrapper_variant_confidence: float = 0.5268,
        suffix_variant_text: str = "\ub9c8\ubc14",
        suffix_variant_confidence: float = 0.9955,
        segments: tuple[tuple[int, int], ...] = ((0, 146), (155, 201)),
    ) -> None:
        self.values = (
            RecognizedText(direct_text, direct_confidence),
            RecognizedText(wrapper_text, wrapper_confidence),
            RecognizedText(suffix_text, suffix_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9991),
            RecognizedText(target_variant_text, 0.9994),
            RecognizedText(target_variant_text, 0.9996),
            RecognizedText(target_variant_text, 0.9993),
            RecognizedText(wrapper_variant_text, wrapper_variant_confidence),
            RecognizedText(wrapper_variant_text, 0.7453),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.9994),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SegmentedPairedWrappedFourPlusTwoRecognizer(
    ConfirmedPairedWrappedFourPlusTwoRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("\uac00\ub098\ub2e4", 0.9998),
            RecognizedText("\ub77c\ub9c8", 0.9996),
            RecognizedText("\ubc14\uc0ac\uc544\uc790\ucc28", 0.9989),
            RecognizedText("\uce74\ud0c0\ud30c\ud558\uac70", 0.9987),
            RecognizedText("/\uac00\ub098\ub2e4\ub77c/\ub9c8\ubc14", 0.5405),
            RecognizedText("\u2014\uac00\ub098\ub2e4\ub77c-\ub9c8\ubc14", 0.5476),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((41, 112), (122, 169), (177, 300), (309, 429), (439, 640))
        return super().word_boxes(_image, space_threshold)


class PairedWrappedFourPlusTwoDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(10, 5, 660, 33.174),
                0.9,
            ),
        )


_PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES = (
    BoundingBox(41, 0, 112, 28.174),
    BoundingBox(122, 0, 169, 28.174),
    BoundingBox(177, 0, 300, 28.174),
    BoundingBox(309, 0, 429, 28.174),
    BoundingBox(439, 0, 640, 28.174),
)


def paired_wrapped_four_plus_two_words(
    *,
    text: str = "/\uac00\ub098\ub2e4\ub77c\u2014\ub9c8\ubc14",
    confidence: float = 0.5476,
    boxes: tuple[BoundingBox, ...] = _PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES,
    confidences: tuple[float, ...] = (0.9998, 0.9996, 0.9989, 0.9987),
) -> list[tuple[str, BoundingBox, float]]:
    return [
        ("\uac00\ub098\ub2e4", boxes[0], confidences[0]),
        ("\ub77c\ub9c8", boxes[1], confidences[1]),
        ("\ubc14\uc0ac\uc544\uc790\ucc28", boxes[2], confidences[2]),
        ("\uce74\ud0c0\ud30c\ud558\uac70", boxes[3], confidences[3]),
        (text, boxes[4], confidence),
    ]


def test_confirmed_paired_wrapped_four_plus_two_recovers() -> None:
    words = paired_wrapped_four_plus_two_words()
    recovered = _recover_confirmed_paired_wrapped_four_plus_two_split(
        words,
        Image.new("RGB", (650, 29)),
        BoundingBox(0, 0, 650, 28.174),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(),
    )
    assert recovered == [
        *words[:4],
        (
            "/\uac00\ub098\ub2e4\ub77c\u2014",
            BoundingBox(439, 0, 585, 28.174),
            0.5268,
        ),
        ("\ub9c8\ubc14", BoundingBox(594, 0, 640, 28.174), 0.5476),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            direct_text="/\uac00\ub098\ub2e4\ub9c8/\ub9c8\ubc14"
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(direct_confidence=0.5399),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            wrapper_text="/\uac00\ub098\ub2e4\ub9c8/"
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(wrapper_confidence=0.5179),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(suffix_text="\ub9c8\uc0ac"),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(suffix_confidence=0.9996),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            target_variant_text="\uac00\ub098\ub2e4\ub9c8"
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            target_variant_confidence=0.9987
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            wrapper_variant_text="/\uac00\ub098\ub2e4\ub9c8/"
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            wrapper_variant_confidence=0.5259
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            suffix_variant_text="\ub9c8\uc0ac"
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            suffix_variant_confidence=0.9949
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            segments=((2, 146), (155, 201))
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            segments=((0, 146), (154, 201))
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            segments=((0, 146), (155, 199))
        ),
        ConfirmedPairedWrappedFourPlusTwoRecognizer(
            segments=((0, 144), (153, 201))
        ),
    ],
)
def test_confirmed_paired_wrapped_four_plus_two_requires_crop_evidence(
    recognizer,
) -> None:
    words = paired_wrapped_four_plus_two_words()
    assert (
        _recover_confirmed_paired_wrapped_four_plus_two_split(
            words,
            Image.new("RGB", (650, 29)),
            BoundingBox(0, 0, 650, 28.174),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        paired_wrapped_four_plus_two_words(
            text="A\uac00\ub098\ub2e4\ub77c\u2014\ub9c8\ubc14"
        ),
        paired_wrapped_four_plus_two_words(
            text="/\uac00\ub098A\ub77c\u2014\ub9c8\ubc14"
        ),
        paired_wrapped_four_plus_two_words(
            text="/\uac00\ub098\ub2e4\ub77c/\ub9c8\ubc14"
        ),
        paired_wrapped_four_plus_two_words(
            text="/\uac00\ub098\ub2e4\ub77cA\ub9c8\ubc14"
        ),
        paired_wrapped_four_plus_two_words(confidence=0.5469),
        paired_wrapped_four_plus_two_words(
            confidences=(0.9996, 0.9996, 0.9989, 0.9987)
        ),
        paired_wrapped_four_plus_two_words(
            confidences=(0.9998, 0.9994, 0.9989, 0.9987)
        ),
        paired_wrapped_four_plus_two_words(
            confidences=(0.9998, 0.9996, 0.9987, 0.9987)
        ),
        paired_wrapped_four_plus_two_words(
            confidences=(0.9998, 0.9996, 0.9989, 0.9985)
        ),
        paired_wrapped_four_plus_two_words(
            boxes=(
                BoundingBox(41, 0, 111, 28.174),
                *_PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES[1:],
            )
        ),
        paired_wrapped_four_plus_two_words(
            boxes=(
                _PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES[0],
                BoundingBox(121, 0, 168, 28.174),
                *_PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES[2:],
            )
        ),
        paired_wrapped_four_plus_two_words(
            boxes=(
                *_PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES[:3],
                BoundingBox(309, 0, 428, 28.174),
                _PAIRED_WRAPPED_FOUR_PLUS_TWO_BOXES[4],
            )
        ),
        paired_wrapped_four_plus_two_words()[:4],
    ],
)
def test_confirmed_paired_wrapped_four_plus_two_requires_word_profile(
    words,
) -> None:
    assert (
        _recover_confirmed_paired_wrapped_four_plus_two_split(
            words,
            Image.new("RGB", (650, 29)),
            BoundingBox(0, 0, 650, 28.174),
            ConfirmedPairedWrappedFourPlusTwoRecognizer(),
        )
        == words
    )


def test_engine_recovers_paired_wrapped_four_plus_two_segment() -> None:
    engine = PaddleOcrEngine(
        PairedWrappedFourPlusTwoDetector(),
        SegmentedPairedWrappedFourPlusTwoRecognizer(),
    )
    document = engine.recognize(Image.new("RGB", (680, 50)))
    assert [word.text for word in document.lines[0].eojeols] == [
        "\uac00\ub098\ub2e4",
        "\ub77c\ub9c8",
        "\ubc14\uc0ac\uc544\uc790\ucc28",
        "\uce74\ud0c0\ud30c\ud558\uac70",
        "\uac00\ub098\ub2e4\ub77c",
        "\ub9c8\ubc14",
    ]
    assert document.lines[0].eojeols[4].box == BoundingBox(
        473.3333333333333,
        5,
        570.6666666666666,
        33.174,
    )


class ConfirmedMismatchedWrappedThreePlusOneRecognizer:
    def __init__(
        self,
        *,
        opening_text: str = "\u201c\u201d",
        opening_confidence: float = 0.2715,
        target_text: str = "\uac00\ub098\ub2e4",
        target_confidence: float = 0.9996,
        punctuation_text: str = "\u2019",
        punctuation_confidence: float = 0.5115,
        suffix_text: str = "\ub77c",
        suffix_confidence: float = 0.9989,
        target_variant_text: str = "\uac00\ub098\ub2e4",
        target_variant_confidence: float = 0.9989,
        wrapper_variant_text: str = "\u2018\uac00\ub098\ub2e4\u2019",
        wrapper_variant_confidence: float = 0.5468,
        suffix_variant_text: str = "\ub77c",
        suffix_variant_confidence: float = 0.9989,
        segments: tuple[tuple[int, int], ...] = (
            (0, 58),
            (57, 255),
            (269, 305),
            (328, 393),
        ),
    ) -> None:
        self.values = (
            RecognizedText(opening_text, opening_confidence),
            RecognizedText(target_text, target_confidence),
            RecognizedText(punctuation_text, punctuation_confidence),
            RecognizedText(suffix_text, suffix_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9993),
            RecognizedText(wrapper_variant_text, wrapper_variant_confidence),
            RecognizedText(wrapper_variant_text, 0.6044),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.9992),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SegmentedMismatchedWrappedThreePlusOneRecognizer(
    ConfirmedMismatchedWrappedThreePlusOneRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("\u201c\uac00\ub098\ub2e4\u2019\ub77c", 0.5821),
            RecognizedText("\u201c\uac00\ub098\ub2e4\u2019\ub77c", 0.5941),
            RecognizedText("\ub9c8\ubc14\uc0ac", 0.99995),
            RecognizedText("\uc544\uc790\ucc28", 0.9998),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((20, 413), (442, 657), (681, 893))
        return super().word_boxes(_image, space_threshold)


class MismatchedWrappedThreePlusOneDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(10, 5, 910, 80.72),
                0.9,
            ),
        )


def mismatched_wrapped_three_plus_one_words(
    *,
    text: str = "\u201c\uac00\ub098\ub2e4\u2019\ub77c",
    confidence: float = 0.5941,
    height: float = 75.72,
    following_box: BoundingBox | None = None,
    following_confidence: float = 0.99995,
    trailing_box: BoundingBox | None = None,
    trailing_confidence: float = 0.9998,
) -> list[tuple[str, BoundingBox, float]]:
    following_box = following_box or BoundingBox(442, 0, 657, 75.72)
    trailing_box = trailing_box or BoundingBox(681, 0, 893, 75.72)
    return [
        (text, BoundingBox(20, 0, 413, height), confidence),
        ("\ub9c8\ubc14\uc0ac", following_box, following_confidence),
        ("\uc544\uc790\ucc28", trailing_box, trailing_confidence),
    ]


def test_confirmed_mismatched_wrapped_three_plus_one_recovers() -> None:
    words = mismatched_wrapped_three_plus_one_words()

    recovered = _recover_confirmed_mismatched_wrapped_three_plus_one_split(
        words,
        Image.new("RGB", (900, 76)),
        BoundingBox(0, 0, 900, 75.72),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(),
    )

    assert recovered == [
        (
            "\u201c\uac00\ub098\ub2e4\u2019",
            BoundingBox(20, 0, 325, 75.72),
            0.5468,
        ),
        ("\ub77c", BoundingBox(348, 0, 413, 75.72), 0.5941),
        words[1],
        words[2],
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            opening_text="\u201cA"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            opening_confidence=0.2709
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            target_text="\uac00\ub098\ub77c"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            target_confidence=0.9994
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            punctuation_text="."
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            punctuation_confidence=0.5109
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            suffix_text="\ub9c8"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            suffix_confidence=0.9987
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            target_variant_text="\uac00\ub098\ub77c"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            target_variant_confidence=0.9987
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            wrapper_variant_text="\u2018\uac00\ub098\ub77c\u2019"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            wrapper_variant_confidence=0.539
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            suffix_variant_text="\ub9c8"
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            suffix_variant_confidence=0.9987
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            segments=((2, 58), (57, 255), (269, 305), (328, 393))
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            segments=((0, 58), (56, 255), (269, 305), (328, 393))
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            segments=((0, 58), (57, 255), (268, 305), (328, 393))
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            segments=((0, 58), (57, 255), (269, 305), (327, 393))
        ),
        ConfirmedMismatchedWrappedThreePlusOneRecognizer(
            segments=((0, 58), (57, 255), (269, 305), (328, 391))
        ),
    ],
)
def test_confirmed_mismatched_wrapped_three_plus_one_requires_crop_evidence(
    recognizer,
) -> None:
    words = mismatched_wrapped_three_plus_one_words()

    assert (
        _recover_confirmed_mismatched_wrapped_three_plus_one_split(
            words,
            Image.new("RGB", (900, 76)),
            BoundingBox(0, 0, 900, 75.72),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        mismatched_wrapped_three_plus_one_words(
            text="A\uac00\ub098\ub2e4\u2019\ub77c"
        ),
        mismatched_wrapped_three_plus_one_words(
            text="\u201c\uac00A\ub2e4\u2019\ub77c"
        ),
        mismatched_wrapped_three_plus_one_words(
            text="\u201c\uac00\ub098\ub2e4A\ub77c"
        ),
        mismatched_wrapped_three_plus_one_words(
            text="\u201c\uac00\ub098\ub2e4\u201d\ub77c"
        ),
        mismatched_wrapped_three_plus_one_words(confidence=0.5939),
        mismatched_wrapped_three_plus_one_words(
            following_confidence=0.9998
        ),
        mismatched_wrapped_three_plus_one_words(
            following_box=BoundingBox(441, 0, 656, 75.72)
        ),
        mismatched_wrapped_three_plus_one_words(
            following_box=BoundingBox(442, 0, 656, 75.72)
        ),
        mismatched_wrapped_three_plus_one_words(
            trailing_confidence=0.9996
        ),
        mismatched_wrapped_three_plus_one_words(
            trailing_box=BoundingBox(680, 0, 892, 75.72)
        ),
        mismatched_wrapped_three_plus_one_words(
            trailing_box=BoundingBox(681, 0, 892, 75.72)
        ),
        mismatched_wrapped_three_plus_one_words()[:2],
    ],
)
def test_confirmed_mismatched_wrapped_three_plus_one_requires_word_profile(
    words,
) -> None:
    assert (
        _recover_confirmed_mismatched_wrapped_three_plus_one_split(
            words,
            Image.new("RGB", (900, 76)),
            BoundingBox(0, 0, 900, 75.72),
            ConfirmedMismatchedWrappedThreePlusOneRecognizer(),
        )
        == words
    )


def test_engine_recovers_mismatched_wrapped_three_plus_one_segment() -> None:
    engine = PaddleOcrEngine(
        MismatchedWrappedThreePlusOneDetector(),
        SegmentedMismatchedWrappedThreePlusOneRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (930, 100)))
    target = document.lines[0].eojeols[0]

    assert [word.text for word in document.lines[0].eojeols] == [
        "\uac00\ub098\ub2e4",
        "\ub77c",
        "\ub9c8\ubc14\uc0ac",
        "\uc544\uc790\ucc28",
    ]
    assert target.box == BoundingBox(91, 5, 274, 80.72)


class ConfirmedWrappedFivePlusFourRecognizer:
    def __init__(
        self,
        *,
        first_ctc_text: str = "-가나다라마-",
        first_ctc_confidence: float = 0.6445,
        middle_ctc_text: str = "",
        middle_ctc_confidence: float = 0.0,
        suffix_ctc_text: str = "바사아자",
        suffix_ctc_confidence: float = 0.9994,
        wrapper_variant_text: str = "-가나다라마—",
        wrapper_variant_confidence: float = 0.56,
        target_variant_text: str = "가나다라마",
        target_variant_confidence: float = 0.9997,
        suffix_variant_text: str = "바사아자",
        suffix_variant_confidence: float = 0.9988,
        segments: tuple[tuple[int, int], ...] = (
            (0, 347),
            (346, 384),
            (402, 620),
        ),
    ) -> None:
        self.values = (
            RecognizedText(first_ctc_text, first_ctc_confidence),
            RecognizedText(middle_ctc_text, middle_ctc_confidence),
            RecognizedText(suffix_ctc_text, suffix_ctc_confidence),
            RecognizedText(wrapper_variant_text, wrapper_variant_confidence),
            RecognizedText(wrapper_variant_text, 0.66),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9998),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.9995),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SegmentedWrappedFivePlusFourRecognizer(ConfirmedWrappedFivePlusFourRecognizer):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("—가나다라마-바사아자", 0.8675),
            RecognizedText("c", 0.139),
            RecognizedText("c", 0.139),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((35, 655), (680, 703))
        return super().word_boxes(_image, space_threshold)


class WrappedFivePlusFourDetector:
    def detect(self, _image):
        height = 620 / 9.785
        return (DetectedRegion(BoundingBox(10.6, 5, 712.4, 5 + height), 0.9),)


def test_engine_recovers_wrapped_five_plus_four_segmented_line() -> None:
    engine = PaddleOcrEngine(
        WrappedFivePlusFourDetector(),
        SegmentedWrappedFivePlusFourRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (730, 100)))

    assert document.lines[0].text == "—가나다라마- 바사아자"
    assert [word.text for word in document.lines[0].eojeols] == [
        "가나다라마",
        "바사아자",
    ]


def test_confirmed_wrapped_five_plus_four_recovers() -> None:
    height = 620 / 9.785
    words = [
        (
            "—가나다라마-바사아자",
            BoundingBox(20, 0, 640, height),
            0.8675,
        )
    ]

    assert _recover_confirmed_wrapped_five_plus_four_split(
        words,
        Image.new("RGB", (660, 64)),
        BoundingBox(0, 0, 660, height),
        ConfirmedWrappedFivePlusFourRecognizer(),
    ) == [
        (
            "—가나다라마-",
            BoundingBox(20, 0, 404, height),
            0.56,
        ),
        (
            "바사아자",
            BoundingBox(422, 0, 640, height),
            0.8675,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedWrappedFivePlusFourRecognizer(first_ctc_text="-가나다라자-"),
        ConfirmedWrappedFivePlusFourRecognizer(first_ctc_confidence=0.6439),
        ConfirmedWrappedFivePlusFourRecognizer(middle_ctc_text="-"),
        ConfirmedWrappedFivePlusFourRecognizer(middle_ctc_confidence=0.001),
        ConfirmedWrappedFivePlusFourRecognizer(suffix_ctc_text="바사아차"),
        ConfirmedWrappedFivePlusFourRecognizer(suffix_ctc_confidence=0.9992),
        ConfirmedWrappedFivePlusFourRecognizer(wrapper_variant_text="-가나다라자-"),
        ConfirmedWrappedFivePlusFourRecognizer(wrapper_variant_confidence=0.549),
        ConfirmedWrappedFivePlusFourRecognizer(target_variant_text="가나다라자"),
        ConfirmedWrappedFivePlusFourRecognizer(target_variant_confidence=0.9995),
        ConfirmedWrappedFivePlusFourRecognizer(suffix_variant_text="바사아차"),
        ConfirmedWrappedFivePlusFourRecognizer(suffix_variant_confidence=0.9986),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((2, 347), (346, 384), (402, 620))
        ),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((0, 347), (345, 384), (402, 620))
        ),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((0, 347), (346, 384), (401, 620))
        ),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((0, 347), (346, 384), (403, 620))
        ),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((0, 347), (346, 374), (392, 620))
        ),
        ConfirmedWrappedFivePlusFourRecognizer(
            segments=((0, 347), (346, 384), (402, 618))
        ),
    ],
)
def test_confirmed_wrapped_five_plus_four_requires_profile(recognizer) -> None:
    height = 620 / 9.785
    words = [
        (
            "—가나다라마-바사아자",
            BoundingBox(20, 0, 640, height),
            0.8675,
        )
    ]

    assert (
        _recover_confirmed_wrapped_five_plus_four_split(
            words,
            Image.new("RGB", (660, 64)),
            BoundingBox(0, 0, 660, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("text", "confidence", "height"),
    [
        ("A가나다라마-바사아자", 0.8675, 620 / 9.785),
        ("—가나다라A-바사아자", 0.8675, 620 / 9.785),
        ("—가나다라마A바사아자", 0.8675, 620 / 9.785),
        ("—가나다라마-바사아A", 0.8675, 620 / 9.785),
        ("—가나다라마-바사아자", 0.8669, 620 / 9.785),
        ("—가나다라마-바사아자", 0.8681, 620 / 9.785),
        ("—가나다라마-바사아자", 0.8675, 63.5),
    ],
)
def test_confirmed_wrapped_five_plus_four_requires_word_shape(
    text,
    confidence,
    height,
) -> None:
    words = [(text, BoundingBox(20, 0, 640, height), confidence)]

    assert (
        _recover_confirmed_wrapped_five_plus_four_split(
            words,
            Image.new("RGB", (660, 64)),
            BoundingBox(0, 0, 660, height),
            ConfirmedWrappedFivePlusFourRecognizer(),
        )
        == words
    )

class ConfirmedPunctuatedThreePlusThreeRecognizer:
    def __init__(
        self,
        *,
        first_ctc_text: str = "가나다",
        first_ctc_confidence: float = 0.99985,
        second_ctc_text: str = "“라마바”",
        second_ctc_confidence: float = 0.272,
        prefix_variant_text: str = "가나다",
        prefix_variant_confidence: float = 0.99966,
        target_variant_text: str = "라마바",
        target_variant_confidence: float = 0.9987,
        segments: tuple[tuple[int, int], ...] = ((20, 185), (203, 475)),
    ) -> None:
        self.values = (
            RecognizedText(first_ctc_text, first_ctc_confidence),
            RecognizedText(second_ctc_text, second_ctc_confidence),
            RecognizedText(prefix_variant_text, prefix_variant_confidence),
            RecognizedText(prefix_variant_text, 0.99962),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.9998),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.005
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class SingleSegmentPunctuatedThreePlusThreeRecognizer(
    ConfirmedPunctuatedThreePlusThreeRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText("가나다-라마바—", 0.6525),
            RecognizedText("가나다-라마바—", 0.618),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((20, 475),)
        return super().word_boxes(_image, space_threshold)


class PunctuatedThreePlusThreeDetector:
    def detect(self, _image):
        height = 496.48 / 9.095
        return (DetectedRegion(BoundingBox(10, 5, 506.48, 5 + height), 0.9),)


def test_engine_recovers_punctuated_three_plus_three_single_segment_line() -> None:
    engine = PaddleOcrEngine(
        PunctuatedThreePlusThreeDetector(),
        SingleSegmentPunctuatedThreePlusThreeRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (520, 100)))

    assert document.lines[0].text == "가나다 -라마바—"
    assert [word.text for word in document.lines[0].eojeols] == ["가나다", "라마바"]


def test_confirmed_punctuated_three_plus_three_recovers() -> None:
    height = 496.48 / 9.095
    words = [
        (
            "가나다-라마바—",
            BoundingBox(0, 0, 496.48, height),
            0.6525,
        )
    ]

    assert _recover_confirmed_punctuated_three_plus_three_split(
        words,
        Image.new("RGB", (498, 56)),
        BoundingBox(0, 0, 496.48, height),
        ConfirmedPunctuatedThreePlusThreeRecognizer(),
    ) == [
        (
            "가나다",
            BoundingBox(20, 0, 185, height),
            0.6525,
        ),
        (
            "-라마바—",
            BoundingBox(203, 0, 475, height),
            0.272,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedPunctuatedThreePlusThreeRecognizer(first_ctc_text="가나자"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(first_ctc_confidence=0.9997),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_text="“라마자”"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_text="라마바"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_text="A라마바”"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_text="“라마바A"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_confidence=0.27),
        ConfirmedPunctuatedThreePlusThreeRecognizer(second_ctc_confidence=0.274),
        ConfirmedPunctuatedThreePlusThreeRecognizer(prefix_variant_text="가나자"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            prefix_variant_confidence=0.9995
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(target_variant_text="라마자"),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            target_variant_confidence=0.9985
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            segments=((19, 185), (203, 475))
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            segments=((20, 185), (202, 475))
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            segments=((20, 185), (204, 475))
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            segments=((20, 175), (193, 475))
        ),
        ConfirmedPunctuatedThreePlusThreeRecognizer(
            segments=((20, 185), (203, 474))
        ),
    ],
)
def test_confirmed_punctuated_three_plus_three_requires_profile(recognizer) -> None:
    height = 496.48 / 9.095
    words = [
        (
            "가나다-라마바—",
            BoundingBox(0, 0, 496.48, height),
            0.6525,
        )
    ]

    assert (
        _recover_confirmed_punctuated_three_plus_three_split(
            words,
            Image.new("RGB", (498, 56)),
            BoundingBox(0, 0, 496.48, height),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("text", "confidence", "height"),
    [
        ("가나A-라마바—", 0.6525, 496.48 / 9.095),
        ("가나다A라마바—", 0.6525, 496.48 / 9.095),
        ("가나다-라마바A", 0.6525, 496.48 / 9.095),
        ("가나다-라마바—", 0.6519, 496.48 / 9.095),
        ("가나다-라마바—", 0.6531, 496.48 / 9.095),
        ("가나다-라마바—", 0.6525, 54.7),
    ],
)
def test_confirmed_punctuated_three_plus_three_requires_word_shape(
    text,
    confidence,
    height,
) -> None:
    words = [(text, BoundingBox(0, 0, 496.48, height), confidence)]

    assert (
        _recover_confirmed_punctuated_three_plus_three_split(
            words,
            Image.new("RGB", (498, 56)),
            BoundingBox(0, 0, 496.48, height),
            ConfirmedPunctuatedThreePlusThreeRecognizer(),
        )
        == words
    )


class ConfirmedPunctuatedThreePlusThreePlusOneRecognizer:
    def __init__(
        self,
        *,
        first_ctc_text: str = "\uac00\ub098\ub2e4",
        first_ctc_confidence: float = 0.99955,
        second_ctc_text: str = ";\ub77c\ub9c8\ubc14",
        second_ctc_confidence: float = 0.973,
        third_ctc_text: str = "?",
        third_ctc_confidence: float = 0.9785,
        fourth_ctc_text: str = "\uc0ac",
        fourth_ctc_confidence: float = 0.99995,
        prefix_variant_text: str = "\uac00\ub098\ub2e4,",
        prefix_variant_confidence: float = 0.89,
        target_variant_text: str = "\ub77c\ub9c8\ubc14",
        target_variant_confidence: float = 0.99995,
        suffix_variant_text: str = "\uc0ac",
        suffix_variant_confidence: float = 0.99995,
        segments: tuple[tuple[int, int], ...] = (
            (0, 80),
            (87, 178),
            (177, 194),
            (202, 225),
        ),
    ) -> None:
        self.values = (
            RecognizedText(first_ctc_text, first_ctc_confidence),
            RecognizedText(second_ctc_text, second_ctc_confidence),
            RecognizedText(third_ctc_text, third_ctc_confidence),
            RecognizedText(fourth_ctc_text, fourth_ctc_confidence),
            RecognizedText(prefix_variant_text, 0.95),
            RecognizedText(prefix_variant_text, prefix_variant_confidence),
            RecognizedText(target_variant_text, target_variant_confidence),
            RecognizedText(target_variant_text, 0.99997),
            RecognizedText(suffix_variant_text, suffix_variant_confidence),
            RecognizedText(suffix_variant_text, 0.99999),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.001
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_punctuated_three_plus_three_plus_one_recovers() -> None:
    height = 225 / 8.52
    words = [
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14.\uc0ac",
            BoundingBox(20, 0, 245, height),
            0.8445,
        )
    ]

    assert _recover_confirmed_punctuated_three_plus_three_plus_one_split(
        words,
        Image.new("RGB", (265, 27)),
        BoundingBox(0, 0, 265, height),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(),
    ) == [
        (
            "\uac00\ub098\ub2e4,",
            BoundingBox(20, 0, 120, height),
            0.8445,
        ),
        (
            "\ub77c\ub9c8\ubc14.",
            BoundingBox(120, 0, 220, height),
            0.8445,
        ),
        (
            "\uc0ac",
            BoundingBox(220, 0, 245, height),
            0.8445,
        ),
    ]

@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            first_ctc_text="\uac00\ub098\ub77c"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            first_ctc_confidence=0.9994
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            second_ctc_text=";\ub77c\uac00\ubc14"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            second_ctc_confidence=0.9719
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            third_ctc_text="A"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            third_ctc_confidence=0.9779
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            fourth_ctc_text="\uc790"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            fourth_ctc_confidence=0.9998
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            prefix_variant_text="\uac00\ub098\ub77c,"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            prefix_variant_confidence=0.879
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            target_variant_text="\ub77c\ub9c8\uc790"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            target_variant_confidence=0.9998
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            suffix_variant_text="\uc790"
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            suffix_variant_confidence=0.9998
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            segments=((0, 80), (86, 178), (177, 194), (202, 225))
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            segments=((0, 80), (87, 177), (177, 194), (202, 225))
        ),
        ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(
            segments=((2, 80), (87, 178), (177, 194), (202, 223))
        ),
    ],
)
def test_confirmed_punctuated_three_plus_three_plus_one_requires_profile(
    recognizer,
) -> None:
    height = 225 / 8.52
    words = [
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14.\uc0ac",
            BoundingBox(20, 0, 245, height),
            0.8445,
        )
    ]

    assert (
        _recover_confirmed_punctuated_three_plus_three_plus_one_split(
            words,
            Image.new("RGB", (265, 27)),
            BoundingBox(0, 0, 265, height),
            recognizer,
        )
        == words
    )

@pytest.mark.parametrize(
    ("text", "confidence", "height"),
    [
        (
            "\uac00\ub098A,\ub77c\ub9c8\ubc14.\uc0ac",
            0.8445,
            225 / 8.52,
        ),
        (
            "\uac00\ub098\ub2e4A\ub77c\ub9c8\ubc14.\uc0ac",
            0.8445,
            225 / 8.52,
        ),
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14A\uc0ac",
            0.8445,
            225 / 8.52,
        ),
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14.\uc0ac",
            0.8439,
            225 / 8.52,
        ),
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14.\uc0ac",
            0.8451,
            225 / 8.52,
        ),
        (
            "\uac00\ub098\ub2e4,\ub77c\ub9c8\ubc14.\uc0ac",
            0.8445,
            26.3,
        ),
    ],
)
def test_confirmed_punctuated_three_plus_three_plus_one_requires_word_shape(
    text,
    confidence,
    height,
) -> None:
    words = [(text, BoundingBox(20, 0, 245, height), confidence)]

    assert (
        _recover_confirmed_punctuated_three_plus_three_plus_one_split(
            words,
            Image.new("RGB", (265, 27)),
            BoundingBox(0, 0, 265, height),
            ConfirmedPunctuatedThreePlusThreePlusOneRecognizer(),
        )
        == words
    )

class ConfirmedTwoPlusTwoRecognizer:
    def __init__(
        self,
        *,
        second_text: str = "\ub2e4\ub77c",
        second_confidence: float = 0.99995,
        segments: tuple[tuple[int, int], ...] = ((0, 165), (164, 330)),
    ) -> None:
        self.values = (
            RecognizedText("\uac00\ub098", 0.99994),
            RecognizedText(second_text, second_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.01
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_two_plus_two_split_recovers_reviewed_profile() -> None:
    height = 77.48
    words = [
        ("\uac00\ub098\ub2e4\ub77c", BoundingBox(20, 0, 350, height), 0.99987),
    ]

    assert _recover_confirmed_two_plus_two_split(
        words,
        Image.new("RGB", (370, 79)),
        BoundingBox(0, 0, 370, height),
        ConfirmedTwoPlusTwoRecognizer(),
    ) == [
        ("\uac00\ub098", BoundingBox(20, 0, 185, height), 0.99987),
        ("\ub2e4\ub77c", BoundingBox(184, 0, 350, height), 0.99987),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedTwoPlusTwoRecognizer(second_text="\ub2e4\ub9c8"),
        ConfirmedTwoPlusTwoRecognizer(second_confidence=0.99989),
        ConfirmedTwoPlusTwoRecognizer(segments=((0, 165), (163, 330))),
        ConfirmedTwoPlusTwoRecognizer(segments=((0, 164), (164, 330))),
        ConfirmedTwoPlusTwoRecognizer(segments=((2, 165), (164, 328))),
    ],
)
def test_confirmed_two_plus_two_split_requires_exact_reviewed_profile(
    recognizer,
) -> None:
    words = [(
        "\uac00\ub098\ub2e4\ub77c",
        BoundingBox(20, 0, 350, 77.48),
        0.99987,
    )]

    assert (
        _recover_confirmed_two_plus_two_split(
            words,
            Image.new("RGB", (370, 79)),
            BoundingBox(0, 0, 370, 77.48),
            recognizer,
        )
        == words
    )


def test_confirmed_two_plus_two_split_requires_word_confidence() -> None:
    words = [(
        "\uac00\ub098\ub2e4\ub77c",
        BoundingBox(20, 0, 350, 77.48),
        0.99979,
    )]

    assert (
        _recover_confirmed_two_plus_two_split(
            words,
            Image.new("RGB", (370, 79)),
            BoundingBox(0, 0, 370, 77.48),
            ConfirmedTwoPlusTwoRecognizer(),
        )
        == words
    )


_RETRY_FIRST = "".join(map(chr, (0xAC00, 0xB098, 0xB2E4, 0xB77C)))
_RETRY_DIRECT = "".join(map(chr, (0xB9C8, 0xBC14, 0xC0AC)))
_RETRY_SELECTED = "".join(map(chr, (0xC544, 0xC790, 0xCC28)))
_RETRY_ONE = chr(0xCE74)
_RETRY_TWO = "".join(map(chr, (0xD0C0, 0xD30C)))
_RETRY_STRUCTURED = "A12-345?"
_RETRY_HEIGHT = 35.21739130434784


class ConfirmedDirectRetryRegressionRecognizer:
    def __init__(
        self,
        *,
        variant_texts: tuple[str, ...] | None = None,
        variant_confidences: tuple[float, ...] = (0.601, 0.627, 0.592, 0.684, 0.569),
    ) -> None:
        texts = variant_texts or (_RETRY_DIRECT,) * 5
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                texts,
                variant_confidences,
                strict=True,
            )
        )
        self.recognition_calls = 0

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class DirectRetryRegressionRecognizer(
    ConfirmedDirectRetryRegressionRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText(_RETRY_FIRST, 0.999655),
            RecognizedText(_RETRY_DIRECT, 0.575256),
            RecognizedText(_RETRY_SELECTED, 0.765657),
            RecognizedText("K", 0.985161),
            RecognizedText(_RETRY_STRUCTURED, 0.982269),
            RecognizedText(_RETRY_ONE, 0.999974),
            RecognizedText(_RETRY_TWO, 0.999927),
            *self.values,
        )

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return (
                (34, 163),
                (180, 274),
                (290, 310),
                (309, 446),
                (460, 491),
                (505, 566),
            )
        return ((0, _image.width),)


class DirectRetryRegressionDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    88.4,
                    153.58695652173913,
                    691.6,
                    188.80434782608697,
                ),
                0.9,
            ),
        )


def direct_retry_regression_words() -> tuple[
    list[tuple[str, BoundingBox, float]],
    list[tuple[str, BoundingBox, float]],
]:
    boxes = (
        BoundingBox(34, 0, 163, _RETRY_HEIGHT),
        BoundingBox(180, 0, 274, _RETRY_HEIGHT),
        BoundingBox(290, 0, 310, _RETRY_HEIGHT),
        BoundingBox(309, 0, 446, _RETRY_HEIGHT),
        BoundingBox(460, 0, 491, _RETRY_HEIGHT),
        BoundingBox(505, 0, 566, _RETRY_HEIGHT),
    )
    raw = [
        (_RETRY_FIRST, boxes[0], 0.999655),
        (_RETRY_DIRECT, boxes[1], 0.575256),
        ("K", boxes[2], 0.985161),
        (_RETRY_STRUCTURED, boxes[3], 0.982269),
        (_RETRY_ONE, boxes[4], 0.999974),
        (_RETRY_TWO, boxes[5], 0.999927),
    ]
    selected = list(raw)
    selected[1] = (_RETRY_SELECTED, boxes[1], 0.765657)
    return selected, raw


def test_confirmed_direct_retry_regression_preserves_stable_direct_reading() -> None:
    selected, raw = direct_retry_regression_words()

    recovered = _recover_confirmed_direct_retry_regression(
        selected,
        raw,
        Image.new("RGB", (604, 36)),
        BoundingBox(0, 0, 603.2, _RETRY_HEIGHT),
        ConfirmedDirectRetryRegressionRecognizer(),
    )

    expected = list(selected)
    expected[1] = (_RETRY_DIRECT, raw[1][1], 0.569)
    assert recovered == expected


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedDirectRetryRegressionRecognizer(
            variant_texts=(
                _RETRY_SELECTED,
                _RETRY_DIRECT,
                _RETRY_DIRECT,
                _RETRY_DIRECT,
                _RETRY_DIRECT,
            )
        ),
        ConfirmedDirectRetryRegressionRecognizer(
            variant_confidences=(0.599, 0.627, 0.592, 0.684, 0.569)
        ),
        ConfirmedDirectRetryRegressionRecognizer(
            variant_confidences=(0.601, 0.619, 0.592, 0.684, 0.569)
        ),
        ConfirmedDirectRetryRegressionRecognizer(
            variant_confidences=(0.601, 0.627, 0.589, 0.684, 0.569)
        ),
        ConfirmedDirectRetryRegressionRecognizer(
            variant_confidences=(0.601, 0.627, 0.592, 0.679, 0.569)
        ),
        ConfirmedDirectRetryRegressionRecognizer(
            variant_confidences=(0.601, 0.627, 0.592, 0.684, 0.568)
        ),
    ],
)
def test_confirmed_direct_retry_regression_requires_all_crop_evidence(
    recognizer,
) -> None:
    selected, raw = direct_retry_regression_words()

    assert (
        _recover_confirmed_direct_retry_regression(
            selected,
            raw,
            Image.new("RGB", (604, 36)),
            BoundingBox(0, 0, 603.2, _RETRY_HEIGHT),
            recognizer,
        )
        == selected
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "box-mismatch",
        "no-disagreement",
        "raw-confidence",
        "selected-confidence",
        "first-shape",
        "marker",
        "structured-shape",
        "candidate-width",
        "candidate-gap",
        "trailing-confidence",
    ],
)
def test_confirmed_direct_retry_regression_requires_exact_line_profile(
    case: str,
) -> None:
    selected, raw = direct_retry_regression_words()
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "box-mismatch":
        raw[1] = (
            raw[1][0],
            BoundingBox(181, 0, 274, _RETRY_HEIGHT),
            raw[1][2],
        )
    elif case == "no-disagreement":
        selected[1] = (raw[1][0], raw[1][1], selected[1][2])
    elif case == "raw-confidence":
        raw[1] = (raw[1][0], raw[1][1], 0.5751)
    elif case == "selected-confidence":
        selected[1] = (selected[1][0], selected[1][1], 0.7655)
    elif case == "first-shape":
        raw[0] = selected[0] = (_RETRY_FIRST[:-1], raw[0][1], raw[0][2])
    elif case == "marker":
        raw[2] = selected[2] = ("M", raw[2][1], raw[2][2])
    elif case == "structured-shape":
        raw[3] = selected[3] = ("ABCDEFGH", raw[3][1], raw[3][2])
    elif case == "candidate-width":
        box = BoundingBox(180, 0, 275, _RETRY_HEIGHT)
        raw[1] = (raw[1][0], box, raw[1][2])
        selected[1] = (selected[1][0], box, selected[1][2])
    elif case == "candidate-gap":
        box = BoundingBox(289, 0, 310, _RETRY_HEIGHT)
        raw[2] = (raw[2][0], box, raw[2][2])
        selected[2] = (selected[2][0], box, selected[2][2])
    else:
        raw[5] = selected[5] = (raw[5][0], raw[5][1], 0.9998)

    assert (
        _recover_confirmed_direct_retry_regression(
            selected,
            raw,
            Image.new("RGB", (604, 36)),
            BoundingBox(0, 0, 603.2, _RETRY_HEIGHT),
            ConfirmedDirectRetryRegressionRecognizer(),
        )
        == selected
    )


def test_engine_preserves_confirmed_direct_reading_over_retry_regression() -> None:
    engine = PaddleOcrEngine(
        DirectRetryRegressionDetector(),
        DirectRetryRegressionRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (800, 350)))

    assert [word.text for word in document.lines[0].eojeols] == [
        _RETRY_FIRST,
        _RETRY_DIRECT,
        _RETRY_ONE,
        _RETRY_TWO,
    ]
    assert document.lines[0].eojeols[1].box == BoundingBox(
        268.4,
        153.58695652173913,
        362.4,
        188.80434782608697,
    )


_TERMINAL_THREE_HEIGHT = 28.17391304347825
_TERMINAL_THREE_RAW = "".join(chr(0xCA00 + offset) for offset in range(3))
_TERMINAL_THREE_SELECTED = "".join(chr(0xCA10 + offset) for offset in range(3))
_TERMINAL_THREE_RECOVERED = "".join(chr(0xCA20 + offset) for offset in range(3))


def _terminal_three_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def terminal_three_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(122.64, 0, 207.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(223.64, 0, 311.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(323.64, 0, 411.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(424.64, 0, 526.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(540.64, 0, 623.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(639.64, 0, 757.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(771.64, 0, 983.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(998.64, 0, 1081.64, _TERMINAL_THREE_HEIGHT),
        BoundingBox(1092.64, 0, 1150.64, _TERMINAL_THREE_HEIGHT),
    )
    texts = (
        _terminal_three_hangul(0xCA30, 3),
        _terminal_three_hangul(0xCA33, 3),
        _terminal_three_hangul(0xCA36, 3),
        _TERMINAL_THREE_RAW + ".",
        _terminal_three_hangul(0xCA39, 3),
        _terminal_three_hangul(0xCA3C, 4),
        _terminal_three_hangul(0xCA40, 7) + ".",
        _terminal_three_hangul(0xCA47, 3),
        "1",
    )
    confidences = (
        0.999858,
        0.999741,
        0.999385,
        0.486436,
        0.999128,
        0.999758,
        0.879968,
        0.998294,
        0.250173,
    )
    return list(zip(texts, boxes, confidences, strict=True))


def terminal_three_selected_words() -> list[tuple[str, BoundingBox, float]]:
    selected = terminal_three_raw_words()[:8]
    selected[3] = (
        _TERMINAL_THREE_SELECTED + ".",
        selected[3][1],
        0.509816,
    )
    return selected


_TERMINAL_THREE_DIRECT_CONFIDENCES = (
    0.900825,
    0.898308,
    0.886824,
    0.881067,
    0.886181,
)
_TERMINAL_THREE_ENHANCED_CONFIDENCES = (
    0.919025,
    0.912967,
    0.912771,
    0.908847,
    0.905918,
    0.904881,
    0.901426,
    0.896409,
)


class ConfirmedTerminalThreeRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_TERMINAL_THREE_RECOVERED,) * 5,
        direct_confidences: tuple[float, ...] = _TERMINAL_THREE_DIRECT_CONFIDENCES,
        enhanced_texts: tuple[str, ...] = (_TERMINAL_THREE_RECOVERED,) * 8,
        enhanced_confidences: tuple[float, ...] = (
            _TERMINAL_THREE_ENHANCED_CONFIDENCES
        ),
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_terminal_three_substitution_recovers_reading() -> None:
    raw = terminal_three_raw_words()
    selected = terminal_three_selected_words()
    recognizer = ConfirmedTerminalThreeRecognizer()

    recovered = _recover_confirmed_terminal_three_substitution(
        selected,
        raw,
        Image.new("RGB", (1094, 29)),
        BoundingBox(56.64, 0, 1149.36, _TERMINAL_THREE_HEIGHT),
        recognizer,
    )

    expected = list(selected)
    expected[3] = (
        _TERMINAL_THREE_RECOVERED + ".",
        raw[3][1],
        0.486436,
    )
    assert recovered == expected
    assert recognizer.calls == 13


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedTerminalThreeRecognizer(
            direct_texts=(
                _TERMINAL_THREE_RAW,
                *([_TERMINAL_THREE_RECOVERED] * 4),
            )
        ),
        ConfirmedTerminalThreeRecognizer(
            direct_texts=(
                _TERMINAL_THREE_RECOVERED,
                _TERMINAL_THREE_SELECTED,
                *([_TERMINAL_THREE_RECOVERED] * 3),
            )
        ),
        ConfirmedTerminalThreeRecognizer(
            direct_confidences=(0.9007, 0.898308, 0.886824, 0.881067, 0.886181)
        ),
        ConfirmedTerminalThreeRecognizer(
            enhanced_texts=(
                *([_TERMINAL_THREE_RECOVERED] * 4),
                _TERMINAL_THREE_SELECTED,
                *([_TERMINAL_THREE_RECOVERED] * 3),
            )
        ),
        ConfirmedTerminalThreeRecognizer(
            enhanced_confidences=(
                0.919025,
                0.912967,
                0.912771,
                0.9087,
                0.905918,
                0.904881,
                0.901426,
                0.896409,
            )
        ),
        ConfirmedTerminalThreeRecognizer(
            direct_texts=(
                _TERMINAL_THREE_RECOVERED + "A",
                *([_TERMINAL_THREE_RECOVERED] * 4),
            )
        ),
    ],
)
def test_confirmed_terminal_three_substitution_requires_crop_consensus(
    recognizer,
) -> None:
    raw = terminal_three_raw_words()
    selected = terminal_three_selected_words()

    assert (
        _recover_confirmed_terminal_three_substitution(
            selected,
            raw,
            Image.new("RGB", (1094, 29)),
            BoundingBox(56.64, 0, 1149.36, _TERMINAL_THREE_HEIGHT),
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 13


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "noncandidate-mismatch",
        "candidate-box",
        "candidate-interior",
        "terminal-mismatch",
        "raw-shape",
        "selected-shape",
        "raw-confidence",
        "selected-confidence",
        "target-width",
        "target-gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_terminal_three_substitution_requires_exact_profile(case: str) -> None:
    raw = terminal_three_raw_words()
    selected = terminal_three_selected_words()
    crop = Image.new("RGB", (1094, 29))
    line_box = BoundingBox(56.64, 0, 1149.36, _TERMINAL_THREE_HEIGHT)
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "noncandidate-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.99985)
    elif case == "candidate-box":
        selected[3] = (
            selected[3][0],
            BoundingBox(424.64, 0, 525.64, _TERMINAL_THREE_HEIGHT),
            selected[3][2],
        )
    elif case == "candidate-interior":
        selected[3] = (_TERMINAL_THREE_RAW + ".", selected[3][1], selected[3][2])
    elif case == "terminal-mismatch":
        selected[3] = (
            _TERMINAL_THREE_SELECTED + ",",
            selected[3][1],
            selected[3][2],
        )
    elif case == "raw-shape":
        raw[1] = (raw[1][0][:-1] + "A", raw[1][1], raw[1][2])
        selected[1] = raw[1]
    elif case == "selected-shape":
        selected[3] = (
            _TERMINAL_THREE_SELECTED[:-1] + "A.",
            selected[3][1],
            selected[3][2],
        )
    elif case == "raw-confidence":
        raw[3] = (raw[3][0], raw[3][1], 0.4863)
    elif case == "selected-confidence":
        selected[3] = (selected[3][0], selected[3][1], 0.5097)
    elif case == "target-width":
        box = BoundingBox(424.64, 0, 525.64, _TERMINAL_THREE_HEIGHT)
        raw[3] = (raw[3][0], box, raw[3][2])
        selected[3] = (selected[3][0], box, selected[3][2])
    elif case == "target-gap":
        box = BoundingBox(539.64, 0, 622.64, _TERMINAL_THREE_HEIGHT)
        raw[4] = (raw[4][0], box, raw[4][2])
        selected[4] = raw[4]
    elif case == "line-height":
        line_box = BoundingBox(56.64, 0, 1149.36, 0)
    else:
        crop = Image.new("RGB", (450, 29))
    recognizer = ConfirmedTerminalThreeRecognizer()

    assert (
        _recover_confirmed_terminal_three_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class TerminalThreeEngineRecognizer:
    def __init__(self) -> None:
        raw = terminal_three_raw_words()
        initial = [
            RecognizedText(raw[0][0], raw[0][2]),
            RecognizedText(raw[1][0], raw[1][2]),
            RecognizedText(raw[2][0], raw[2][2]),
            RecognizedText(raw[3][0], raw[3][2]),
            RecognizedText(_TERMINAL_THREE_SELECTED + ".", 0.509816),
            RecognizedText(raw[4][0], raw[4][2]),
            RecognizedText(raw[5][0], raw[5][2]),
            RecognizedText(raw[6][0], raw[6][2]),
            RecognizedText(raw[7][0], raw[7][2]),
            RecognizedText(raw[8][0], raw[8][2]),
            RecognizedText(raw[8][0], 0.248961),
        ]
        self.values = tuple(initial) + ConfirmedTerminalThreeRecognizer().values
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (66, 151),
            (167, 255),
            (267, 355),
            (368, 470),
            (484, 567),
            (583, 701),
            (715, 927),
            (942, 1025),
            (1036, 1094),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class TerminalThreeEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    56.64,
                    158.08695652173913,
                    1149.36,
                    186.26086956521738,
                ),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_terminal_three_substitution() -> None:
    recognizer = TerminalThreeEngineRecognizer()
    engine = PaddleOcrEngine(TerminalThreeEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[3]
    assert target.text == _TERMINAL_THREE_RECOVERED
    assert target.box == BoundingBox(
        424.64,
        158.08695652173913,
        501.14,
        186.26086956521738,
    )
    assert target.confidence == 0.486436


_ENHANCED_TWO_HEIGHT = 14.086956521739125
_ENHANCED_TWO_DIRECT = "".join(chr(0xCB00 + offset) for offset in range(2))
_ENHANCED_TWO_RECOVERED = "".join(chr(0xCB10 + offset) for offset in range(2))


def _enhanced_two_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def enhanced_two_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(18, 0, 40, _ENHANCED_TWO_HEIGHT),
        BoundingBox(43, 0, 90, _ENHANCED_TWO_HEIGHT),
        BoundingBox(93, 0, 108, _ENHANCED_TWO_HEIGHT),
        BoundingBox(111, 0, 169, _ENHANCED_TWO_HEIGHT),
        BoundingBox(172, 0, 207, _ENHANCED_TWO_HEIGHT),
        BoundingBox(211, 0, 272, _ENHANCED_TWO_HEIGHT),
    )
    texts = (
        _ENHANCED_TWO_DIRECT,
        _enhanced_two_hangul(0xCB20, 4),
        _enhanced_two_hangul(0xCB24, 1) + ")",
        _enhanced_two_hangul(0xCB25, 5),
        _enhanced_two_hangul(0xCB2A, 3),
        _enhanced_two_hangul(0xCB2D, 5) + ".",
    )
    confidences = (
        0.870781,
        0.999571,
        0.991686,
        0.999705,
        0.999335,
        0.942514,
    )
    return list(zip(texts, boxes, confidences, strict=True))


_ENHANCED_TWO_DIRECT_CONFIDENCES = (0.590744, 0.56195, 0.536506, 0.49422)
_ENHANCED_TWO_ENHANCED_CONFIDENCES = (
    0.934284,
    0.991545,
    0.976423,
    0.998939,
    0.986655,
    0.999466,
    0.999306,
    0.999241,
)


class ConfirmedEnhancedTwoRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_ENHANCED_TWO_RECOVERED,) * 4,
        direct_confidences: tuple[float, ...] = _ENHANCED_TWO_DIRECT_CONFIDENCES,
        enhanced_texts: tuple[str, ...] = (_ENHANCED_TWO_RECOVERED,) * 8,
        enhanced_confidences: tuple[float, ...] = _ENHANCED_TWO_ENHANCED_CONFIDENCES,
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_enhanced_two_substitution_recovers_reading() -> None:
    words = enhanced_two_words()
    recognizer = ConfirmedEnhancedTwoRecognizer()

    recovered = _recover_confirmed_enhanced_two_substitution(
        words,
        words,
        Image.new("RGB", (289, 15)),
        BoundingBox(0, 0, 288.84, _ENHANCED_TWO_HEIGHT),
        recognizer,
    )

    expected = list(words)
    expected[0] = (
        _ENHANCED_TWO_RECOVERED,
        words[0][1],
        0.49422,
    )
    assert recovered == expected
    assert recognizer.calls == 12


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedEnhancedTwoRecognizer(
            enhanced_texts=(
                _ENHANCED_TWO_DIRECT,
                *([_ENHANCED_TWO_RECOVERED] * 7),
            )
        ),
        ConfirmedEnhancedTwoRecognizer(
            direct_texts=(
                _ENHANCED_TWO_RECOVERED,
                _ENHANCED_TWO_DIRECT,
                _ENHANCED_TWO_RECOVERED,
                _ENHANCED_TWO_RECOVERED,
            )
        ),
        ConfirmedEnhancedTwoRecognizer(
            direct_confidences=(0.590744, 0.5618, 0.536506, 0.49422)
        ),
        ConfirmedEnhancedTwoRecognizer(
            enhanced_texts=(
                *([_ENHANCED_TWO_RECOVERED] * 5),
                _ENHANCED_TWO_DIRECT,
                _ENHANCED_TWO_RECOVERED,
                _ENHANCED_TWO_RECOVERED,
            )
        ),
        ConfirmedEnhancedTwoRecognizer(
            enhanced_confidences=(
                0.934284,
                0.991545,
                0.9763,
                0.998939,
                0.986655,
                0.999466,
                0.999306,
                0.999241,
            )
        ),
        ConfirmedEnhancedTwoRecognizer(
            enhanced_texts=(
                _ENHANCED_TWO_RECOVERED + "A",
                *([_ENHANCED_TWO_RECOVERED] * 7),
            )
        ),
    ],
)
def test_confirmed_enhanced_two_substitution_requires_crop_consensus(
    recognizer,
) -> None:
    words = enhanced_two_words()

    assert (
        _recover_confirmed_enhanced_two_substitution(
            words,
            words,
            Image.new("RGB", (289, 15)),
            BoundingBox(0, 0, 288.84, _ENHANCED_TWO_HEIGHT),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "neighbor-shape",
        "middle-category",
        "terminal-category",
        "confidence",
        "target-width",
        "target-gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_enhanced_two_substitution_requires_exact_profile(case: str) -> None:
    raw = enhanced_two_words()
    selected = list(raw)
    crop = Image.new("RGB", (289, 15))
    line_box = BoundingBox(0, 0, 288.84, _ENHANCED_TWO_HEIGHT)
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.87075)
    elif case == "neighbor-shape":
        raw[1] = (raw[1][0][:-1] + "A", raw[1][1], raw[1][2])
        selected = list(raw)
    elif case == "middle-category":
        raw[2] = (raw[2][0][:-1] + "|", raw[2][1], raw[2][2])
        selected = list(raw)
    elif case == "terminal-category":
        raw[5] = (raw[5][0][:-1] + "|", raw[5][1], raw[5][2])
        selected = list(raw)
    elif case == "confidence":
        raw[0] = (raw[0][0], raw[0][1], 0.8706)
        selected = list(raw)
    elif case == "target-width":
        raw[0] = (
            raw[0][0],
            BoundingBox(18, 0, 41, _ENHANCED_TWO_HEIGHT),
            raw[0][2],
        )
        selected = list(raw)
    elif case == "target-gap":
        raw[1] = (
            raw[1][0],
            BoundingBox(42, 0, 89, _ENHANCED_TWO_HEIGHT),
            raw[1][2],
        )
        selected = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(0, 0, 288.84, 0)
    else:
        crop = Image.new("RGB", (46, 15))
    recognizer = ConfirmedEnhancedTwoRecognizer()

    assert (
        _recover_confirmed_enhanced_two_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class EnhancedTwoEngineRecognizer:
    def __init__(self) -> None:
        raw = enhanced_two_words()
        values = [
            *(RecognizedText(text, confidence) for text, _box, confidence in raw),
            *ConfirmedEnhancedTwoRecognizer().values,
        ]
        self.values = tuple(values)
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (18, 40),
            (43, 90),
            (93, 108),
            (111, 169),
            (172, 207),
            (211, 272),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class EnhancedTwoEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(104.08, 153.3913043478261, 392.92, 167.47826086956522),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_enhanced_two_substitution() -> None:
    recognizer = EnhancedTwoEngineRecognizer()
    engine = PaddleOcrEngine(EnhancedTwoEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    assert document.lines[0].eojeols[0].text == _ENHANCED_TWO_RECOVERED
    assert document.lines[0].eojeols[0].box == BoundingBox(
        104.08 + 18,
        153.3913043478261,
        104.08 + 40,
        167.47826086956522,
    )


_ENHANCED_WRAPPED_HEIGHT = 14.086956521739125
_ENHANCED_WRAPPED_DIRECT = "".join(chr(0xCC20 + offset) for offset in range(4))
_ENHANCED_WRAPPED_RECOVERED = "".join(chr(0xCC30 + offset) for offset in range(4))


def _enhanced_wrapped_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def enhanced_wrapped_four_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(53, 0, 85, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(88, 0, 164, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(169, 0, 214, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(220, 0, 298, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(301, 0, 333, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(336, 0, 399, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(403, 0, 474, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(480, 0, 526, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(529, 0, 592, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(596, 0, 640, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(646, 0, 740, _ENHANCED_WRAPPED_HEIGHT),
        BoundingBox(743, 0, 778, _ENHANCED_WRAPPED_HEIGHT),
    )
    texts = (
        _enhanced_wrapped_hangul(0xCC00, 2),
        _enhanced_wrapped_hangul(0xCC02, 5),
        _enhanced_wrapped_hangul(0xCC07, 3),
        _enhanced_wrapped_hangul(0xCC0A, 5),
        _enhanced_wrapped_hangul(0xCC0F, 2),
        _enhanced_wrapped_hangul(0xCC11, 4),
        "[" + _ENHANCED_WRAPPED_DIRECT + "]",
        _enhanced_wrapped_hangul(0xCC40, 3),
        _enhanced_wrapped_hangul(0xCC43, 4),
        _enhanced_wrapped_hangul(0xCC47, 3),
        _enhanced_wrapped_hangul(0xCC4A, 6),
        _enhanced_wrapped_hangul(0xCC50, 2) + ")",
    )
    confidences = (
        0.999122,
        0.999422,
        0.999662,
        0.999737,
        0.999993,
        0.999212,
        0.927493,
        0.999886,
        0.999932,
        0.998828,
        0.998839,
        0.982999,
    )
    return list(zip(texts, boxes, confidences, strict=True))


_ENHANCED_WRAPPED_DIRECT_TEXTS = (
    _ENHANCED_WRAPPED_RECOVERED,
    _ENHANCED_WRAPPED_RECOVERED,
    _ENHANCED_WRAPPED_RECOVERED,
    _ENHANCED_WRAPPED_RECOVERED,
    _ENHANCED_WRAPPED_RECOVERED,
    _ENHANCED_WRAPPED_RECOVERED,
    "[" + _ENHANCED_WRAPPED_RECOVERED,
    "[" + _ENHANCED_WRAPPED_RECOVERED,
)
_ENHANCED_WRAPPED_DIRECT_CONFIDENCES = (
    0.999816,
    0.999541,
    0.998467,
    0.892696,
    0.999422,
    0.998511,
    0.998713,
    0.998445,
)
_ENHANCED_WRAPPED_ENHANCED_TEXTS = (
    "[" + _ENHANCED_WRAPPED_RECOVERED + "]",
    "[" + _ENHANCED_WRAPPED_RECOVERED + "]",
    _ENHANCED_WRAPPED_RECOVERED + "]",
    "[" + _ENHANCED_WRAPPED_RECOVERED + "]",
    _ENHANCED_WRAPPED_RECOVERED,
    "[" + _ENHANCED_WRAPPED_RECOVERED + "]",
    _ENHANCED_WRAPPED_RECOVERED + "]",
)
_ENHANCED_WRAPPED_ENHANCED_CONFIDENCES = (
    0.991281,
    0.981893,
    0.618867,
    0.933609,
    0.999189,
    0.959106,
    0.983472,
)


class ConfirmedEnhancedWrappedFourRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = _ENHANCED_WRAPPED_DIRECT_TEXTS,
        direct_confidences: tuple[float, ...] = _ENHANCED_WRAPPED_DIRECT_CONFIDENCES,
        enhanced_texts: tuple[str, ...] = _ENHANCED_WRAPPED_ENHANCED_TEXTS,
        enhanced_confidences: tuple[float, ...] = _ENHANCED_WRAPPED_ENHANCED_CONFIDENCES,
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_enhanced_wrapped_four_substitution_recovers_interior() -> None:
    words = enhanced_wrapped_four_words()
    recognizer = ConfirmedEnhancedWrappedFourRecognizer()

    recovered = _recover_confirmed_enhanced_wrapped_four_substitution(
        words,
        words,
        Image.new("RGB", (830, 15)),
        BoundingBox(0, 0, 828.24, _ENHANCED_WRAPPED_HEIGHT),
        recognizer,
    )

    expected = list(words)
    expected[6] = (
        _ENHANCED_WRAPPED_RECOVERED,
        BoundingBox(
            403 + 71 / 6,
            0,
            474 - 71 / 6,
            _ENHANCED_WRAPPED_HEIGHT,
        ),
        0.618867,
    )
    assert recovered == expected
    assert recognizer.calls == 15


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedEnhancedWrappedFourRecognizer(
            enhanced_texts=(
                "(" + _ENHANCED_WRAPPED_RECOVERED + ")",
                *_ENHANCED_WRAPPED_ENHANCED_TEXTS[1:],
            )
        ),
        ConfirmedEnhancedWrappedFourRecognizer(
            enhanced_texts=(
                "[" + _ENHANCED_WRAPPED_DIRECT + "]",
                *_ENHANCED_WRAPPED_ENHANCED_TEXTS[1:],
            )
        ),
        ConfirmedEnhancedWrappedFourRecognizer(
            direct_texts=(
                *_ENHANCED_WRAPPED_DIRECT_TEXTS[:2],
                _ENHANCED_WRAPPED_DIRECT,
                *_ENHANCED_WRAPPED_DIRECT_TEXTS[3:],
            )
        ),
        ConfirmedEnhancedWrappedFourRecognizer(
            direct_confidences=(
                *_ENHANCED_WRAPPED_DIRECT_CONFIDENCES[:3],
                0.8925,
                *_ENHANCED_WRAPPED_DIRECT_CONFIDENCES[4:],
            )
        ),
        ConfirmedEnhancedWrappedFourRecognizer(
            enhanced_texts=(
                *_ENHANCED_WRAPPED_ENHANCED_TEXTS[:4],
                _ENHANCED_WRAPPED_DIRECT,
                *_ENHANCED_WRAPPED_ENHANCED_TEXTS[5:],
            )
        ),
        ConfirmedEnhancedWrappedFourRecognizer(
            enhanced_confidences=(
                *_ENHANCED_WRAPPED_ENHANCED_CONFIDENCES[:2],
                0.6187,
                *_ENHANCED_WRAPPED_ENHANCED_CONFIDENCES[3:],
            )
        ),
    ],
)
def test_confirmed_enhanced_wrapped_four_substitution_requires_crop_consensus(
    recognizer,
) -> None:
    words = enhanced_wrapped_four_words()

    assert (
        _recover_confirmed_enhanced_wrapped_four_substitution(
            words,
            words,
            Image.new("RGB", (830, 15)),
            BoundingBox(0, 0, 828.24, _ENHANCED_WRAPPED_HEIGHT),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "neighbor-shape",
        "candidate-wrapper",
        "terminal-shape",
        "confidence",
        "target-width",
        "target-gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_enhanced_wrapped_four_substitution_requires_exact_profile(
    case: str,
) -> None:
    raw = enhanced_wrapped_four_words()
    selected = list(raw)
    line_box = BoundingBox(0, 0, 828.24, _ENHANCED_WRAPPED_HEIGHT)
    crop = Image.new("RGB", (830, 15))
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.99915)
    elif case == "neighbor-shape":
        raw[1] = (raw[1][0][:-1] + "A", raw[1][1], raw[1][2])
        selected = list(raw)
    elif case == "candidate-wrapper":
        raw[6] = ("(" + raw[6][0][1:-1] + "]", raw[6][1], raw[6][2])
        selected = list(raw)
    elif case == "terminal-shape":
        raw[11] = (raw[11][0][:-1] + "|", raw[11][1], raw[11][2])
        selected = list(raw)
    elif case == "confidence":
        raw[6] = (raw[6][0], raw[6][1], 0.9273)
        selected = list(raw)
    elif case == "target-width":
        raw[6] = (
            raw[6][0],
            BoundingBox(403, 0, 475, _ENHANCED_WRAPPED_HEIGHT),
            raw[6][2],
        )
        selected = list(raw)
    elif case == "target-gap":
        raw[7] = (
            raw[7][0],
            BoundingBox(479, 0, 525, _ENHANCED_WRAPPED_HEIGHT),
            raw[7][2],
        )
        selected = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(0, 0, 828.24, 0)
    else:
        crop = Image.new("RGB", (475, 15))
    recognizer = ConfirmedEnhancedWrappedFourRecognizer()

    assert (
        _recover_confirmed_enhanced_wrapped_four_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class EnhancedWrappedFourEngineRecognizer:
    def __init__(self) -> None:
        raw = enhanced_wrapped_four_words()
        segments = [
            RecognizedText("", 0.0),
            *(RecognizedText(text, confidence) for text, _box, confidence in raw),
            RecognizedText("", 0.0),
        ]
        values = []
        for value in segments:
            values.append(value)
            if value.confidence < 0.72:
                values.append(RecognizedText("", 0.0))
        values.extend(ConfirmedEnhancedWrappedFourRecognizer().values)
        self.values = tuple(values)
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (8, 47),
            (53, 85),
            (88, 164),
            (169, 214),
            (220, 298),
            (301, 333),
            (336, 399),
            (403, 474),
            (480, 526),
            (529, 592),
            (596, 640),
            (646, 740),
            (743, 778),
            (786, 830),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class EnhancedWrappedFourEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(68.88, 155.3478260869565, 897.12, 169.43478260869563),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_enhanced_wrapped_four_substitution() -> None:
    recognizer = EnhancedWrappedFourEngineRecognizer()
    engine = PaddleOcrEngine(EnhancedWrappedFourEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    assert document.lines[0].eojeols[6].text == _ENHANCED_WRAPPED_RECOVERED
    assert document.lines[0].eojeols[6].box == BoundingBox(
        68.88 + 403 + 71 / 6,
        155.3478260869565,
        68.88 + 474 - 71 / 6,
        169.43478260869563,
    )


_PAIRED_WRAPPER_HEIGHT = 15.84
_PAIRED_WRAPPER_SELECTED_INDEXES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)


def _paired_wrapper_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


_PAIRED_WRAPPER_DIRECT = _paired_wrapper_hangul(0xD400, 4)
_PAIRED_WRAPPER_RECOVERED = _paired_wrapper_hangul(0xD410, 4)
_PAIRED_WRAPPER_NEIGHBORS = (
    _paired_wrapper_hangul(0xD420, 3),
    _paired_wrapper_hangul(0xD423, 4),
    _paired_wrapper_hangul(0xD427, 3),
    _paired_wrapper_hangul(0xD42A, 2),
    _paired_wrapper_hangul(0xD42C, 4),
    _paired_wrapper_hangul(0xD430, 5),
    _paired_wrapper_hangul(0xD435, 1),
    _paired_wrapper_hangul(0xD436, 2),
    _paired_wrapper_hangul(0xD438, 7),
)


def paired_wrapper_four_words() -> tuple[
    list[tuple[str, BoundingBox, float]],
    list[tuple[str, BoundingBox, float]],
]:
    boxes = (
        BoundingBox(43, 0, 50, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(49, 0, 87, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(86, 0, 150, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(153, 0, 196, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(201, 0, 230, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(234, 0, 293, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(297, 0, 370, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(375, 0, 389, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(393, 0, 420, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(426, 0, 528, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(533, 0, 591, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(590, 0, 600, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(605, 0, 639, _PAIRED_WRAPPER_HEIGHT),
        BoundingBox(644, 0, 680, _PAIRED_WRAPPER_HEIGHT),
    )
    texts = (
        "A",
        *_PAIRED_WRAPPER_NEIGHBORS,
        "[" + _PAIRED_WRAPPER_DIRECT,
        "|",
        _paired_wrapper_hangul(0xD43F, 2) + ")",
        "7",
    )
    confidences = (
        0.348547,
        0.69239,
        0.998497,
        0.999928,
        0.999994,
        0.999786,
        0.999858,
        0.998661,
        0.999951,
        0.999655,
        0.85953,
        0.988034,
        0.584838,
        0.212375,
    )
    raw = list(zip(texts, boxes, confidences, strict=True))
    return paired_wrapper_selected(raw), raw


def paired_wrapper_selected(
    raw: list[tuple[str, BoundingBox, float]],
) -> list[tuple[str, BoundingBox, float]]:
    selected = [raw[index] for index in _PAIRED_WRAPPER_SELECTED_INDEXES]
    selected[0] = (selected[0][0], selected[0][1], 0.908223)
    selected[10] = (selected[10][0], selected[10][1], 0.997919)
    return selected


class ConfirmedPairedWrapperFourRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] | None = None,
        direct_confidences: tuple[float, ...] = (
            0.997876,
            0.999535,
            0.981432,
            0.999898,
            0.990671,
            0.996925,
            0.843608,
            0.999881,
            0.99974,
        ),
        enhanced_texts: tuple[str, ...] | None = None,
        enhanced_confidences: tuple[float, ...] = (
            0.999436,
            0.999952,
            0.998851,
            0.994735,
            0.999413,
        ),
    ) -> None:
        wrapped = "[" + _PAIRED_WRAPPER_RECOVERED + "]"
        direct = direct_texts or (
            wrapped,
            wrapped,
            "[" + _PAIRED_WRAPPER_RECOVERED,
            "[" + _PAIRED_WRAPPER_RECOVERED,
            wrapped,
            wrapped,
            _PAIRED_WRAPPER_RECOVERED + "]",
            wrapped,
            _PAIRED_WRAPPER_RECOVERED,
        )
        enhanced = enhanced_texts or (wrapped,) * 5
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct, *enhanced),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_paired_wrapper_four_substitution_recovers_interior() -> None:
    selected, raw = paired_wrapper_four_words()
    recognizer = ConfirmedPairedWrapperFourRecognizer()

    recovered = _recover_confirmed_paired_wrapper_four_substitution(
        selected,
        raw,
        Image.new("RGB", (680, 16)),
        BoundingBox(0, 0, 679.76, _PAIRED_WRAPPER_HEIGHT),
        recognizer,
    )

    expected = list(selected)
    expected[9] = (
        _PAIRED_WRAPPER_RECOVERED,
        BoundingBox(544.6, 0, 591, _PAIRED_WRAPPER_HEIGHT),
        0.843608,
    )
    assert recovered == expected
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedPairedWrapperFourRecognizer(
            direct_texts=(
                "(" + _PAIRED_WRAPPER_RECOVERED + "]",
                *(["[" + _PAIRED_WRAPPER_RECOVERED + "]"] * 8),
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            direct_texts=(
                "(" + _PAIRED_WRAPPER_RECOVERED + ")",
                *(["[" + _PAIRED_WRAPPER_RECOVERED + "]"] * 8),
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            direct_texts=(
                "[" + _PAIRED_WRAPPER_RECOVERED + "]",
                "[" + _PAIRED_WRAPPER_DIRECT + "]",
                *(["[" + _PAIRED_WRAPPER_RECOVERED + "]"] * 7),
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            direct_confidences=(
                0.9977,
                0.999535,
                0.981432,
                0.999898,
                0.990671,
                0.996925,
                0.843608,
                0.999881,
                0.99974,
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            direct_confidences=(
                0.997876,
                0.999535,
                0.981432,
                0.999898,
                0.990671,
                0.996925,
                0.8435,
                0.999881,
                0.99974,
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            enhanced_texts=(
                "[" + _PAIRED_WRAPPER_DIRECT + "]",
                *(["[" + _PAIRED_WRAPPER_RECOVERED + "]"] * 4),
            )
        ),
        ConfirmedPairedWrapperFourRecognizer(
            enhanced_confidences=(0.999436, 0.999952, 0.998851, 0.9946, 0.999413)
        ),
    ],
)
def test_confirmed_paired_wrapper_four_substitution_requires_crop_consensus(
    recognizer,
) -> None:
    selected, raw = paired_wrapper_four_words()

    assert (
        _recover_confirmed_paired_wrapper_four_substitution(
            selected,
            raw,
            Image.new("RGB", (680, 16)),
            BoundingBox(0, 0, 679.76, _PAIRED_WRAPPER_HEIGHT),
            recognizer,
        )
        == selected
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-text",
        "selected-box",
        "ascii-edge",
        "neighbor-shape",
        "candidate-wrapper",
        "right-category",
        "following-shape",
        "raw-confidence",
        "selected-confidence",
        "target-width",
        "target-gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_paired_wrapper_four_substitution_requires_exact_profile(
    case: str,
) -> None:
    selected, raw = paired_wrapper_four_words()
    line_box = BoundingBox(0, 0, 679.76, _PAIRED_WRAPPER_HEIGHT)
    crop = Image.new("RGB", (680, 16))
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-text":
        selected[9] = (
            "[" + _PAIRED_WRAPPER_RECOVERED,
            selected[9][1],
            selected[9][2],
        )
    elif case == "selected-box":
        selected[9] = (
            selected[9][0],
            BoundingBox(534, 0, 591, _PAIRED_WRAPPER_HEIGHT),
            selected[9][2],
        )
    elif case == "ascii-edge":
        raw[0] = ("(", raw[0][1], raw[0][2])
        selected = paired_wrapper_selected(raw)
    elif case == "neighbor-shape":
        raw[2] = (raw[2][0][:-1] + "A", raw[2][1], raw[2][2])
        selected = paired_wrapper_selected(raw)
    elif case == "candidate-wrapper":
        raw[10] = ("{" + raw[10][0][1:], raw[10][1], raw[10][2])
        selected = paired_wrapper_selected(raw)
    elif case == "right-category":
        raw[11] = ("]", raw[11][1], raw[11][2])
        selected = paired_wrapper_selected(raw)
    elif case == "following-shape":
        raw[12] = (raw[12][0][:-1] + "|", raw[12][1], raw[12][2])
        selected = paired_wrapper_selected(raw)
    elif case == "raw-confidence":
        raw[10] = (raw[10][0], raw[10][1], 0.8594)
        selected = paired_wrapper_selected(raw)
    elif case == "selected-confidence":
        selected[0] = (selected[0][0], selected[0][1], 0.9081)
    elif case == "target-width":
        raw[10] = (
            raw[10][0],
            BoundingBox(533, 0, 592, _PAIRED_WRAPPER_HEIGHT),
            raw[10][2],
        )
        selected = paired_wrapper_selected(raw)
    elif case == "target-gap":
        raw[10] = (
            raw[10][0],
            BoundingBox(532, 0, 591, _PAIRED_WRAPPER_HEIGHT),
            raw[10][2],
        )
        selected = paired_wrapper_selected(raw)
    elif case == "line-height":
        line_box = BoundingBox(0, 0, 679.76, 0)
    else:
        crop = Image.new("RGB", (599, 16))
    recognizer = ConfirmedPairedWrapperFourRecognizer()

    assert (
        _recover_confirmed_paired_wrapper_four_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class PairedWrapperFourEngineRecognizer:
    def __init__(self) -> None:
        _selected, raw = paired_wrapper_four_words()
        retry_values = {
            0: RecognizedText("B", 0.452154),
            1: RecognizedText(raw[1][0], 0.908223),
            12: RecognizedText(raw[12][0], 0.997919),
            13: RecognizedText(raw[13][0], 0.1),
        }
        values = []
        for index, (text, _box, confidence) in enumerate(raw):
            values.append(RecognizedText(text, confidence))
            if confidence < 0.72:
                values.append(retry_values[index])
        values.extend(ConfirmedPairedWrapperFourRecognizer().values)
        self.values = tuple(values)
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (43, 50),
            (49, 87),
            (86, 150),
            (153, 196),
            (201, 230),
            (234, 293),
            (297, 370),
            (375, 389),
            (393, 420),
            (426, 528),
            (533, 591),
            (590, 600),
            (605, 639),
            (644, 680),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class PairedWrapperFourEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(78.12, 154.96, 757.88, 170.8),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_paired_wrapper_four_substitution() -> None:
    recognizer = PairedWrapperFourEngineRecognizer()
    engine = PaddleOcrEngine(PairedWrapperFourEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    assert document.lines[0].eojeols[9].text == _PAIRED_WRAPPER_RECOVERED
    assert document.lines[0].eojeols[9].box == BoundingBox(
        622.72,
        154.96,
        669.12,
        170.8,
    )
_RIGHT_WRAPPER_HEIGHT = 22.9
_RIGHT_WRAPPER_SELECTED_INDEXES = (1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _right_wrapper_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


_RIGHT_WRAPPER_PREVIOUS = _right_wrapper_hangul(0xD650, 5)
_RIGHT_WRAPPER_DIRECT = _right_wrapper_hangul(0xD660, 5)
_RIGHT_WRAPPER_RECOVERED = _right_wrapper_hangul(0xD670, 5)
_RIGHT_WRAPPER_FOLLOWING = (
    _right_wrapper_hangul(0xD680, 1),
    _right_wrapper_hangul(0xD681, 4),
    _right_wrapper_hangul(0xD685, 3),
    _right_wrapper_hangul(0xD688, 1),
    _right_wrapper_hangul(0xD689, 3),
    _right_wrapper_hangul(0xD68C, 3),
    _right_wrapper_hangul(0xD68F, 4),
    _right_wrapper_hangul(0xD693, 1),
    _right_wrapper_hangul(0xD694, 2),
    _right_wrapper_hangul(0xD696, 4) + ".",
)


def right_wrapper_five_words() -> tuple[
    list[tuple[str, BoundingBox, float]],
    list[tuple[str, BoundingBox, float]],
]:
    boxes = (
        BoundingBox(64, 0, 70, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(70, 0, 179, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(186, 0, 194, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(194, 0, 298, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(298, 0, 306, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(306, 0, 311, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(317, 0, 340, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(346, 0, 438, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(444, 0, 514, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(520, 0, 539, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(548, 0, 615, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(623, 0, 692, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(698, 0, 789, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(796, 0, 819, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(830, 0, 871, _RIGHT_WRAPPER_HEIGHT),
        BoundingBox(877, 0, 975, _RIGHT_WRAPPER_HEIGHT),
    )
    texts = (
        "(",
        _RIGHT_WRAPPER_PREVIOUS,
        "[",
        _RIGHT_WRAPPER_DIRECT,
        "|",
        ")",
        *_RIGHT_WRAPPER_FOLLOWING,
    )
    confidences = (
        0.186633,
        0.99997,
        0.455827,
        0.97731,
        0.444427,
        0.897864,
        0.999957,
        0.999859,
        0.999786,
        0.999991,
        0.999972,
        0.999934,
        0.999921,
        0.999973,
        0.998287,
        0.977605,
    )
    raw = list(zip(texts, boxes, confidences, strict=True))
    selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    return selected, raw


class ConfirmedRightWrapperFiveRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_RIGHT_WRAPPER_RECOVERED,) * 7,
        direct_confidences: tuple[float, ...] = (
            0.999785,
            0.999914,
            0.999543,
            0.999634,
            0.999663,
            0.999876,
            0.994755,
        ),
        enhanced_texts: tuple[str, ...] = (_RIGHT_WRAPPER_RECOVERED,) * 3,
        enhanced_confidences: tuple[float, ...] = (0.999727, 0.999925, 0.996725),
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_right_wrapper_five_substitution_recovers_consensus() -> None:
    selected, raw = right_wrapper_five_words()
    recognizer = ConfirmedRightWrapperFiveRecognizer()

    recovered = _recover_confirmed_right_wrapper_five_substitution(
        selected,
        raw,
        Image.new("RGB", (1041, 23)),
        BoundingBox(0, 0, 1040.52, _RIGHT_WRAPPER_HEIGHT),
        recognizer,
    )

    expected = list(selected)
    expected[1] = (_RIGHT_WRAPPER_RECOVERED, raw[3][1], raw[3][2])
    assert recovered == expected
    assert recognizer.calls == 10


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedRightWrapperFiveRecognizer(
            direct_texts=(
                _RIGHT_WRAPPER_DIRECT,
                *(_RIGHT_WRAPPER_RECOVERED,) * 6,
            )
        ),
        ConfirmedRightWrapperFiveRecognizer(
            direct_texts=(
                *(_RIGHT_WRAPPER_RECOVERED,) * 6,
                _RIGHT_WRAPPER_DIRECT,
            )
        ),
        ConfirmedRightWrapperFiveRecognizer(
            direct_confidences=(
                0.9996,
                0.999914,
                0.999543,
                0.999634,
                0.999663,
                0.999876,
                0.994755,
            )
        ),
        ConfirmedRightWrapperFiveRecognizer(
            direct_confidences=(
                0.999785,
                0.999914,
                0.999543,
                0.999634,
                0.999663,
                0.999876,
                0.9946,
            )
        ),
        ConfirmedRightWrapperFiveRecognizer(
            enhanced_texts=(
                _RIGHT_WRAPPER_DIRECT,
                _RIGHT_WRAPPER_RECOVERED,
                _RIGHT_WRAPPER_RECOVERED,
            )
        ),
        ConfirmedRightWrapperFiveRecognizer(
            enhanced_confidences=(0.999727, 0.999925, 0.9966)
        ),
    ],
)
def test_confirmed_right_wrapper_five_substitution_requires_crop_consensus(
    recognizer,
) -> None:
    selected, raw = right_wrapper_five_words()

    assert (
        _recover_confirmed_right_wrapper_five_substitution(
            selected,
            raw,
            Image.new("RGB", (1041, 23)),
            BoundingBox(0, 0, 1040.52, _RIGHT_WRAPPER_HEIGHT),
            recognizer,
        )
        == selected
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "left-category",
        "right-category",
        "candidate-shape",
        "candidate-confidence",
        "following-confidence",
        "candidate-width",
        "previous-gap",
        "terminal-category",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_right_wrapper_five_substitution_requires_exact_profile(
    case: str,
) -> None:
    selected, raw = right_wrapper_five_words()
    line_box = BoundingBox(0, 0, 1040.52, _RIGHT_WRAPPER_HEIGHT)
    crop = Image.new("RGB", (1041, 23))
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        selected[1] = (_RIGHT_WRAPPER_RECOVERED, selected[1][1], selected[1][2])
    elif case == "left-category":
        raw[2] = (chr(0xD69A), raw[2][1], raw[2][2])
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "right-category":
        raw[4] = (")", raw[4][1], raw[4][2])
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "candidate-shape":
        raw[3] = (_RIGHT_WRAPPER_DIRECT[:-1], raw[3][1], raw[3][2])
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "candidate-confidence":
        raw[3] = (raw[3][0], raw[3][1], 0.9772)
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "following-confidence":
        raw[6] = (raw[6][0], raw[6][1], 0.9996)
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "candidate-width":
        raw[3] = (
            raw[3][0],
            BoundingBox(194, 0, 299, _RIGHT_WRAPPER_HEIGHT),
            raw[3][2],
        )
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "previous-gap":
        raw[2] = (
            raw[2][0],
            BoundingBox(187, 0, 194, _RIGHT_WRAPPER_HEIGHT),
            raw[2][2],
        )
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "terminal-category":
        raw[15] = (raw[15][0][:-1] + chr(0xD69B), raw[15][1], raw[15][2])
        selected = [raw[index] for index in _RIGHT_WRAPPER_SELECTED_INDEXES]
    elif case == "line-height":
        line_box = BoundingBox(0, 0, 1040.52, 0)
    else:
        crop = Image.new("RGB", (300, 23))
    recognizer = ConfirmedRightWrapperFiveRecognizer()

    assert (
        _recover_confirmed_right_wrapper_five_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class RightWrapperFiveEngineRecognizer:
    def __init__(self) -> None:
        _selected, raw = right_wrapper_five_words()
        direct_by_segment = (
            raw[0],
            raw[1],
            raw[2],
            raw[3],
            raw[4],
            raw[5],
            *raw[6:14],
            ("", BoundingBox(819, 0, 822, _RIGHT_WRAPPER_HEIGHT), 0.0),
            raw[14],
            raw[15],
            ("", BoundingBox(982, 0, 1041, _RIGHT_WRAPPER_HEIGHT), 0.0),
        )
        values = []
        for text, _box, confidence in direct_by_segment:
            values.append(RecognizedText(text, confidence))
            if confidence < 0.72:
                values.append(RecognizedText(text, max(0.0, confidence - 0.05)))
        values.extend(ConfirmedRightWrapperFiveRecognizer().values)
        self.values = tuple(values)
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (64, 70),
            (70, 179),
            (186, 194),
            (194, 298),
            (298, 306),
            (306, 311),
            (317, 340),
            (346, 438),
            (444, 514),
            (520, 539),
            (548, 615),
            (623, 692),
            (698, 789),
            (796, 819),
            (827, 830),
            (830, 871),
            (877, 975),
            (982, 1041),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class RightWrapperFiveEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(56.24, 157.3, 1096.76, 180.2),
                0.9888,
            ),
        )


def test_engine_recovers_confirmed_right_wrapper_five_substitution() -> None:
    recognizer = RightWrapperFiveEngineRecognizer()
    engine = PaddleOcrEngine(RightWrapperFiveEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    assert document.lines[0].eojeols[1].text == _RIGHT_WRAPPER_RECOVERED
    assert document.lines[0].eojeols[1].box == BoundingBox(
        250.24,
        157.3,
        354.24,
        180.2,
    )
class ConfirmedSubstitutionRecognizer:
    def __init__(self, *values: RecognizedText) -> None:
        self.values = values
        self.recognition_calls = 0

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


@pytest.mark.parametrize(
    ("words", "recognizer", "expected"),
    [
        (
            [("[\uac00]", BoundingBox(20, 0, 44, 18), 0.745)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub098", 0.987),
                RecognizedText("\ub098", 0.996),
            ),
            [("[\ub098]", BoundingBox(20, 0, 44, 18), 0.745)],
        ),
        (
            [("\uac00\ub098", BoundingBox(20, 0, 42, 18), 0.905)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub2e4\ub77c", 0.9997),
            ),
            [("\ub2e4\ub77c", BoundingBox(20, 0, 42, 18), 0.905)],
        ),
        (
            [("[\uac00\ub098\ub2e4\ub77c]", BoundingBox(20, 0, 80, 18), 0.86)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub9c8\ubc14\uc0ac\uc544", 0.66),
                RecognizedText("\ub9c8\ubc14\uc0ac\uc544", 0.73),
            ),
            [("[\ub9c8\ubc14\uc0ac\uc544]", BoundingBox(20, 0, 80, 18), 0.66)],
        ),
        (
            [("\uac00\ub098\ub2e4\u2026", BoundingBox(20, 0, 64, 18), 0.935)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub77c\ub9c8\ubc14", 0.68),
                RecognizedText("\ub77c\ub9c8\ubc14", 0.72),
            ),
            [("\ub77c\ub9c8\ubc14\u2026", BoundingBox(20, 0, 64, 18), 0.68)],
        ),
        (
            [("\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14", BoundingBox(20, 0, 86, 18), 0.565)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub77c\ub9c8\ubc14\uc0ac\uc544\uc790", 0.55),
                RecognizedText("\ub77c\ub9c8\ubc14\uc0ac\uc544\uc790", 0.56),
            ),
            [("\ub77c\ub9c8\ubc14\uc0ac\uc544\uc790", BoundingBox(20, 0, 86, 18), 0.55)],
        ),
    ],
)
def test_confirmed_substitution_readings_recovers_reviewed_profiles(
    words,
    recognizer,
    expected,
) -> None:
    assert (
        _recover_confirmed_substitution_readings(
            words,
            Image.new("RGB", (110, 18)),
            BoundingBox(0, 0, 110, 18),
            recognizer,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("words", "recognizer"),
    [
        (
            [("[\uac00]", BoundingBox(20, 0, 44, 18), 0.745)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub098", 0.987),
                RecognizedText("\ub2e4", 0.996),
            ),
        ),
        (
            [("[\uac00]", BoundingBox(20, 0, 44, 18), 0.745)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub098", 0.9869),
                RecognizedText("\ub098", 0.996),
            ),
        ),
        (
            [("\uac00\ub098", BoundingBox(20, 0, 42, 18), 0.905)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub2e4\ub77c", 0.99969),
            ),
        ),
        (
            [("[\uac00\ub098\ub2e4\ub77c]", BoundingBox(20, 0, 80, 18), 0.86)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub9c8\ubc14\uc0ac\uc544", 0.659),
                RecognizedText("\ub9c8\ubc14\uc0ac\uc544", 0.73),
            ),
        ),
        (
            [("\uac00\ub098\ub2e4\u2026", BoundingBox(20, 0, 64, 18), 0.935)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub77c\ub9c8\ubc14", 0.68),
                RecognizedText("\ub77c\ub9c8\ub2e4", 0.72),
            ),
        ),
        (
            [("\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14", BoundingBox(20, 0, 86, 18), 0.565)],
            ConfirmedSubstitutionRecognizer(
                RecognizedText("\ub77c\ub9c8\ubc14\uc0ac\uc544\uc790", 0.55),
                RecognizedText("ABCDEF", 0.99),
            ),
        ),
    ],
)
def test_confirmed_substitution_readings_requires_exact_profile(
    words,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_substitution_readings(
            words,
            Image.new("RGB", (110, 18)),
            BoundingBox(0, 0, 110, 18),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    "words",
    [
        [("[\uac00]", BoundingBox(20, 0, 44, 18), 0.7399)],
        [("\uac00\ub098", BoundingBox(20, 0, 42, 18), 0.8999)],
        [("[\uac00\ub098\ub2e4\ub77c]", BoundingBox(20, 0, 80, 18), 0.8499)],
        [("\uac00\ub098\ub2e4\u2026", BoundingBox(20, 0, 64, 18), 0.9401)],
        [("\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14", BoundingBox(20, 0, 86, 18), 0.5701)],
        [("\uac00\ub098\ub2e4\ub77c", BoundingBox(20, 0, 80, 18), 0.86)],
        [("-\uac00\ub098\ub2e4\ub77c-", BoundingBox(20, 0, 80, 18), 0.86)],
    ],
)
def test_confirmed_substitution_readings_requires_word_shape(words) -> None:
    assert (
        _recover_confirmed_substitution_readings(
            words,
            Image.new("RGB", (110, 18)),
            BoundingBox(0, 0, 110, 18),
            ConfirmedSubstitutionRecognizer(),
        )
        == words
    )

class ConfirmedTwoPlusPunctuatedTwoRecognizer:
    def __init__(
        self,
        *,
        second_text: str = "\ub9d0\ud55c\u2026",
        second_confidence: float = 0.99974,
        segments: tuple[tuple[int, int], ...] = ((0, 32), (38, 89)),
    ) -> None:
        self.values = (
            RecognizedText("\uac19\uc774", 0.999957),
            RecognizedText(second_text, second_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.04
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_two_plus_punctuated_two_split_recovers_reviewed_profile() -> None:
    height = 17.6
    words = [
        ("\uac19\uc774\ub9d0\ud55c\u2026", BoundingBox(20, 0, 109, height), 0.885137),
    ]

    recovered = _recover_confirmed_two_plus_punctuated_two_split(
        words,
        Image.new("RGB", (130, 18)),
        BoundingBox(0, 0, 130, height),
        ConfirmedTwoPlusPunctuatedTwoRecognizer(),
    )

    assert recovered == [
        ("\uac19\uc774", BoundingBox(20, 0, 52, height), 0.885137),
        ("\ub9d0\ud55c\u2026", BoundingBox(58, 0, 109, height), 0.885137),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedTwoPlusPunctuatedTwoRecognizer(second_text="\ub9d0\ud588\u2026"),
        ConfirmedTwoPlusPunctuatedTwoRecognizer(second_confidence=0.99969),
        ConfirmedTwoPlusPunctuatedTwoRecognizer(segments=((0, 32), (39, 89))),
        ConfirmedTwoPlusPunctuatedTwoRecognizer(segments=((0, 31), (37, 89))),
        ConfirmedTwoPlusPunctuatedTwoRecognizer(segments=((2, 34), (40, 87))),
    ],
)
def test_confirmed_two_plus_punctuated_two_split_requires_exact_profile(
    recognizer,
) -> None:
    words = [
        ("\uac19\uc774\ub9d0\ud55c\u2026", BoundingBox(20, 0, 109, 17.6), 0.885137),
    ]

    assert (
        _recover_confirmed_two_plus_punctuated_two_split(
            words,
            Image.new("RGB", (130, 18)),
            BoundingBox(0, 0, 130, 17.6),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize("confidence", [0.8799, 0.8901])
def test_confirmed_two_plus_punctuated_two_split_requires_word_confidence(
    confidence,
) -> None:
    words = [
        ("\uac19\uc774\ub9d0\ud55c\u2026", BoundingBox(20, 0, 109, 17.6), confidence),
    ]

    assert (
        _recover_confirmed_two_plus_punctuated_two_split(
            words,
            Image.new("RGB", (130, 18)),
            BoundingBox(0, 0, 130, 17.6),
            ConfirmedTwoPlusPunctuatedTwoRecognizer(),
        )
        == words
    )


class ConfirmedTwoPlusFourRecognizer:
    def __init__(
        self,
        first_text: str = "\uc0ac\ub78c",
        second_text: str = "\ub4e4\uc5d0\uac8c\ub294",
        first_confidence: float = 0.9999,
        second_confidence: float = 0.8477,
        segments: tuple[tuple[int, int], ...] = ((0, 33), (39, 107)),
    ) -> None:
        self.values = (
            RecognizedText(first_text, first_confidence),
            RecognizedText(second_text, second_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.01
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_two_plus_four_split_recovers_low_ctc_space() -> None:
    words = [
        ("\uc55e\ub9d0", BoundingBox(0, 0, 15, 18), 0.999),
        ("\uc0ac\ub78c\ub4e4\uc5d0\uac8c\ub294", BoundingBox(20, 0, 127, 18), 0.8735),
        ("\ub4b7\ub9d0", BoundingBox(134, 0, 154, 18), 0.999),
    ]

    recovered = _recover_confirmed_two_plus_four_splits(
        words,
        Image.new("RGB", (165, 18)),
        BoundingBox(0, 0, 165, 18),
        ConfirmedTwoPlusFourRecognizer(),
    )

    assert recovered == [
        words[0],
        ("\uc0ac\ub78c", BoundingBox(20, 0, 53, 18), 0.8735),
        ("\ub4e4\uc5d0\uac8c\ub294", BoundingBox(59, 0, 127, 18), 0.8477),
        words[2],
    ]


def test_confirmed_two_plus_four_split_recovers_structured_identifier() -> None:
    words = [("\uc791\ub1442026", BoundingBox(10, 0, 113, 25), 0.9986)]

    recovered = _recover_confirmed_two_plus_four_splits(
        words,
        Image.new("RGB", (125, 25)),
        BoundingBox(0, 0, 125, 25),
        ConfirmedTwoPlusFourRecognizer(
            first_text="\uc791\ub144",
            second_text="2026",
            first_confidence=0.9999,
            second_confidence=0.9992,
            segments=((0, 46), (54, 103)),
        ),
    )

    assert recovered == [
        ("\uc791\ub144", BoundingBox(10, 0, 56, 25), 0.9986),
        ("2026", BoundingBox(64, 0, 113, 25), 0.9986),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedTwoPlusFourRecognizer(second_text="\ub4e4\uc5d0\uac8c\ub3c4"),
        ConfirmedTwoPlusFourRecognizer(second_confidence=0.8399),
        ConfirmedTwoPlusFourRecognizer(segments=((0, 33), (38, 106))),
    ],
)
def test_confirmed_two_plus_four_split_requires_exact_strong_parts(
    recognizer,
) -> None:
    words = [
        ("\uc0ac\ub78c\ub4e4\uc5d0\uac8c\ub294", BoundingBox(20, 0, 127, 18), 0.8735),
    ]

    assert (
        _recover_confirmed_two_plus_four_splits(
            words,
            Image.new("RGB", (140, 18)),
            BoundingBox(0, 0, 140, 18),
            recognizer,
        )
        == words
    )


def test_confirmed_two_plus_four_split_requires_detector_confidence_floor() -> None:
    words = [
        ("\uc0ac\ub78c\ub4e4\uc5d0\uac8c\ub294", BoundingBox(20, 0, 127, 18), 0.6499),
    ]

    assert (
        _recover_confirmed_two_plus_four_splits(
            words,
            Image.new("RGB", (140, 18)),
            BoundingBox(0, 0, 140, 18),
            ConfirmedTwoPlusFourRecognizer(
                second_confidence=0.9999,
            ),
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


class ConfirmedSevenCharacterRecognizer:
    def __init__(
        self,
        first_text: str,
        second_text: str,
        first_confidence: float,
        second_confidence: float,
        segments: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        self.values = (
            RecognizedText(first_text, first_confidence),
            RecognizedText(second_text, second_confidence),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.01
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


@pytest.mark.parametrize(
    ("text", "recognizer", "height", "word_confidence", "expected"),
    [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac",
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c\ub9c8",
                "\ubc14\uc0ac",
                0.9794,
                0.9998,
                ((0, 73), (77, 106)),
            ),
            12.32,
            0.9673,
            [
                (
                    "\uac00\ub098\ub2e4\ub77c\ub9c8",
                    BoundingBox(20, 0, 93, 12.32),
                    0.9673,
                ),
                ("\ubc14\uc0ac", BoundingBox(97, 0, 126, 12.32), 0.9673),
            ],
        ),
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac",
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c",
                "\ub9c8\ubc14\uc0ac",
                0.99991,
                0.99999,
                ((0, 112), (122, 208)),
            ),
            29.93,
            0.9998,
            [
                (
                    "\uac00\ub098\ub2e4\ub77c",
                    BoundingBox(20, 0, 132, 29.93),
                    0.9998,
                ),
                (
                    "\ub9c8\ubc14\uc0ac",
                    BoundingBox(142, 0, 228, 29.93),
                    0.9998,
                ),
            ],
        ),
    ],
)
def test_confirmed_seven_character_split_recovers_reviewed_profiles(
    text,
    recognizer,
    height,
    word_confidence,
    expected,
) -> None:
    words = [
        (
            text,
            BoundingBox(20, 0, expected[-1][1].right, height),
            word_confidence,
        )
    ]

    recovered = _recover_confirmed_seven_character_splits(
        words,
        Image.new("RGB", (250, round(height))),
        BoundingBox(0, 0, 250, height),
        recognizer,
    )

    assert recovered == expected


@pytest.mark.parametrize(
    ("recognizer", "height", "right"),
    [
        (
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c\ub9c8",
                "\ubc14\uc544",
                0.9794,
                0.9998,
                ((0, 73), (77, 106)),
            ),
            12.32,
            126,
        ),
        (
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c\ub9c8",
                "\ubc14\uc0ac",
                0.9789,
                0.9998,
                ((0, 73), (77, 106)),
            ),
            12.32,
            126,
        ),
        (
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c",
                "\ub9c8\ubc14\uc0ac",
                0.99989,
                0.99999,
                ((0, 112), (122, 208)),
            ),
            29.93,
            228,
        ),
        (
            ConfirmedSevenCharacterRecognizer(
                "\uac00\ub098\ub2e4\ub77c",
                "\ub9c8\ubc14\uc0ac",
                0.99991,
                0.99999,
                ((0, 112), (120, 208)),
            ),
            29.93,
            228,
        ),
    ],
)
def test_confirmed_seven_character_split_requires_exact_strong_parts(
    recognizer,
    height,
    right,
) -> None:
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14\uc0ac",
            BoundingBox(20, 0, right, height),
            0.9673,
        )
    ]

    assert (
        _recover_confirmed_seven_character_splits(
            words,
            Image.new("RGB", (250, round(height))),
            BoundingBox(0, 0, 250, height),
            recognizer,
        )
        == words
    )


class ConfirmedThreePlusTwoSplitRecognizer:
    def __init__(
        self,
        *,
        second_text: str = "\ub77c\ub9c8",
        first_confidence: float = 0.9993,
        segments: tuple[tuple[int, int], ...] = ((0, 93), (103, 159)),
    ) -> None:
        self.values = (
            RecognizedText("\uac00\ub098\ub2e4", first_confidence),
            RecognizedText(second_text, 0.9999),
        )
        self.segments = segments
        self.recognition_calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        assert space_threshold == 0.01
        return self.segments

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


def test_confirmed_three_plus_two_split_recovers_reviewed_profile() -> None:
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(20, 0, 179, 29.94),
            0.9991,
        )
    ]

    recovered = _recover_confirmed_three_plus_two_split(
        words,
        Image.new("RGB", (210, 30)),
        BoundingBox(0, 0, 210, 29.94),
        ConfirmedThreePlusTwoSplitRecognizer(),
    )

    assert recovered == [
        ("\uac00\ub098\ub2e4", BoundingBox(20, 0, 113, 29.94), 0.9991),
        ("\ub77c\ub9c8", BoundingBox(123, 0, 179, 29.94), 0.9991),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedThreePlusTwoSplitRecognizer(second_text="\ub77c\ubc14"),
        ConfirmedThreePlusTwoSplitRecognizer(first_confidence=0.9991),
        ConfirmedThreePlusTwoSplitRecognizer(segments=((0, 93), (101, 159))),
        ConfirmedThreePlusTwoSplitRecognizer(segments=((2, 93), (103, 159))),
    ],
)
def test_confirmed_three_plus_two_split_requires_exact_strong_parts(
    recognizer,
) -> None:
    words = [
        (
            "\uac00\ub098\ub2e4\ub77c\ub9c8",
            BoundingBox(20, 0, 179, 29.94),
            0.9991,
        )
    ]

    assert (
        _recover_confirmed_three_plus_two_split(
            words,
            Image.new("RGB", (210, 30)),
            BoundingBox(0, 0, 210, 29.94),
            recognizer,
        )
        == words
    )


class RelativeGapTwoPlusTwoRecognizer:
    def __init__(self, confidence: float = 0.999) -> None:
        self.confidence = confidence

    def recognize(self, _image):
        return RecognizedText("한계성에", self.confidence)


class LineInitialTwoPlusTwoRecognizer:
    def __init__(
        self,
        *,
        candidate_confidence: float = 0.99992,
        competitor_confidence: float = 0.7,
    ) -> None:
        self.candidate_confidence = candidate_confidence
        self.competitor_confidence = competitor_confidence

    def recognize(self, image):
        if image.width == 63:
            return RecognizedText("\ud55c\uacc4\uc131\uc5d0", self.candidate_confidence)
        return RecognizedText("competing", self.competitor_confidence)


def test_line_initial_two_plus_two_pair_merges_exact_union() -> None:
    words = [
        ("\ud55c\uacc4", BoundingBox(50.2, 0, 78.2, 20), 0.99982),
        ("\uc131\uc5d0", BoundingBox(83.36, 0, 112.36, 20), 0.99991),
        ("\ub4b7\ub9d0", BoundingBox(121.66, 0, 141.66, 20), 0.999),
    ]

    recovered = _recover_relative_gap_two_plus_two_pairs(
        words,
        Image.new("RGB", (160, 20)),
        BoundingBox(0, 0, 160, 20),
        LineInitialTwoPlusTwoRecognizer(),
    )

    assert recovered == [
        ("\ud55c\uacc4\uc131\uc5d0", BoundingBox(50.2, 0, 112.36, 20), 0.99982),
        words[2],
    ]


def test_line_initial_two_plus_two_pair_rejects_weak_union() -> None:
    words = [
        ("\ud55c\uacc4", BoundingBox(50.2, 0, 78.2, 20), 0.99982),
        ("\uc131\uc5d0", BoundingBox(83.36, 0, 112.36, 20), 0.99991),
        ("\ub4b7\ub9d0", BoundingBox(121.66, 0, 141.66, 20), 0.999),
    ]

    assert (
        _recover_relative_gap_two_plus_two_pairs(
            words,
            Image.new("RGB", (160, 20)),
            BoundingBox(0, 0, 160, 20),
            LineInitialTwoPlusTwoRecognizer(candidate_confidence=0.99989),
        )
        == words
    )


def test_line_initial_two_plus_two_pair_rejects_strong_competitor() -> None:
    words = [
        ("\ud55c\uacc4", BoundingBox(50.2, 0, 78.2, 20), 0.99982),
        ("\uc131\uc5d0", BoundingBox(83.36, 0, 112.36, 20), 0.99991),
        ("\ub4b7\ub9d0", BoundingBox(121.66, 0, 141.66, 20), 0.999),
    ]

    assert (
        _recover_relative_gap_two_plus_two_pairs(
            words,
            Image.new("RGB", (160, 20)),
            BoundingBox(0, 0, 160, 20),
            LineInitialTwoPlusTwoRecognizer(competitor_confidence=0.9),
        )
        == words
    )


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


class ConfirmedWrappedFourSyllableTripletRecognizer:
    def __init__(
        self,
        *,
        first_middle_text: str = '"\uac00\ub098\ub2e4\ub77c',
        first_middle_confidence: float = 0.9977,
        middle_last_text: str = '\ub098\ub2e4\ub77c"',
        middle_last_confidence: float = 0.9883,
        combined_text: str = '"\uac00\ub098\ub2e4\ub77c"',
        combined_confidence: float = 0.9959,
    ) -> None:
        self.values = (
            RecognizedText(first_middle_text, first_middle_confidence),
            RecognizedText(middle_last_text, middle_last_confidence),
            RecognizedText(combined_text, combined_confidence),
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def _wrapped_four_syllable_triplet_words():
    return [
        (
            "\uc774\uc804\ub9d0",
            BoundingBox(0, 0, 50, 20),
            0.999,
        ),
        (
            '"\uac00',
            BoundingBox(53, 0, 77, 20),
            0.9882,
        ),
        (
            "\ub098\ub2e4\ub77c",
            BoundingBox(76, 0, 129, 20),
            0.9791,
        ),
        (
            '"0',
            BoundingBox(128, 0, 139.8, 20),
            0.4337,
        ),
        (
            "\ub2e4\uc74c\ub9d0",
            BoundingBox(143.8, 0, 194, 20),
            0.999,
        ),
    ]


def test_confirmed_wrapped_four_syllable_triplet_recovers() -> None:
    words = _wrapped_four_syllable_triplet_words()

    recovered = _recover_confirmed_wrapped_four_syllable_triplet(
        words,
        Image.new("RGB", (200, 20)),
        BoundingBox(0, 0, 200, 20),
        ConfirmedWrappedFourSyllableTripletRecognizer(),
    )

    assert recovered == [
        words[0],
        (
            '"\uac00\ub098\ub2e4\ub77c"',
            BoundingBox(53, 0, 139.8, 20),
            0.9791,
        ),
        words[4],
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedWrappedFourSyllableTripletRecognizer(
            first_middle_text='"\uac00\ub098\ub2e4\ub9c8'
        ),
        ConfirmedWrappedFourSyllableTripletRecognizer(first_middle_confidence=0.9969),
        ConfirmedWrappedFourSyllableTripletRecognizer(middle_last_text='\ub098\ub2e4\ub9c8"'),
        ConfirmedWrappedFourSyllableTripletRecognizer(middle_last_confidence=0.9879),
        ConfirmedWrappedFourSyllableTripletRecognizer(combined_text="\"\uac00\ub098\ub2e4\ub77c'"),
        ConfirmedWrappedFourSyllableTripletRecognizer(combined_confidence=0.9949),
    ],
)
def test_confirmed_wrapped_four_syllable_triplet_requires_crop_agreement(
    recognizer,
) -> None:
    words = _wrapped_four_syllable_triplet_words()

    assert (
        _recover_confirmed_wrapped_four_syllable_triplet(
            words,
            Image.new("RGB", (200, 20)),
            BoundingBox(0, 0, 200, 20),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (
            1,
            (
                "\uac00\ub098",
                BoundingBox(53, 0, 77, 20),
                0.9882,
            ),
        ),
        (
            1,
            (
                '"\uac00',
                BoundingBox(53, 0, 77, 20),
                0.9879,
            ),
        ),
        (
            2,
            (
                "\ub098\ub2e4\ub77c",
                BoundingBox(76, 0, 129, 20),
                0.9789,
            ),
        ),
        (
            3,
            (
                '"0',
                BoundingBox(128, 0, 139.8, 20),
                0.4401,
            ),
        ),
        (
            1,
            (
                '"\uac00',
                BoundingBox(53, 0, 76.9, 20),
                0.9882,
            ),
        ),
        (
            0,
            (
                "\uc774\uc804\ub9d0",
                BoundingBox(0, 0, 50.2, 20),
                0.999,
            ),
        ),
        (
            4,
            (
                "\ub2e4\uc74c\ub9d0",
                BoundingBox(143.7, 0, 194, 20),
                0.999,
            ),
        ),
        (
            3,
            (
                '"0',
                BoundingBox(128, 0, 139.5, 20),
                0.4337,
            ),
        ),
    ],
)
def test_confirmed_wrapped_four_syllable_triplet_requires_word_profile(
    index,
    replacement,
) -> None:
    words = _wrapped_four_syllable_triplet_words()
    words[index] = replacement

    assert (
        _recover_confirmed_wrapped_four_syllable_triplet(
            words,
            Image.new("RGB", (200, 20)),
            BoundingBox(0, 0, 200, 20),
            ConfirmedWrappedFourSyllableTripletRecognizer(),
        )
        == words
    )


class RawWrapperEvidenceRecognizer:
    def __init__(self) -> None:
        self.values = (
            RecognizedText("\uc774\uc804\ub9d0", 0.6),
            RecognizedText("\uace0\uce5c\ub9d0", 0.999),
            RecognizedText('"\uac00', 0.9882),
            RecognizedText("\ub098\ub2e4\ub77c", 0.9791),
            RecognizedText("/0", 0.4337),
            RecognizedText("??", 0.8),
            RecognizedText("\ub2e4\uc74c\ub9d0", 0.999),
            RecognizedText('"\uac00\ub098\ub2e4\ub77c', 0.9977),
            RecognizedText("\ub098\ub2e4\ub77c/", 0.9883),
            RecognizedText('"\uac00\ub098\ub2e4\ub77c/', 0.9959),
        )
        self.calls = 0

    def word_boxes(self, _image):
        return (
            (0, 50),
            (53, 77),
            (76, 129),
            (128, 139.8),
            (143.8, 194),
        )

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


class RawWrapperEvidenceDetector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(0, 0, 200, 20), 0.99),)


def test_engine_uses_raw_wrapper_evidence_without_discarding_other_retry() -> None:
    recognizer = RawWrapperEvidenceRecognizer()
    engine = PaddleOcrEngine(RawWrapperEvidenceDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (200, 20)))

    assert [word.text for word in document.lines[0].eojeols] == [
        "\uace0\uce5c\ub9d0",
        "\uac00\ub098\ub2e4\ub77c",
        "\ub2e4\uc74c\ub9d0",
    ]
    assert recognizer.calls == len(recognizer.values)


class ConfirmedTerminalPunctuatedOverlapRecognizer:
    def __init__(
        self,
        *,
        combined_text: str = "\uac00\ub098\ub2e4:",
        combined_confidence: float = 0.7884,
        enhanced_text: str = "\uac00\ub098\ub2e4:",
        enhanced_confidence: float = 0.9586,
        padded_text: str = "\uac00\ub098\ub2e4:",
        padded_confidence: float = 0.9963,
    ) -> None:
        self.values = (
            RecognizedText(combined_text, combined_confidence),
            RecognizedText(enhanced_text, enhanced_confidence),
            RecognizedText(padded_text, padded_confidence),
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def _terminal_punctuated_overlap_words():
    return [
        ("\uc774\uc804", BoundingBox(0, 0, 50, 20), 0.999),
        ("\uac00", BoundingBox(55.68, 0, 70, 20), 0.9045),
        ("\ub098\ub2e4:", BoundingBox(69.054, 0, 116.28, 20), 0.4872),
        ("0", BoundingBox(116.28, 0, 126, 20), 0.2502),
    ]


def test_confirmed_terminal_punctuated_overlap_pair_recovers() -> None:
    words = _terminal_punctuated_overlap_words()

    recovered = _recover_confirmed_terminal_punctuated_overlap_pair(
        words,
        Image.new("RGB", (130, 20)),
        BoundingBox(0, 0, 130, 20),
        ConfirmedTerminalPunctuatedOverlapRecognizer(),
    )

    assert recovered == [
        words[0],
        (
            "\uac00\ub098\ub2e4:",
            BoundingBox(55.68, 0, 116.28, 20),
            0.4872,
        ),
        words[3],
    ]


@pytest.mark.parametrize(
    ("index", "replacement"),
    [
        (1, ("\uac00\ub098", BoundingBox(55.68, 0, 70, 20), 0.9045)),
        (2, ("\ub098\ub2e4\ub77c", BoundingBox(69.054, 0, 116.28, 20), 0.4872)),
        (3, ("\ub9c8", BoundingBox(116.28, 0, 126, 20), 0.2502)),
        (1, ("\uac00", BoundingBox(55.68, 0, 70, 20), 0.8999)),
        (2, ("\ub098\ub2e4:", BoundingBox(69.054, 0, 116.28, 20), 0.4799)),
        (3, ("0", BoundingBox(116.28, 0, 126, 20), 0.2399)),
        (1, ("\uac00", BoundingBox(55.68, 0, 69.98, 20), 0.9045)),
        (0, ("\uc774\uc804", BoundingBox(0, 0, 50.1, 20), 0.999)),
        (3, ("0", BoundingBox(116.32, 0, 126, 20), 0.2502)),
        (2, ("\ub098\ub2e4:", BoundingBox(69.054, 0, 116.5, 20), 0.4872)),
    ],
)
def test_confirmed_terminal_punctuated_overlap_pair_requires_profile(
    index,
    replacement,
) -> None:
    words = _terminal_punctuated_overlap_words()
    words[index] = replacement
    recognizer = ConfirmedTerminalPunctuatedOverlapRecognizer()

    assert (
        _recover_confirmed_terminal_punctuated_overlap_pair(
            words,
            Image.new("RGB", (130, 20)),
            BoundingBox(0, 0, 130, 20),
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedTerminalPunctuatedOverlapRecognizer(combined_text="\uac00\ub098\ub77c:"),
        ConfirmedTerminalPunctuatedOverlapRecognizer(combined_confidence=0.7879),
        ConfirmedTerminalPunctuatedOverlapRecognizer(enhanced_text="\uac00\ub098\ub77c:"),
        ConfirmedTerminalPunctuatedOverlapRecognizer(enhanced_confidence=0.9579),
        ConfirmedTerminalPunctuatedOverlapRecognizer(padded_text="\uac00\ub098\ub77c:"),
        ConfirmedTerminalPunctuatedOverlapRecognizer(padded_confidence=0.9959),
    ],
)
def test_confirmed_terminal_punctuated_overlap_pair_requires_crop_agreement(
    recognizer,
) -> None:
    words = _terminal_punctuated_overlap_words()

    assert (
        _recover_confirmed_terminal_punctuated_overlap_pair(
            words,
            Image.new("RGB", (130, 20)),
            BoundingBox(0, 0, 130, 20),
            recognizer,
        )
        == words
    )


class RawTerminalPunctuatedOverlapRecognizer:
    def __init__(self) -> None:
        self.values = (
            RecognizedText("\uc774\uc804", 0.999),
            RecognizedText("\uac00", 0.9045),
            RecognizedText("\ub098\ub2e4:", 0.4872),
            RecognizedText("\ub098\ub2e4", 0.99),
            RecognizedText("0", 0.2502),
            RecognizedText("", 0.99),
            RecognizedText("\uac00\ub098\ub2e4:", 0.7884),
            RecognizedText("\uac00\ub098\ub2e4:", 0.9586),
            RecognizedText("\uac00\ub098\ub2e4:", 0.9963),
        )
        self.calls = 0

    def word_boxes(self, _image):
        return (
            (0, 50),
            (55.68, 70),
            (69.054, 116.28),
            (116.28, 126),
        )

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


class RawTerminalPunctuatedOverlapDetector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(0, 0, 130, 20), 0.99),)


def test_engine_uses_raw_terminal_punctuated_overlap_evidence() -> None:
    recognizer = RawTerminalPunctuatedOverlapRecognizer()
    engine = PaddleOcrEngine(RawTerminalPunctuatedOverlapDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (130, 20)))

    assert [word.text for word in document.lines[0].eojeols] == [
        "\uc774\uc804",
        "\uac00\ub098\ub2e4",
    ]
    assert recognizer.calls == len(recognizer.values)


_TERMINAL_WRAPPED_FOUR_HEIGHT = 14.08695652173913
_TERMINAL_WRAPPED_FOUR_LINE = BoundingBox(
    86.96, 0, 624.04, _TERMINAL_WRAPPED_FOUR_HEIGHT
)
_TERMINAL_WRAPPED_FOUR_SELECTED_INDEXES = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
)


def _terminal_wrapped_four_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


_TERMINAL_WRAPPED_FOUR_DIRECT = _terminal_wrapped_four_hangul(0xC900, 4)
_TERMINAL_WRAPPED_FOUR_RETRY = _terminal_wrapped_four_hangul(0xC910, 3)
_TERMINAL_WRAPPED_FOUR_RECOVERED = _terminal_wrapped_four_hangul(0xC920, 4)
_TERMINAL_WRAPPED_FOUR_PREFIX = _terminal_wrapped_four_hangul(0xC930, 3)


def terminal_wrapped_four_words() -> tuple[
    list[tuple[str, BoundingBox, float]],
    list[tuple[str, BoundingBox, float]],
]:
    boxes = (
        BoundingBox(121.96, 0, 213.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(217.96, 0, 263.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(266.96, 0, 290.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(289.96, 0, 304.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(307.96, 0, 320.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(323.96, 0, 335.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(338.96, 0, 350.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(349.96, 0, 389.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(392.96, 0, 415.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(418.96, 0, 453.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(456.96, 0, 468.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(471.96, 0, 494.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
        BoundingBox(498.96, 0, 589.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
    )
    texts = (
        _terminal_wrapped_four_hangul(0xC940, 8),
        _terminal_wrapped_four_hangul(0xC948, 4),
        _terminal_wrapped_four_hangul(0xC94C, 2),
        _terminal_wrapped_four_hangul(0xC94E, 1),
        "12",
        ",",
        _terminal_wrapped_four_hangul(0xC94F, 1),
        _terminal_wrapped_four_hangul(0xC950, 2) + "A1",
        _terminal_wrapped_four_hangul(0xC952, 2),
        _terminal_wrapped_four_hangul(0xC954, 3),
        _terminal_wrapped_four_hangul(0xC957, 1),
        _terminal_wrapped_four_hangul(0xC958, 2),
        (
            _TERMINAL_WRAPPED_FOUR_PREFIX
            + "("
            + _TERMINAL_WRAPPED_FOUR_DIRECT
            + ")"
        ),
    )
    confidences = (
        0.876248,
        0.997898,
        0.999492,
        0.961724,
        0.997670,
        0.955685,
        0.999466,
        0.999472,
        0.986963,
        0.501630,
        0.999793,
        0.999458,
        0.976379,
    )
    raw = list(zip(texts, boxes, confidences, strict=True))
    return terminal_wrapped_four_selected(raw), raw


def terminal_wrapped_four_selected(
    raw: list[tuple[str, BoundingBox, float]],
) -> list[tuple[str, BoundingBox, float]]:
    selected = [raw[index] for index in _TERMINAL_WRAPPED_FOUR_SELECTED_INDEXES]
    selected[8] = (_TERMINAL_WRAPPED_FOUR_RETRY, selected[8][1], 0.829677)
    return selected


_TERMINAL_WRAPPED_FOUR_DIRECT_CONFIDENCES = (
    0.823365,
    0.808633,
    0.788579,
    0.786463,
    0.556883,
    0.550508,
    0.531670,
)
_TERMINAL_WRAPPED_FOUR_ENHANCED_CONFIDENCES = (
    0.800378,
    0.786039,
    0.781194,
    0.746324,
    0.739568,
    0.724976,
    0.650494,
)


class ConfirmedTerminalWrappedFourRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_TERMINAL_WRAPPED_FOUR_RECOVERED,) * 7,
        direct_confidences: tuple[float, ...] = (
            _TERMINAL_WRAPPED_FOUR_DIRECT_CONFIDENCES
        ),
        enhanced_texts: tuple[str, ...] = (_TERMINAL_WRAPPED_FOUR_RECOVERED,) * 7,
        enhanced_confidences: tuple[float, ...] = (
            _TERMINAL_WRAPPED_FOUR_ENHANCED_CONFIDENCES
        ),
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_terminal_wrapped_four_substitution_recovers_interior() -> None:
    selected, raw = terminal_wrapped_four_words()
    recognizer = ConfirmedTerminalWrappedFourRecognizer()

    recovered = _recover_confirmed_terminal_wrapped_four_substitution(
        selected,
        raw,
        Image.new("RGB", (539, 15)),
        _TERMINAL_WRAPPED_FOUR_LINE,
        recognizer,
    )

    expected = list(selected)
    expected[-1] = (
        (
            _TERMINAL_WRAPPED_FOUR_PREFIX
            + "("
            + _TERMINAL_WRAPPED_FOUR_RECOVERED
            + ")"
        ),
        selected[-1][1],
        0.531670,
    )
    assert recovered == expected
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "case",
    [
        "direct-disagreement",
        "direct-confidence",
        "enhanced-disagreement",
        "enhanced-confidence",
        "non-hangul",
        "unchanged-interior",
    ],
)
def test_confirmed_terminal_wrapped_four_substitution_requires_consensus(
    case: str,
) -> None:
    direct_texts = [_TERMINAL_WRAPPED_FOUR_RECOVERED] * 7
    direct_confidences = list(_TERMINAL_WRAPPED_FOUR_DIRECT_CONFIDENCES)
    enhanced_texts = [_TERMINAL_WRAPPED_FOUR_RECOVERED] * 7
    enhanced_confidences = list(_TERMINAL_WRAPPED_FOUR_ENHANCED_CONFIDENCES)
    if case == "direct-disagreement":
        direct_texts[2] = _TERMINAL_WRAPPED_FOUR_DIRECT
    elif case == "direct-confidence":
        direct_confidences[0] = 0.8232
    elif case == "enhanced-disagreement":
        enhanced_texts[3] = _TERMINAL_WRAPPED_FOUR_DIRECT
    elif case == "enhanced-confidence":
        enhanced_confidences[-1] = 0.6503
    elif case == "non-hangul":
        direct_texts = ["ABCD"] * 7
        enhanced_texts = ["ABCD"] * 7
    else:
        direct_texts = [_TERMINAL_WRAPPED_FOUR_DIRECT] * 7
        enhanced_texts = [_TERMINAL_WRAPPED_FOUR_DIRECT] * 7
    recognizer = ConfirmedTerminalWrappedFourRecognizer(
        direct_texts=tuple(direct_texts),
        direct_confidences=tuple(direct_confidences),
        enhanced_texts=tuple(enhanced_texts),
        enhanced_confidences=tuple(enhanced_confidences),
    )
    selected, raw = terminal_wrapped_four_words()

    assert (
        _recover_confirmed_terminal_wrapped_four_substitution(
            selected,
            raw,
            Image.new("RGB", (539, 15)),
            _TERMINAL_WRAPPED_FOUR_LINE,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "ordinary-mismatch",
        "retry-text",
        "retry-box",
        "retry-confidence",
        "raw-shape",
        "raw-confidence",
        "target-width",
        "target-gap",
        "candidate-shape",
        "candidate-wrapper",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_terminal_wrapped_four_substitution_requires_exact_profile(
    case: str,
) -> None:
    selected, raw = terminal_wrapped_four_words()
    crop = Image.new("RGB", (539, 15))
    line_box = _TERMINAL_WRAPPED_FOUR_LINE
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "ordinary-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.8761)
    elif case == "retry-text":
        selected[8] = raw[9]
    elif case == "retry-box":
        selected[8] = (
            selected[8][0],
            BoundingBox(419.96, 0, 453.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
            selected[8][2],
        )
    elif case == "retry-confidence":
        selected[8] = (selected[8][0], selected[8][1], 0.8294)
    elif case == "raw-shape":
        raw[0] = (raw[0][0][:-1] + "A", raw[0][1], raw[0][2])
        selected = terminal_wrapped_four_selected(raw)
    elif case == "raw-confidence":
        raw[0] = (raw[0][0], raw[0][1], 0.8761)
        selected = terminal_wrapped_four_selected(raw)
    elif case == "target-width":
        raw[-1] = (
            raw[-1][0],
            BoundingBox(498.96, 0, 588.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
            raw[-1][2],
        )
        selected = terminal_wrapped_four_selected(raw)
    elif case == "target-gap":
        raw[-1] = (
            raw[-1][0],
            BoundingBox(497.96, 0, 589.96, _TERMINAL_WRAPPED_FOUR_HEIGHT),
            raw[-1][2],
        )
        selected = terminal_wrapped_four_selected(raw)
    elif case == "candidate-shape":
        raw[-1] = (raw[-1][0][:-2] + "A)", raw[-1][1], raw[-1][2])
        selected = terminal_wrapped_four_selected(raw)
    elif case == "candidate-wrapper":
        raw[-1] = (
            raw[-1][0][:3] + "[" + raw[-1][0][4:],
            raw[-1][1],
            raw[-1][2],
        )
        selected = terminal_wrapped_four_selected(raw)
    elif case == "line-height":
        line_box = BoundingBox(86.96, 0, 624.04, 0)
    else:
        crop = Image.new("RGB", (490, 15))
    recognizer = ConfirmedTerminalWrappedFourRecognizer()

    assert (
        _recover_confirmed_terminal_wrapped_four_substitution(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class TerminalWrappedFourEngineRecognizer:
    def __init__(self) -> None:
        selected, raw = terminal_wrapped_four_words()
        initial = [RecognizedText("", 0.0), RecognizedText("", 0.0)]
        for index, (value, _box, confidence) in enumerate(raw):
            initial.append(RecognizedText(value, confidence))
            if index == 9:
                initial.append(RecognizedText(selected[8][0], selected[8][2]))
        self.values = tuple(initial) + ConfirmedTerminalWrappedFourRecognizer().values
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (0, 27),
            (35, 127),
            (131, 177),
            (180, 204),
            (203, 218),
            (221, 234),
            (237, 249),
            (252, 264),
            (263, 303),
            (306, 329),
            (332, 367),
            (370, 382),
            (385, 408),
            (412, 503),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class TerminalWrappedFourEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    86.96,
                    153.391304,
                    624.04,
                    167.47826052173913,
                ),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_terminal_wrapped_four_substitution() -> None:
    recognizer = TerminalWrappedFourEngineRecognizer()
    engine = PaddleOcrEngine(TerminalWrappedFourEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[-1]
    assert target.text == _TERMINAL_WRAPPED_FOUR_RECOVERED
    assert target.box == BoundingBox(
        539.4044444444444,
        153.391304,
        579.848888888889,
        167.47826052173913,
    )
    assert target.confidence == 0.531670


_PUNCTUATION_TRIMMED_HEIGHT = 26.413043478260875
_PUNCTUATION_TRIMMED_LINE = BoundingBox(
    58.16, 0, 1070.84, _PUNCTUATION_TRIMMED_HEIGHT
)


def _punctuation_trimmed_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


_PUNCTUATION_TRIMMED_TARGET = _punctuation_trimmed_hangul(0xC800, 1)
_PUNCTUATION_TRIMMED_OTHER = _punctuation_trimmed_hangul(0xC810, 1)


def punctuation_trimmed_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(121.16, 0, 164.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(173.16, 0, 239.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(248.16, 0, 340.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(346.16, 0, 415.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(414.16, 0, 451.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(461.16, 0, 483.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(492.16, 0, 535.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(543.16, 0, 610.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(618.16, 0, 708.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(716.16, 0, 783.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(790.16, 0, 881.16, _PUNCTUATION_TRIMMED_HEIGHT),
        BoundingBox(888.16, 0, 1007.16, _PUNCTUATION_TRIMMED_HEIGHT),
    )
    texts = (
        _punctuation_trimmed_hangul(0xC820, 2),
        _punctuation_trimmed_hangul(0xC822, 3),
        _punctuation_trimmed_hangul(0xC825, 4),
        _punctuation_trimmed_hangul(0xC829, 3),
        _PUNCTUATION_TRIMMED_TARGET + ".",
        _punctuation_trimmed_hangul(0xC82C, 1),
        _punctuation_trimmed_hangul(0xC82D, 2),
        _punctuation_trimmed_hangul(0xC82F, 3),
        _punctuation_trimmed_hangul(0xC832, 4),
        _punctuation_trimmed_hangul(0xC836, 3),
        _punctuation_trimmed_hangul(0xC839, 4),
        _punctuation_trimmed_hangul(0xC83D, 5) + ".",
    )
    confidences = (
        0.999867,
        0.999846,
        0.998404,
        0.999819,
        0.902827,
        0.999919,
        0.999416,
        0.999855,
        0.999052,
        0.999715,
        0.999966,
        0.982917,
    )
    return list(zip(texts, boxes, confidences, strict=True))


_PUNCTUATION_TRIMMED_DIRECT_CONFIDENCES = (
    0.999933,
    0.999929,
    0.999919,
    0.999913,
    0.999882,
    0.999849,
    0.999868,
)
_PUNCTUATION_TRIMMED_ENHANCED_CONFIDENCES = (
    0.999913,
    0.999889,
    0.999893,
    0.999920,
    0.999812,
    0.999787,
    0.999821,
)


class ConfirmedPunctuationTrimmedRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_PUNCTUATION_TRIMMED_TARGET,) * 7,
        direct_confidences: tuple[float, ...] = (
            _PUNCTUATION_TRIMMED_DIRECT_CONFIDENCES
        ),
        enhanced_texts: tuple[str, ...] = (_PUNCTUATION_TRIMMED_TARGET,) * 7,
        enhanced_confidences: tuple[float, ...] = (
            _PUNCTUATION_TRIMMED_ENHANCED_CONFIDENCES
        ),
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_punctuation_trimmed_single_recovers_geometry() -> None:
    words = punctuation_trimmed_words()
    recognizer = ConfirmedPunctuationTrimmedRecognizer()

    recovered = _recover_confirmed_punctuation_trimmed_single(
        words,
        words,
        Image.new("RGB", (1013, 28)),
        _PUNCTUATION_TRIMMED_LINE,
        recognizer,
    )

    expected = list(words)
    expected[4] = (
        _PUNCTUATION_TRIMMED_TARGET,
        BoundingBox(420.16, 0, 445.16, _PUNCTUATION_TRIMMED_HEIGHT),
        0.902827,
    )
    assert recovered == expected
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "case",
    [
        "direct-disagreement",
        "direct-confidence",
        "enhanced-disagreement",
        "enhanced-confidence",
        "non-hangul",
        "candidate-disagreement",
    ],
)
def test_confirmed_punctuation_trimmed_single_requires_crop_consensus(
    case: str,
) -> None:
    direct_texts = [_PUNCTUATION_TRIMMED_TARGET] * 7
    direct_confidences = list(_PUNCTUATION_TRIMMED_DIRECT_CONFIDENCES)
    enhanced_texts = [_PUNCTUATION_TRIMMED_TARGET] * 7
    enhanced_confidences = list(_PUNCTUATION_TRIMMED_ENHANCED_CONFIDENCES)
    words = punctuation_trimmed_words()
    if case == "direct-disagreement":
        direct_texts[2] = _PUNCTUATION_TRIMMED_OTHER
    elif case == "direct-confidence":
        direct_confidences[0] = 0.9998
    elif case == "enhanced-disagreement":
        enhanced_texts[3] = _PUNCTUATION_TRIMMED_OTHER
    elif case == "enhanced-confidence":
        enhanced_confidences[0] = 0.9998
    elif case == "non-hangul":
        direct_texts = ["A"] * 7
        enhanced_texts = ["A"] * 7
    else:
        words[4] = (
            _PUNCTUATION_TRIMMED_OTHER + ".",
            words[4][1],
            words[4][2],
        )
    recognizer = ConfirmedPunctuationTrimmedRecognizer(
        direct_texts=tuple(direct_texts),
        direct_confidences=tuple(direct_confidences),
        enhanced_texts=tuple(enhanced_texts),
        enhanced_confidences=tuple(enhanced_confidences),
    )

    assert (
        _recover_confirmed_punctuation_trimmed_single(
            words,
            words,
            Image.new("RGB", (1013, 28)),
            _PUNCTUATION_TRIMMED_LINE,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "shape",
        "punctuation",
        "confidence",
        "width",
        "gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_punctuation_trimmed_single_requires_exact_profile(
    case: str,
) -> None:
    words = punctuation_trimmed_words()
    raw = list(words)
    crop = Image.new("RGB", (1013, 28))
    line_box = _PUNCTUATION_TRIMMED_LINE
    if case == "selected-count":
        words.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        words[0] = (words[0][0], words[0][1], 0.9998)
    elif case == "shape":
        words[0] = raw[0] = (raw[0][0][:-1] + "A", raw[0][1], raw[0][2])
    elif case == "punctuation":
        words[4] = raw[4] = (
            _PUNCTUATION_TRIMMED_TARGET + "A",
            raw[4][1],
            raw[4][2],
        )
    elif case == "confidence":
        words[4] = raw[4] = (raw[4][0], raw[4][1], 0.9026)
    elif case == "width":
        box = BoundingBox(414.16, 0, 450.16, _PUNCTUATION_TRIMMED_HEIGHT)
        words[4] = raw[4] = (raw[4][0], box, raw[4][2])
    elif case == "gap":
        box = BoundingBox(462.16, 0, 483.16, _PUNCTUATION_TRIMMED_HEIGHT)
        words[5] = raw[5] = (raw[5][0], box, raw[5][2])
    elif case == "line-height":
        line_box = BoundingBox(58.16, 0, 1070.84, 0)
    else:
        crop = Image.new("RGB", (380, 28))
    recognizer = ConfirmedPunctuationTrimmedRecognizer()

    assert (
        _recover_confirmed_punctuation_trimmed_single(
            words,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class PunctuationTrimmedEngineRecognizer:
    def __init__(self) -> None:
        initial = [
            RecognizedText(text, confidence)
            for text, _box, confidence in punctuation_trimmed_words()
        ]
        initial.extend((RecognizedText("", 0.0), RecognizedText("", 0.0)))
        self.values = (
            tuple(initial) + ConfirmedPunctuationTrimmedRecognizer().values
        )
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (63, 106),
            (115, 181),
            (190, 282),
            (288, 357),
            (356, 393),
            (403, 425),
            (434, 477),
            (485, 552),
            (560, 650),
            (658, 725),
            (732, 823),
            (830, 949),
            (958, 1013),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class PunctuationTrimmedEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    58.16,
                    152.6086956521739,
                    1070.84,
                    179.02173913043478,
                ),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_punctuation_trimmed_single() -> None:
    recognizer = PunctuationTrimmedEngineRecognizer()
    engine = PaddleOcrEngine(PunctuationTrimmedEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[4]
    assert target.text == _PUNCTUATION_TRIMMED_TARGET
    assert target.box == BoundingBox(
        420.15999999999997,
        152.6086956521739,
        445.15999999999997,
        179.02173913043478,
    )
    assert target.confidence == 0.902827


_WRAPPED_SINGLE_HEIGHT = 21.13043478260869
_WRAPPED_SINGLE_LINE = BoundingBox(78.84, 0, 748.16, _WRAPPED_SINGLE_HEIGHT)
_WRAPPED_SINGLE_TARGET = "".join(chr(0xC700 + offset) for offset in range(1))
_WRAPPED_SINGLE_OTHER = "".join(chr(0xC710 + offset) for offset in range(1))


def _wrapped_single_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def wrapped_single_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(78.84, 0, 103.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(120.84, 0, 180.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(202.84, 0, 234.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(254.84, 0, 269.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(278.84, 0, 358.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(366.84, 0, 422.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(430.84, 0, 487.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(498.84, 0, 616.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(622.84, 0, 706.84, _WRAPPED_SINGLE_HEIGHT),
        BoundingBox(712.84, 0, 749.84, _WRAPPED_SINGLE_HEIGHT),
    )
    texts = (
        "1",
        _wrapped_single_hangul(0xC720, 3),
        "/" + _WRAPPED_SINGLE_TARGET + "/",
        _wrapped_single_hangul(0xC723, 1),
        _wrapped_single_hangul(0xC724, 4),
        _wrapped_single_hangul(0xC728, 3),
        _wrapped_single_hangul(0xC72B, 3),
        _wrapped_single_hangul(0xC72E, 6),
        _wrapped_single_hangul(0xC734, 4) + ".",
        "2",
    )
    confidences = (
        0.172596,
        0.999956,
        0.679847,
        0.999768,
        0.999707,
        0.999937,
        0.999934,
        0.997799,
        0.952186,
        0.261037,
    )
    return list(zip(texts, boxes, confidences, strict=True))


_WRAPPED_SINGLE_SELECTED_INDEXES = (1, 2, 3, 4, 5, 6, 7, 8)


def wrapped_single_selected_words(
    raw: list[tuple[str, BoundingBox, float]],
) -> list[tuple[str, BoundingBox, float]]:
    selected = [raw[index] for index in _WRAPPED_SINGLE_SELECTED_INDEXES]
    selected[1] = (
        _WRAPPED_SINGLE_TARGET + "/",
        selected[1][1],
        0.994793,
    )
    return selected


_WRAPPED_SINGLE_DIRECT_CONFIDENCES = (
    0.999510,
    0.999361,
    0.999388,
    0.999338,
    0.999310,
    0.999322,
    0.998700,
)
_WRAPPED_SINGLE_ENHANCED_CONFIDENCES = (
    0.999561,
    0.999450,
    0.999413,
    0.999396,
    0.999272,
    0.999231,
    0.999173,
)


class ConfirmedWrappedSingleRecognizer:
    def __init__(
        self,
        *,
        direct_texts: tuple[str, ...] = (_WRAPPED_SINGLE_TARGET,) * 7,
        direct_confidences: tuple[float, ...] = _WRAPPED_SINGLE_DIRECT_CONFIDENCES,
        enhanced_texts: tuple[str, ...] = (_WRAPPED_SINGLE_TARGET,) * 7,
        enhanced_confidences: tuple[float, ...] = (
            _WRAPPED_SINGLE_ENHANCED_CONFIDENCES
        ),
    ) -> None:
        self.values = tuple(
            RecognizedText(text, confidence)
            for text, confidence in zip(
                (*direct_texts, *enhanced_texts),
                (*direct_confidences, *enhanced_confidences),
                strict=True,
            )
        )
        self.calls = 0

    def recognize(self, _image):
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_wrapped_single_geometry_recovers_box() -> None:
    raw = wrapped_single_raw_words()
    selected = wrapped_single_selected_words(raw)
    recognizer = ConfirmedWrappedSingleRecognizer()

    recovered = _recover_confirmed_wrapped_single_geometry(
        selected,
        raw,
        Image.new("RGB", (671, 23)),
        _WRAPPED_SINGLE_LINE,
        recognizer,
    )

    expected = list(selected)
    expected[1] = (
        _WRAPPED_SINGLE_TARGET,
        BoundingBox(205.84, 0, 227.84, _WRAPPED_SINGLE_HEIGHT),
        0.994793,
    )
    assert recovered == expected
    assert recognizer.calls == 14


@pytest.mark.parametrize(
    "case",
    [
        "direct-disagreement",
        "direct-confidence",
        "enhanced-disagreement",
        "enhanced-confidence",
        "non-hangul",
        "candidate-disagreement",
    ],
)
def test_confirmed_wrapped_single_geometry_requires_crop_consensus(
    case: str,
) -> None:
    direct_texts = [_WRAPPED_SINGLE_TARGET] * 7
    direct_confidences = list(_WRAPPED_SINGLE_DIRECT_CONFIDENCES)
    enhanced_texts = [_WRAPPED_SINGLE_TARGET] * 7
    enhanced_confidences = list(_WRAPPED_SINGLE_ENHANCED_CONFIDENCES)
    raw = wrapped_single_raw_words()
    selected = wrapped_single_selected_words(raw)
    if case == "direct-disagreement":
        direct_texts[2] = _WRAPPED_SINGLE_OTHER
    elif case == "direct-confidence":
        direct_confidences[0] = 0.9994
    elif case == "enhanced-disagreement":
        enhanced_texts[3] = _WRAPPED_SINGLE_OTHER
    elif case == "enhanced-confidence":
        enhanced_confidences[0] = 0.9994
    elif case == "non-hangul":
        direct_texts = ["A"] * 7
        enhanced_texts = ["A"] * 7
    else:
        selected[1] = (
            _WRAPPED_SINGLE_OTHER + "/",
            selected[1][1],
            selected[1][2],
        )
    recognizer = ConfirmedWrappedSingleRecognizer(
        direct_texts=tuple(direct_texts),
        direct_confidences=tuple(direct_confidences),
        enhanced_texts=tuple(enhanced_texts),
        enhanced_confidences=tuple(enhanced_confidences),
    )

    assert (
        _recover_confirmed_wrapped_single_geometry(
            selected,
            raw,
            Image.new("RGB", (671, 23)),
            _WRAPPED_SINGLE_LINE,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == (0 if case == "candidate-disagreement" else 14)


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "ordinary-mismatch",
        "candidate-box",
        "raw-shape",
        "selected-shape",
        "wrapper",
        "candidate-relationship",
        "raw-confidence",
        "selected-confidence",
        "width",
        "gap",
        "line-height",
        "crop-bounds",
    ],
)
def test_confirmed_wrapped_single_geometry_requires_exact_profile(
    case: str,
) -> None:
    raw = wrapped_single_raw_words()
    selected = wrapped_single_selected_words(raw)
    crop = Image.new("RGB", (671, 23))
    line_box = _WRAPPED_SINGLE_LINE
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "ordinary-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.9998)
    elif case == "candidate-box":
        selected[1] = (
            selected[1][0],
            BoundingBox(203.84, 0, 234.84, _WRAPPED_SINGLE_HEIGHT),
            selected[1][2],
        )
    elif case == "raw-shape":
        raw[0] = ("A1", raw[0][1], raw[0][2])
        selected = wrapped_single_selected_words(raw)
    elif case == "selected-shape":
        selected[1] = (
            _WRAPPED_SINGLE_TARGET + "A",
            selected[1][1],
            selected[1][2],
        )
    elif case == "wrapper":
        raw[2] = ("(" + _WRAPPED_SINGLE_TARGET + "/", raw[2][1], raw[2][2])
        selected = wrapped_single_selected_words(raw)
    elif case == "candidate-relationship":
        raw[2] = (
            "/" + _WRAPPED_SINGLE_OTHER + "/",
            raw[2][1],
            raw[2][2],
        )
    elif case == "raw-confidence":
        raw[2] = (raw[2][0], raw[2][1], 0.6796)
    elif case == "selected-confidence":
        selected[1] = (selected[1][0], selected[1][1], 0.9946)
    elif case == "width":
        raw[2] = (
            raw[2][0],
            BoundingBox(202.84, 0, 233.84, _WRAPPED_SINGLE_HEIGHT),
            raw[2][2],
        )
        selected = wrapped_single_selected_words(raw)
    elif case == "gap":
        raw[3] = (
            raw[3][0],
            BoundingBox(253.84, 0, 269.84, _WRAPPED_SINGLE_HEIGHT),
            raw[3][2],
        )
        selected = wrapped_single_selected_words(raw)
    elif case == "line-height":
        line_box = BoundingBox(78.84, 0, 748.16, 0)
    else:
        crop = Image.new("RGB", (148, 23))
    recognizer = ConfirmedWrappedSingleRecognizer()

    assert (
        _recover_confirmed_wrapped_single_geometry(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class WrappedSingleEngineRecognizer:
    def __init__(self) -> None:
        raw = wrapped_single_raw_words()
        selected = wrapped_single_selected_words(raw)
        retry_values = {
            0: RecognizedText("1", 0.2),
            2: RecognizedText("", 0.0),
            3: RecognizedText(selected[1][0], selected[1][2]),
            4: RecognizedText("", 0.0),
            11: RecognizedText("2", 0.3),
        }
        raw_by_segment = {
            0: raw[0],
            1: raw[1],
            3: raw[2],
            5: raw[3],
            6: raw[4],
            7: raw[5],
            8: raw[6],
            9: raw[7],
            10: raw[8],
            11: raw[9],
        }
        values = []
        for index in range(12):
            if index in raw_by_segment:
                text, _box, confidence = raw_by_segment[index]
                values.append(RecognizedText(text, confidence))
            else:
                values.append(RecognizedText("", 0.0))
            if values[-1].confidence < 0.72:
                values.append(retry_values[index])
        self.values = tuple(values) + ConfirmedWrappedSingleRecognizer().values
        self.calls = 0

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold != 0.07:
            return ((0, _image.width),)
        return (
            (0, 25),
            (42, 102),
            (109, 125),
            (124, 156),
            (155, 167),
            (176, 191),
            (200, 280),
            (288, 344),
            (352, 409),
            (420, 538),
            (544, 628),
            (634, 671),
        )

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


class WrappedSingleEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    78.84,
                    148.8913043478261,
                    748.16,
                    170.02173913043478,
                ),
                0.99,
            ),
        )


def test_engine_recovers_confirmed_wrapped_single_geometry() -> None:
    recognizer = WrappedSingleEngineRecognizer()
    engine = PaddleOcrEngine(WrappedSingleEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[1]
    assert target.text == _WRAPPED_SINGLE_TARGET
    assert target.box == BoundingBox(
        205.84,
        148.8913043478261,
        227.84,
        170.02173913043478,
    )
    assert target.confidence == 0.994793


_LEADING_PUNCTUATED_HEIGHT = 24.65
_LEADING_PUNCTUATED_LINE = BoundingBox(
    48.72,
    0,
    1198.28,
    _LEADING_PUNCTUATED_HEIGHT,
)
_LEADING_PUNCTUATED_TARGET = chr(0xC780)
_LEADING_PUNCTUATED_OTHER = chr(0xC790)
_LEADING_PUNCTUATED_FOLLOWING = "".join(chr(0xC7A0 + offset) for offset in range(2))


def _leading_punctuated_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def leading_punctuated_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(121.72, 0, 194.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(201.72, 0, 280.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(289.72, 0, 336.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(344.72, 0, 541.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(550.72, 0, 599.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(607.72, 0, 737.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(747.72, 0, 794.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(803.72, 0, 888.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(897.72, 0, 1018.72, _LEADING_PUNCTUATED_HEIGHT),
        BoundingBox(1029.72, 0, 1124.72, _LEADING_PUNCTUATED_HEIGHT),
    )
    texts = (
        _leading_punctuated_hangul(0xC7B0, 3),
        _leading_punctuated_hangul(0xC7B3, 2) + "12",
        _leading_punctuated_hangul(0xC7B5, 2),
        _leading_punctuated_hangul(0xC7B7, 8),
        _leading_punctuated_hangul(0xC7BF, 2),
        _leading_punctuated_hangul(0xC7C1, 3) + "1234",
        _leading_punctuated_hangul(0xC7C4, 2),
        _LEADING_PUNCTUATED_TARGET + ":" + _LEADING_PUNCTUATED_FOLLOWING,
        _leading_punctuated_hangul(0xC7C6, 5),
        _leading_punctuated_hangul(0xC7CB, 4),
    )
    confidences = (
        0.999835,
        0.999621,
        0.999588,
        0.999771,
        0.999843,
        0.998224,
        0.999934,
        0.998760,
        0.999867,
        0.999797,
    )
    return list(zip(texts, boxes, confidences, strict=True))


_LEADING_TARGET_DIRECT_CONFIDENCES = (
    0.999988,
    0.999987,
    0.999987,
    0.999987,
    0.999986,
    0.999985,
    0.999986,
)
_LEADING_TARGET_ENHANCED_CONFIDENCES = (
    0.999990,
    0.999990,
    0.999988,
    0.999990,
    0.999987,
    0.999988,
    0.999985,
)
_LEADING_FOLLOWING_DIRECT_CONFIDENCES = (
    0.999320,
    0.999801,
    0.999802,
    0.999755,
    0.999750,
    0.999729,
    0.999778,
    0.999725,
)
_LEADING_FOLLOWING_ENHANCED_CONFIDENCES = (
    0.999308,
    0.999781,
    0.999771,
    0.999779,
    0.999773,
    0.999756,
    0.999728,
    0.999753,
)


class ConfirmedLeadingPunctuatedRecognizer:
    def __init__(
        self,
        *,
        boundary_direct_text: str | None = None,
        boundary_direct_confidence: float = 0.994520,
        boundary_enhanced_text: str | None = None,
        boundary_enhanced_confidence: float = 0.995141,
        target_direct_texts: tuple[str, ...] = (
            (_LEADING_PUNCTUATED_TARGET,) * 7
        ),
        target_direct_confidences: tuple[float, ...] = (
            _LEADING_TARGET_DIRECT_CONFIDENCES
        ),
        target_enhanced_texts: tuple[str, ...] = (
            (_LEADING_PUNCTUATED_TARGET,) * 7
        ),
        target_enhanced_confidences: tuple[float, ...] = (
            _LEADING_TARGET_ENHANCED_CONFIDENCES
        ),
        following_direct_texts: tuple[str, ...] = (
            (_LEADING_PUNCTUATED_FOLLOWING,) * 8
        ),
        following_direct_confidences: tuple[float, ...] = (
            _LEADING_FOLLOWING_DIRECT_CONFIDENCES
        ),
        following_enhanced_texts: tuple[str, ...] = (
            (_LEADING_PUNCTUATED_FOLLOWING,) * 8
        ),
        following_enhanced_confidences: tuple[float, ...] = (
            _LEADING_FOLLOWING_ENHANCED_CONFIDENCES
        ),
        boundary_mismatch: bool = False,
        default_mismatch: bool = False,
    ) -> None:
        boundary_text = _LEADING_PUNCTUATED_TARGET + ":"
        values = [
            RecognizedText(
                boundary_direct_text
                if boundary_direct_text is not None
                else boundary_text,
                boundary_direct_confidence,
            ),
            RecognizedText(
                boundary_enhanced_text
                if boundary_enhanced_text is not None
                else boundary_text,
                boundary_enhanced_confidence,
            ),
        ]
        for direct_text, direct_confidence, enhanced_text, enhanced_confidence in zip(
            target_direct_texts,
            target_direct_confidences,
            target_enhanced_texts,
            target_enhanced_confidences,
            strict=True,
        ):
            values.extend(
                (
                    RecognizedText(direct_text, direct_confidence),
                    RecognizedText(enhanced_text, enhanced_confidence),
                )
            )
        for direct_text, direct_confidence, enhanced_text, enhanced_confidence in zip(
            following_direct_texts,
            following_direct_confidences,
            following_enhanced_texts,
            following_enhanced_confidences,
            strict=True,
        ):
            values.extend(
                (
                    RecognizedText(direct_text, direct_confidence),
                    RecognizedText(enhanced_text, enhanced_confidence),
                )
            )
        self.values = tuple(values)
        self.calls = 0
        self.word_box_thresholds: list[float] = []
        self.boundary_mismatch = boundary_mismatch
        self.default_mismatch = default_mismatch

    def word_boxes(self, _image, space_threshold: float = 0.07):
        self.word_box_thresholds.append(space_threshold)
        if space_threshold == 0.05:
            return ((0, 30), (38, 85)) if self.default_mismatch else ((0, 85),)
        if self.boundary_mismatch and space_threshold == 0.02:
            return ((0, 85),)
        return ((0, 30), (38, 85))

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_leading_punctuated_single_split_recovers_boundary() -> None:
    raw = leading_punctuated_raw_words()
    recognizer = ConfirmedLeadingPunctuatedRecognizer()

    recovered = _recover_confirmed_leading_punctuated_single_split(
        list(raw),
        raw,
        Image.new("RGB", (1151, 26)),
        _LEADING_PUNCTUATED_LINE,
        recognizer,
    )

    expected = list(raw)
    expected[7:8] = [
        (
            _LEADING_PUNCTUATED_TARGET + ":",
            BoundingBox(804.72, 0, 852.72, _LEADING_PUNCTUATED_HEIGHT),
            0.994520,
        ),
        (
            _LEADING_PUNCTUATED_FOLLOWING,
            BoundingBox(841.72, 0, 888.72, _LEADING_PUNCTUATED_HEIGHT),
            0.998760,
        ),
    ]
    assert recovered == expected
    assert recognizer.calls == 32
    assert recognizer.word_box_thresholds == [0.001, 0.01, 0.02, 0.03, 0.05]


@pytest.mark.parametrize(
    ("case", "expected_calls"),
    [
        ("boundary-direct-text", 2),
        ("boundary-direct-confidence", 2),
        ("boundary-enhanced-text", 2),
        ("boundary-enhanced-confidence", 2),
        ("target-direct-text", 32),
        ("target-direct-confidence", 32),
        ("target-enhanced-text", 32),
        ("target-enhanced-confidence", 32),
        ("following-direct-text", 32),
        ("following-direct-confidence", 32),
        ("following-enhanced-text", 32),
        ("following-enhanced-confidence", 32),
        ("non-hangul", 32),
    ],
)
def test_confirmed_leading_punctuated_single_split_requires_consensus(
    case: str,
    expected_calls: int,
) -> None:
    keyword = {}
    if case == "boundary-direct-text":
        keyword["boundary_direct_text"] = _LEADING_PUNCTUATED_OTHER + ":"
    elif case == "boundary-direct-confidence":
        keyword["boundary_direct_confidence"] = 0.9943
    elif case == "boundary-enhanced-text":
        keyword["boundary_enhanced_text"] = _LEADING_PUNCTUATED_OTHER + ":"
    elif case == "boundary-enhanced-confidence":
        keyword["boundary_enhanced_confidence"] = 0.9949
    elif case == "target-direct-text":
        values = [_LEADING_PUNCTUATED_TARGET] * 7
        values[1] = _LEADING_PUNCTUATED_OTHER
        keyword["target_direct_texts"] = tuple(values)
    elif case == "target-direct-confidence":
        values = list(_LEADING_TARGET_DIRECT_CONFIDENCES)
        values[1] = 0.9998
        keyword["target_direct_confidences"] = tuple(values)
    elif case == "target-enhanced-text":
        values = [_LEADING_PUNCTUATED_TARGET] * 7
        values[2] = _LEADING_PUNCTUATED_OTHER
        keyword["target_enhanced_texts"] = tuple(values)
    elif case == "target-enhanced-confidence":
        values = list(_LEADING_TARGET_ENHANCED_CONFIDENCES)
        values[2] = 0.9998
        keyword["target_enhanced_confidences"] = tuple(values)
    elif case == "following-direct-text":
        values = [_LEADING_PUNCTUATED_FOLLOWING] * 8
        values[3] = _LEADING_PUNCTUATED_OTHER * 2
        keyword["following_direct_texts"] = tuple(values)
    elif case == "following-direct-confidence":
        values = list(_LEADING_FOLLOWING_DIRECT_CONFIDENCES)
        values[3] = 0.9996
        keyword["following_direct_confidences"] = tuple(values)
    elif case == "following-enhanced-text":
        values = [_LEADING_PUNCTUATED_FOLLOWING] * 8
        values[4] = _LEADING_PUNCTUATED_OTHER * 2
        keyword["following_enhanced_texts"] = tuple(values)
    elif case == "following-enhanced-confidence":
        values = list(_LEADING_FOLLOWING_ENHANCED_CONFIDENCES)
        values[4] = 0.9996
        keyword["following_enhanced_confidences"] = tuple(values)
    else:
        keyword["target_direct_texts"] = ("A",) * 7
        keyword["target_enhanced_texts"] = ("A",) * 7
    raw = leading_punctuated_raw_words()
    selected = list(raw)
    recognizer = ConfirmedLeadingPunctuatedRecognizer(**keyword)

    assert (
        _recover_confirmed_leading_punctuated_single_split(
            selected,
            raw,
            Image.new("RGB", (1151, 26)),
            _LEADING_PUNCTUATED_LINE,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == expected_calls


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "ordinary-mismatch",
        "raw-shape",
        "candidate-order",
        "candidate-confidence",
        "width",
        "gap",
        "line-height",
        "crop-bounds",
        "ctc-boundary",
        "ctc-default",
    ],
)
def test_confirmed_leading_punctuated_single_split_requires_exact_profile(
    case: str,
) -> None:
    raw = leading_punctuated_raw_words()
    selected = list(raw)
    crop = Image.new("RGB", (1151, 26))
    line_box = _LEADING_PUNCTUATED_LINE
    recognizer = ConfirmedLeadingPunctuatedRecognizer(
        boundary_mismatch=case == "ctc-boundary",
        default_mismatch=case == "ctc-default",
    )
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "ordinary-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.9998)
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        selected = list(raw)
    elif case == "candidate-order":
        raw[7] = (
            _LEADING_PUNCTUATED_TARGET + _LEADING_PUNCTUATED_FOLLOWING + ":",
            raw[7][1],
            raw[7][2],
        )
        selected = list(raw)
    elif case == "candidate-confidence":
        raw[7] = (raw[7][0], raw[7][1], 0.9986)
        selected = list(raw)
    elif case == "width":
        raw[7] = (
            raw[7][0],
            BoundingBox(803.72, 0, 887.72, _LEADING_PUNCTUATED_HEIGHT),
            raw[7][2],
        )
        selected = list(raw)
    elif case == "gap":
        raw[8] = (
            raw[8][0],
            BoundingBox(902.72, 0, 1023.72, _LEADING_PUNCTUATED_HEIGHT),
            raw[8][2],
        )
        selected = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(48.72, 0, 1198.28, 0)
    elif case == "crop-bounds":
        crop = Image.new("RGB", (839, 26))

    assert (
        _recover_confirmed_leading_punctuated_single_split(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class LeadingPunctuatedNoSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        return RecognizedText("", 0.0)


def test_confirmed_leading_punctuated_single_split_requires_segmenter() -> None:
    raw = leading_punctuated_raw_words()
    recognizer = LeadingPunctuatedNoSegmenter()

    assert (
        _recover_confirmed_leading_punctuated_single_split(
            list(raw),
            raw,
            Image.new("RGB", (1151, 26)),
            _LEADING_PUNCTUATED_LINE,
            recognizer,
        )
        == raw
    )
    assert recognizer.calls == 0


class LeadingPunctuatedEngineRecognizer(ConfirmedLeadingPunctuatedRecognizer):
    def __init__(self) -> None:
        super().__init__()
        helper_values = self.values
        raw = leading_punctuated_raw_words()
        initial_values = [
            RecognizedText(text, confidence) for text, _box, confidence in raw
        ]
        initial_values.extend((RecognizedText("", 0.0), RecognizedText("", 0.0)))
        self.values = tuple(initial_values) + helper_values
        self.calls = 0
        self.word_box_thresholds = []

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return (
                (73, 146),
                (153, 232),
                (241, 288),
                (296, 493),
                (502, 551),
                (559, 689),
                (699, 746),
                (755, 840),
                (849, 970),
                (981, 1076),
                (1085, 1151),
            )
        return super().word_boxes(_image, space_threshold)


class LeadingPunctuatedEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(
                    48.72,
                    157.89,
                    1198.28,
                    182.54,
                ),
                0.9838,
            ),
        )


def test_engine_recovers_confirmed_leading_punctuated_single_split() -> None:
    recognizer = LeadingPunctuatedEngineRecognizer()
    engine = PaddleOcrEngine(LeadingPunctuatedEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[7]
    following = document.lines[0].eojeols[8]
    assert target.text == _LEADING_PUNCTUATED_TARGET
    assert target.box == BoundingBox(
        804.72,
        157.89,
        828.72,
        182.54,
    )
    assert target.confidence == 0.994520
    assert following.text == _LEADING_PUNCTUATED_FOLLOWING
    assert following.box == BoundingBox(
        841.72,
        157.89,
        888.72,
        182.54,
    )
    assert following.confidence == 0.998760


_LOW_CONFIDENCE_HEIGHT = 19.37
_LOW_CONFIDENCE_LINE = BoundingBox(
    84.76,
    0,
    639.24,
    _LOW_CONFIDENCE_HEIGHT,
)


def _low_confidence_hangul(start: int, count: int) -> str:
    return "".join(chr(start + offset) for offset in range(count))


_LOW_CONFIDENCE_TARGET = _low_confidence_hangul(0xC800, 3)
_LOW_CONFIDENCE_FOLLOWING = _low_confidence_hangul(0xC803, 5)
_LOW_CONFIDENCE_OTHER = _low_confidence_hangul(0xC880, 3)


def low_confidence_three_plus_five_raw_words():
    boxes = (
        BoundingBox(121.76, 0, 147.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(153.76, 0, 193.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(198.76, 0, 268.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(272.76, 0, 341.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(347.76, 0, 415.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(421.76, 0, 549.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(554.76, 0, 568.76, _LOW_CONFIDENCE_HEIGHT),
        BoundingBox(573.76, 0, 604.76, _LOW_CONFIDENCE_HEIGHT),
    )
    texts = (
        _low_confidence_hangul(0xC810, 2),
        _low_confidence_hangul(0xC812, 3),
        _low_confidence_hangul(0xC815, 5),
        _low_confidence_hangul(0xC81A, 5),
        _low_confidence_hangul(0xC81F, 5),
        _LOW_CONFIDENCE_TARGET + _LOW_CONFIDENCE_FOLLOWING,
        _low_confidence_hangul(0xC824, 1),
        _low_confidence_hangul(0xC825, 2) + ".",
    )
    confidences = (
        0.999550,
        0.999772,
        0.999274,
        0.999742,
        0.999276,
        0.988116,
        0.998118,
        0.988535,
    )
    return list(zip(texts, boxes, confidences, strict=True))


class ConfirmedLowConfidenceThreePlusFiveRecognizer:
    def __init__(
        self,
        *,
        case: str | None = None,
        boundary_mismatch: bool = False,
        default_mismatch: bool = False,
    ) -> None:
        target_values = [
            value
            for _ in range(7)
            for value in (
                RecognizedText(_LOW_CONFIDENCE_TARGET, 0.9999),
                RecognizedText(_LOW_CONFIDENCE_TARGET, 0.9999),
            )
        ]
        following_values = [
            value
            for _ in range(7)
            for value in (
                RecognizedText(_LOW_CONFIDENCE_FOLLOWING, 0.9995),
                RecognizedText(_LOW_CONFIDENCE_FOLLOWING, 0.9996),
            )
        ]
        values = target_values + following_values
        indexes = {
            "target-direct-text": 0,
            "target-enhanced-text": 1,
            "target-direct-confidence": 2,
            "target-enhanced-confidence": 3,
            "following-direct-text": 14,
            "following-enhanced-text": 15,
            "following-direct-confidence": 16,
            "following-enhanced-confidence": 17,
        }
        if case in indexes:
            index = indexes[case]
            current = values[index]
            if case.endswith("text"):
                replacement = (
                    _LOW_CONFIDENCE_OTHER
                    if case.startswith("target")
                    else _LOW_CONFIDENCE_OTHER + _LOW_CONFIDENCE_OTHER[:2]
                )
                values[index] = RecognizedText(replacement, current.confidence)
            else:
                confidence = 0.9995 if case.startswith("target") else (
                    0.9986 if "direct" in case else 0.9993
                )
                values[index] = RecognizedText(current.text, confidence)
        self.values = tuple(values)
        self.calls = 0
        self.word_box_thresholds: list[float] = []
        self.boundary_mismatch = boundary_mismatch
        self.default_mismatch = default_mismatch

    def word_boxes(self, _image, space_threshold: float = 0.07):
        self.word_box_thresholds.append(space_threshold)
        if space_threshold == 0.04:
            return (
                ((0, 62), (61, 128))
                if self.default_mismatch
                else ((0, 128),)
            )
        if self.boundary_mismatch and space_threshold == 0.02:
            return ((0, 128),)
        return ((0, 62), (61, 128))

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_low_confidence_three_plus_five_split_recovers_boundary() -> None:
    raw = low_confidence_three_plus_five_raw_words()
    recognizer = ConfirmedLowConfidenceThreePlusFiveRecognizer()

    recovered = _recover_confirmed_low_confidence_three_plus_five_split(
        list(raw),
        raw,
        Image.new("RGB", (556, 20)),
        _LOW_CONFIDENCE_LINE,
        recognizer,
    )

    expected = list(raw)
    expected[5:6] = [
        (
            _LOW_CONFIDENCE_TARGET,
            BoundingBox(421.76, 0, 465.76, _LOW_CONFIDENCE_HEIGHT),
            0.988116,
        ),
        (
            _LOW_CONFIDENCE_FOLLOWING,
            BoundingBox(482.76, 0, 549.76, _LOW_CONFIDENCE_HEIGHT),
            0.988116,
        ),
    ]
    assert recovered == expected
    assert recognizer.calls == 28
    assert recognizer.word_box_thresholds == [
        0.001,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
    ]


@pytest.mark.parametrize(
    "case",
    [
        "target-direct-text",
        "target-enhanced-text",
        "target-direct-confidence",
        "target-enhanced-confidence",
        "following-direct-text",
        "following-enhanced-text",
        "following-direct-confidence",
        "following-enhanced-confidence",
    ],
)
def test_confirmed_low_confidence_three_plus_five_split_requires_consensus(
    case: str,
) -> None:
    raw = low_confidence_three_plus_five_raw_words()
    recognizer = ConfirmedLowConfidenceThreePlusFiveRecognizer(case=case)

    assert (
        _recover_confirmed_low_confidence_three_plus_five_split(
            list(raw),
            raw,
            Image.new("RGB", (556, 20)),
            _LOW_CONFIDENCE_LINE,
            recognizer,
        )
        == raw
    )
    assert recognizer.calls == 28


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "ordinary-mismatch",
        "raw-shape",
        "candidate-confidence",
        "width",
        "gap",
        "line-height",
        "crop-bounds",
        "ctc-boundary",
        "ctc-default",
    ],
)
def test_confirmed_low_confidence_three_plus_five_split_requires_exact_profile(
    case: str,
) -> None:
    raw = low_confidence_three_plus_five_raw_words()
    selected = list(raw)
    crop = Image.new("RGB", (556, 20))
    line_box = _LOW_CONFIDENCE_LINE
    recognizer = ConfirmedLowConfidenceThreePlusFiveRecognizer(
        boundary_mismatch=case == "ctc-boundary",
        default_mismatch=case == "ctc-default",
    )
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "ordinary-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.9996)
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        selected = list(raw)
    elif case == "candidate-confidence":
        raw[5] = (raw[5][0], raw[5][1], 0.9879)
        selected = list(raw)
    elif case == "width":
        raw[5] = (
            raw[5][0],
            BoundingBox(421.76, 0, 548.76, _LOW_CONFIDENCE_HEIGHT),
            raw[5][2],
        )
        selected = list(raw)
    elif case == "gap":
        raw[6] = (
            raw[6][0],
            BoundingBox(553.76, 0, 568.76, _LOW_CONFIDENCE_HEIGHT),
            raw[6][2],
        )
        selected = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(84.76, 0, 639.24, 0)
    elif case == "crop-bounds":
        crop = Image.new("RGB", (464, 20))

    assert (
        _recover_confirmed_low_confidence_three_plus_five_split(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class LowConfidenceThreePlusFiveNoSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        return RecognizedText("", 0.0)


def test_confirmed_low_confidence_three_plus_five_split_requires_segmenter() -> None:
    raw = low_confidence_three_plus_five_raw_words()
    recognizer = LowConfidenceThreePlusFiveNoSegmenter()

    assert (
        _recover_confirmed_low_confidence_three_plus_five_split(
            list(raw),
            raw,
            Image.new("RGB", (556, 20)),
            _LOW_CONFIDENCE_LINE,
            recognizer,
        )
        == raw
    )
    assert recognizer.calls == 0


class LowConfidenceThreePlusFiveEngineRecognizer(
    ConfirmedLowConfidenceThreePlusFiveRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        helper_values = self.values
        raw = low_confidence_three_plus_five_raw_words()
        initial_values = [
            RecognizedText(text, confidence) for text, _box, confidence in raw
        ]
        initial_values.extend((RecognizedText("", 0.0), RecognizedText("", 0.0)))
        self.values = tuple(initial_values) + helper_values
        self.calls = 0
        self.word_box_thresholds = []

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return (
                (37, 63),
                (69, 109),
                (114, 184),
                (188, 257),
                (263, 331),
                (337, 465),
                (470, 484),
                (489, 520),
                (521, 556),
            )
        return super().word_boxes(_image, space_threshold)


class LowConfidenceThreePlusFiveEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(84.76, 147.33, 639.24, 166.7),
                0.994632,
            ),
        )


def test_engine_recovers_confirmed_low_confidence_three_plus_five_split() -> None:
    recognizer = LowConfidenceThreePlusFiveEngineRecognizer()
    engine = PaddleOcrEngine(
        LowConfidenceThreePlusFiveEngineDetector(),
        recognizer,
    )

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[5]
    following = document.lines[0].eojeols[6]
    assert target.text == _LOW_CONFIDENCE_TARGET
    assert target.box == BoundingBox(421.76, 147.33, 465.76, 166.7)
    assert target.confidence == 0.988116
    assert following.text == _LOW_CONFIDENCE_FOLLOWING
    assert following.box == BoundingBox(482.76, 147.33, 549.76, 166.7)
    assert following.confidence == 0.988116


_LEADING_THREE_SIX_HEIGHT = 29.94
_LEADING_THREE_SIX_LINE = BoundingBox(
    106.92,
    0,
    398.08,
    _LEADING_THREE_SIX_HEIGHT,
)


def _leading_three_six_hangul(start: int, count: int) -> str:
    return "".join(chr(start + offset) for offset in range(count))


_LEADING_THREE_SIX_TARGET = _leading_three_six_hangul(0xC900, 3)
_LEADING_THREE_SIX_FOLLOWING = _leading_three_six_hangul(0xC903, 6) + "."
_LEADING_THREE_SIX_OTHER = _leading_three_six_hangul(0xC980, 3)


def leading_three_plus_six_raw_words():
    return [
        (
            _leading_three_six_hangul(0xC910, 1),
            BoundingBox(122.92, 0, 143.92, _LEADING_THREE_SIX_HEIGHT),
            0.999912,
        ),
        (
            _LEADING_THREE_SIX_TARGET + _LEADING_THREE_SIX_FOLLOWING,
            BoundingBox(153.92, 0, 383.92, _LEADING_THREE_SIX_HEIGHT),
            0.991408,
        ),
    ]


class ConfirmedLeadingThreePlusSixRecognizer:
    def __init__(
        self,
        *,
        case: str | None = None,
        boundary_mismatch: bool = False,
        default_mismatch: bool = False,
    ) -> None:
        target_values = [
            value
            for _ in range(7)
            for value in (
                RecognizedText(_LEADING_THREE_SIX_TARGET, 0.9999),
                RecognizedText(_LEADING_THREE_SIX_TARGET, 0.9999),
            )
        ]
        following_confidences = (
            (0.9922, 0.9932),
            (0.9882, 0.9914),
            (0.9942, 0.9950),
            (0.9954, 0.9965),
            (0.9939, 0.9948),
            (0.9953, 0.9963),
        )
        following_values = [
            value
            for direct, retry in following_confidences
            for value in (
                RecognizedText(_LEADING_THREE_SIX_FOLLOWING, direct),
                RecognizedText(_LEADING_THREE_SIX_FOLLOWING, retry),
            )
        ]
        values = target_values + following_values
        indexes = {
            "target-direct-text": 0,
            "target-enhanced-text": 1,
            "target-direct-confidence": 2,
            "target-enhanced-confidence": 3,
            "following-direct-text": 14,
            "following-enhanced-text": 15,
            "following-direct-confidence": 16,
            "following-enhanced-confidence": 17,
        }
        if case in indexes:
            index = indexes[case]
            current = values[index]
            if case.endswith("text"):
                replacement = (
                    _LEADING_THREE_SIX_OTHER
                    if case.startswith("target")
                    else _LEADING_THREE_SIX_OTHER * 2 + "."
                )
                values[index] = RecognizedText(replacement, current.confidence)
            else:
                confidence = 0.9997 if case.startswith("target") else (
                    0.9880 if "direct" in case else 0.9912
                )
                values[index] = RecognizedText(current.text, confidence)
        self.values = tuple(values)
        self.calls = 0
        self.word_box_thresholds: list[float] = []
        self.boundary_mismatch = boundary_mismatch
        self.default_mismatch = default_mismatch

    def word_boxes(self, _image, space_threshold: float = 0.07):
        self.word_box_thresholds.append(space_threshold)
        if space_threshold == 0.02:
            return (
                ((0, 82), (92, 230))
                if self.default_mismatch
                else ((0, 230),)
            )
        if self.boundary_mismatch and space_threshold == 0.005:
            return ((0, 230),)
        return ((0, 82), (92, 230))

    def recognize(self, _image):
        if self.calls >= len(self.values):
            return RecognizedText("", 0.0)
        value = self.values[self.calls]
        self.calls += 1
        return value


def test_confirmed_leading_three_plus_six_split_recovers_boundary() -> None:
    raw = leading_three_plus_six_raw_words()
    recognizer = ConfirmedLeadingThreePlusSixRecognizer()

    recovered = _recover_confirmed_leading_three_plus_six_punctuated_split(
        list(raw),
        raw,
        Image.new("RGB", (293, 31)),
        _LEADING_THREE_SIX_LINE,
        recognizer,
    )

    expected = list(raw)
    expected[1:2] = [
        (
            _LEADING_THREE_SIX_TARGET,
            BoundingBox(153.92, 0, 221.92, _LEADING_THREE_SIX_HEIGHT),
            0.991408,
        ),
        (
            _LEADING_THREE_SIX_FOLLOWING,
            BoundingBox(245.92, 0, 383.92, _LEADING_THREE_SIX_HEIGHT),
            0.9882,
        ),
    ]
    assert recovered == expected
    assert recognizer.calls == 26
    assert recognizer.word_box_thresholds == [
        0.0005,
        0.001,
        0.003,
        0.005,
        0.01,
        0.015,
        0.02,
    ]


@pytest.mark.parametrize(
    "case",
    [
        "target-direct-text",
        "target-enhanced-text",
        "target-direct-confidence",
        "target-enhanced-confidence",
        "following-direct-text",
        "following-enhanced-text",
        "following-direct-confidence",
        "following-enhanced-confidence",
    ],
)
def test_confirmed_leading_three_plus_six_split_requires_consensus(
    case: str,
) -> None:
    raw = leading_three_plus_six_raw_words()
    recognizer = ConfirmedLeadingThreePlusSixRecognizer(case=case)

    assert (
        _recover_confirmed_leading_three_plus_six_punctuated_split(
            list(raw),
            raw,
            Image.new("RGB", (293, 31)),
            _LEADING_THREE_SIX_LINE,
            recognizer,
        )
        == raw
    )
    assert recognizer.calls == 26


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "ordinary-mismatch",
        "raw-shape",
        "candidate-order",
        "candidate-confidence",
        "width",
        "gap",
        "line-height",
        "crop-bounds",
        "ctc-boundary",
        "ctc-default",
    ],
)
def test_confirmed_leading_three_plus_six_split_requires_exact_profile(
    case: str,
) -> None:
    raw = leading_three_plus_six_raw_words()
    selected = list(raw)
    crop = Image.new("RGB", (293, 31))
    line_box = _LEADING_THREE_SIX_LINE
    recognizer = ConfirmedLeadingThreePlusSixRecognizer(
        boundary_mismatch=case == "ctc-boundary",
        default_mismatch=case == "ctc-default",
    )
    if case == "selected-count":
        selected.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "ordinary-mismatch":
        selected[0] = (selected[0][0], selected[0][1], 0.9999)
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        selected = list(raw)
    elif case == "candidate-order":
        raw[1] = (
            "."
            + _LEADING_THREE_SIX_TARGET
            + _LEADING_THREE_SIX_FOLLOWING[:-1],
            raw[1][1],
            raw[1][2],
        )
        selected = list(raw)
    elif case == "candidate-confidence":
        raw[1] = (raw[1][0], raw[1][1], 0.9912)
        selected = list(raw)
    elif case == "width":
        raw[1] = (
            raw[1][0],
            BoundingBox(153.92, 0, 382.92, _LEADING_THREE_SIX_HEIGHT),
            raw[1][2],
        )
        selected = list(raw)
    elif case == "gap":
        raw[1] = (
            raw[1][0],
            BoundingBox(152.92, 0, 382.92, _LEADING_THREE_SIX_HEIGHT),
            raw[1][2],
        )
        selected = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(106.92, 0, 398.08, 0)
    elif case == "crop-bounds":
        crop = Image.new("RGB", (276, 31))

    assert (
        _recover_confirmed_leading_three_plus_six_punctuated_split(
            selected,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == selected
    )
    assert recognizer.calls == 0


class LeadingThreePlusSixNoSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image):
        self.calls += 1
        return RecognizedText("", 0.0)


def test_confirmed_leading_three_plus_six_split_requires_segmenter() -> None:
    raw = leading_three_plus_six_raw_words()
    recognizer = LeadingThreePlusSixNoSegmenter()

    assert (
        _recover_confirmed_leading_three_plus_six_punctuated_split(
            list(raw),
            raw,
            Image.new("RGB", (293, 31)),
            _LEADING_THREE_SIX_LINE,
            recognizer,
        )
        == raw
    )
    assert recognizer.calls == 0


class LeadingThreePlusSixEngineRecognizer(ConfirmedLeadingThreePlusSixRecognizer):
    def __init__(self) -> None:
        super().__init__()
        helper_values = self.values
        raw = leading_three_plus_six_raw_words()
        initial_values = [
            RecognizedText(text, confidence) for text, _box, confidence in raw
        ]
        self.values = tuple(initial_values) + helper_values
        self.calls = 0
        self.word_box_thresholds = []

    def word_boxes(self, _image, space_threshold: float = 0.07):
        if space_threshold == 0.07:
            return ((16, 37), (47, 277))
        return super().word_boxes(_image, space_threshold)


class LeadingThreePlusSixEngineDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(106.92, 152.8, 398.08, 182.74),
                0.994157,
            ),
        )


def test_engine_recovers_confirmed_leading_three_plus_six_split() -> None:
    recognizer = LeadingThreePlusSixEngineRecognizer()
    engine = PaddleOcrEngine(LeadingThreePlusSixEngineDetector(), recognizer)

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    target = document.lines[0].eojeols[1]
    following = document.lines[0].eojeols[2]
    assert target.text == _LEADING_THREE_SIX_TARGET
    assert target.box == BoundingBox(106.92 + 47, 152.8, 106.92 + 115, 182.74)
    assert target.confidence == 0.991408
    assert following.text == _LEADING_THREE_SIX_FOLLOWING[:-1]
    following_left = 106.92 + 139
    assert following.box == BoundingBox(
        following_left,
        152.8,
        following_left + 6 * (138 / 7),
        182.74,
    )
    assert following.confidence == 0.9882
