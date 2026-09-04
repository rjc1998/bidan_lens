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
    _recover_confirmed_internal_dash_wrapped_two_split,
    _recover_confirmed_internal_paired_wrapped_three_split,
    _recover_confirmed_internal_paired_wrapped_two_split,
    _recover_confirmed_isolated_dash_wrapped_four_plus_seven_split,
    _recover_confirmed_isolated_five_plus_three_punctuated_split,
    _recover_confirmed_isolated_mixed_prefix_split,
    _recover_confirmed_isolated_paired_wrapped_two_plus_two_split,
    _recover_confirmed_isolated_three_plus_five_punctuated_split,
    _recover_confirmed_leading_dash_three_plus_five_split,
    _recover_confirmed_leading_paired_wrapped_three_split,
    _recover_confirmed_leading_punctuated_single_split,
    _recover_confirmed_leading_three_plus_six_punctuated_split,
    _recover_confirmed_low_confidence_three_plus_five_split,
    _recover_confirmed_mismatched_curly_four_plus_four_split,
    _recover_confirmed_mismatched_curly_three_plus_three_split,
    _recover_confirmed_mismatched_curly_two_plus_one_split,
    _recover_confirmed_mismatched_wrapped_three_plus_one_split,
    _recover_confirmed_misplaced_curly_single_plus_structured_split,
    _recover_confirmed_numeric_ellipsis_tail_split,
    _recover_confirmed_one_plus_one_split,
    _recover_confirmed_overlapping_symbol_jamo_single,
    _recover_confirmed_paired_wrapped_four_plus_two_split,
    _recover_confirmed_paired_wrapped_three_plus_three_split,
    _recover_confirmed_paired_wrapper_four_substitution,
    _recover_confirmed_punctuated_three_plus_three_plus_one_split,
    _recover_confirmed_punctuated_three_plus_three_split,
    _recover_confirmed_punctuation_trimmed_single,
    _recover_confirmed_right_wrapper_five_substitution,
    _recover_confirmed_seven_character_splits,
    _recover_confirmed_substitution_readings,
    _recover_confirmed_terminal_dash_wrapped_two_plus_two_split,
    _recover_confirmed_terminal_paired_wrapped_single_split,
    _recover_confirmed_terminal_punctuated_overlap_pair,
    _recover_confirmed_terminal_three_plus_wrapped_two_split,
    _recover_confirmed_terminal_three_substitution,
    _recover_confirmed_terminal_wrapped_four_substitution,
    _recover_confirmed_terminal_wrapped_two_plus_one_split,
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
    _recover_confirmed_wrapped_single_plus_four_geometry,
    _recover_confirmed_wrapped_three_plus_four_split,
    _recover_initial_overlapping_word_pair,
    _recover_isolated_close_word_pairs,
    _recover_isolated_overlapping_word_pairs,
    _recover_overlapping_suffix_pairs,
    _recover_overlapping_word_triplets,
    _recover_relative_gap_three_plus_one_pairs,
    _recover_relative_gap_two_plus_two_pairs,
    _recover_terminal_digit_hangul_pair,
    _recover_terminal_overlapping_word_pair,
    _recover_word_boundaries,
    _remove_tiny_contained_fragments,
    _retry_binarized_small_hangul_word,
    _retry_confirmed_large_first_hangul_word,
    _retry_confirmed_trimmed_two_hangul_word,
    _split_cross_segment_quote_boundary,
    _split_mandatory_auxiliary_spacing,
    _split_punctuation_wrapped_word,
    _split_trailing_punctuation_boundary,
)


class Detector:
    def detect(self, _image):
        return (DetectedRegion(BoundingBox(10, 5, 110, 35), 0.9),)


def test_cross_segment_quote_recovers_opening_boundary() -> None:
    words = [
        ('\ub300\ud574"\uad6d\uac00\uc5d0', BoundingBox(10, 5, 130, 35), 0.96),
        ('\ub300\ud55c', BoundingBox(140, 5, 200, 35), 0.99),
        ('\uc788\ub2e4"\uba70', BoundingBox(210, 5, 290, 35), 0.99),
    ]

    parts = _split_cross_segment_quote_boundary(words)

    assert [part[0] for part in parts] == [
        '\ub300\ud574',
        '"\uad6d\uac00\uc5d0',
        '\ub300\ud55c',
        '\uc788\ub2e4"\uba70',
    ]
    assert parts[0][1] == BoundingBox(10, 5, 50, 35)
    assert parts[1][1] == BoundingBox(50, 5, 130, 35)


def test_cross_segment_quote_requires_confident_closer() -> None:
    quote = chr(34)
    words = [
        (f'\ub300\ud574{quote}\uad6d\uac00\uc5d0', BoundingBox(10, 5, 130, 35), 0.96),
        (f'\uc788\ub2e4{quote}\uba70', BoundingBox(140, 5, 220, 35), 0.9499),
    ]

    assert _split_cross_segment_quote_boundary(words) == words


@pytest.mark.parametrize(
    ('first', 'last', 'confidence'),
    [
        ('\ub300\ud574"\uad6d\uac00\uc5d0', '\uc788\ub2e4\uba70', 0.96),
        ('\ub300\ud574(\uad6d\uac00\uc5d0', '\uc788\ub2e4)\uba70', 0.96),
        ('\ub300\ud574"\uad6d\uac00\uc5d0', '\uc788\ub2e4"\uba70', 0.9499),
        ('\ub300\ud574"\uad6d', '\uc788\ub2e4"\uba70', 0.96),
        ('\ub300\ud574"\uad6d\uac00\uc5d0', '\uc788\ub2e4"\ub77c\uace0', 0.96),
    ],
)
def test_cross_segment_quote_requires_bounded_pairing(
    first: str,
    last: str,
    confidence: float,
) -> None:
    words = [
        (first, BoundingBox(10, 5, 130, 35), confidence),
        (last, BoundingBox(140, 5, 220, 35), 0.99),
    ]

    assert _split_cross_segment_quote_boundary(words) == words


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


class BinarizedRetryRecognizer:
    supports_binarized_small_text_retry = True

    def __init__(self, results: tuple[RecognizedText, ...]) -> None:
        self.results = iter(results)
        self.sizes: list[tuple[int, int]] = []

    def recognize(self, image: Image.Image) -> RecognizedText:
        self.sizes.append(image.size)
        return next(self.results)


def test_binarized_small_hangul_retry_accepts_unanimous_stronger_reading() -> None:
    recognizer = BinarizedRetryRecognizer(
        (
            RecognizedText('\uacb0\uc815\ub860\uc774\ub780', 0.99995),
            RecognizedText('\uacb0\uc815\ub860\uc774\ub780', 0.99993),
            RecognizedText('\uacb0\uc815\ub860\uc774\ub780', 0.99961),
        )
    )

    result = _retry_binarized_small_hangul_word(
        Image.new('RGB', (20, 12)),
        14.1,
        RecognizedText('\uae38\uc815\ub860\uc774\ub780', 0.9974),
        recognizer,
    )

    assert result == RecognizedText('\uacb0\uc815\ub860\uc774\ub780', 0.99961)
    assert recognizer.sizes == [(60, 36), (60, 36), (60, 36)]


def test_binarized_small_hangul_retry_accepts_slightly_taller_two_syllable_word() -> None:
    recognizer = BinarizedRetryRecognizer(
        (
            RecognizedText('\uacb0\uc815', 0.9998),
            RecognizedText('\uacb0\uc815', 0.9997),
            RecognizedText('\uacb0\uc815', 0.9996),
        )
    )

    result = _retry_binarized_small_hangul_word(
        Image.new('RGB', (20, 12)),
        15.9,
        RecognizedText('\uae38\uc815', 0.9973),
        recognizer,
    )

    assert result == RecognizedText('\uacb0\uc815', 0.9996)
    assert recognizer.sizes == [(60, 36), (60, 36), (60, 36)]


@pytest.mark.parametrize(
    ('line_height', 'recognized'),
    [
        (15.91, RecognizedText('\uac00\ub098', 0.9)),
        (15.9, RecognizedText('\uac00\ub098\ub2e4', 0.9)),
        (14.1, RecognizedText('\uac00', 0.9)),
        (14.1, RecognizedText('\uac00\ub098\ub2e4\ub77c\ub9c8\ubc14', 0.9)),
        (14.1, RecognizedText('\uac00A', 0.9)),
        (14.1, RecognizedText('\uac00\ub098', 0.998)),
    ],
)
def test_binarized_small_hangul_retry_requires_bounded_trigger(
    line_height: float,
    recognized: RecognizedText,
) -> None:
    recognizer = BinarizedRetryRecognizer(())

    assert (
        _retry_binarized_small_hangul_word(
            Image.new('RGB', (20, 12)),
            line_height,
            recognized,
            recognizer,
        )
        == recognized
    )
    assert recognizer.sizes == []


@pytest.mark.parametrize(
    'retries',
    [
        (
            RecognizedText('\uacb0\uc815', 0.99),
            RecognizedText('\uacb0\uc815', 0.99),
            RecognizedText('\uae38\uc815', 0.99),
        ),
        (
            RecognizedText('\uacb0\uc815', 0.95),
            RecognizedText('\uacb0\uc815', 0.95),
            RecognizedText('\uacb0\uc815', 0.94),
        ),
        (
            RecognizedText('\uae38\uc815', 0.99),
            RecognizedText('\uae38\uc815', 0.99),
            RecognizedText('\uae38\uc815', 0.99),
        ),
    ],
)
def test_binarized_small_hangul_retry_rejects_weak_or_ambiguous_evidence(
    retries: tuple[RecognizedText, ...],
) -> None:
    original = RecognizedText('\uae38\uc815', 0.95)
    recognizer = BinarizedRetryRecognizer(retries)

    assert (
        _retry_binarized_small_hangul_word(
            Image.new('RGB', (20, 12)),
            14.1,
            original,
            recognizer,
        )
        == original
    )


def test_binarized_small_hangul_retry_requires_strong_consensus() -> None:
    original = RecognizedText('\uae38\uc815', 0.8)
    recognizer = BinarizedRetryRecognizer(
        (
            RecognizedText('\uacb0\uc815', 0.939),
            RecognizedText('\uacb0\uc815', 0.938),
            RecognizedText('\uacb0\uc815', 0.937),
        )
    )

    assert (
        _retry_binarized_small_hangul_word(
            Image.new('RGB', (20, 12)),
            14.1,
            original,
            recognizer,
        )
        == original
    )



def test_trimmed_two_hangul_retry_accepts_five_matching_variants() -> None:
    recognizer = BinarizedRetryRecognizer(
        tuple(RecognizedText('\ub9d0\ub9cc', confidence) for confidence in (
            0.9991,
            0.9989,
            0.9988,
            0.9987,
            0.9986,
        ))
    )

    result = _retry_confirmed_trimmed_two_hangul_word(
        Image.new('RGB', (100, 18)),
        16,
        49,
        54,
        17.6,
        RecognizedText('\ub9e1\ub9cc', 0.744),
        recognizer,
    )

    assert result == RecognizedText('\ub9d0\ub9cc', 0.9986)
    assert recognizer.sizes == [(58, 36), (60, 36), (62, 36), (64, 36), (66, 36)]


def test_trimmed_two_hangul_retry_rejects_disagreement() -> None:
    recognizer = BinarizedRetryRecognizer(
        (
            RecognizedText('\ub9d0\ub9cc', 0.999),
            RecognizedText('\ub9d0\ub9cc', 0.999),
            RecognizedText('\ub9e1\ub9cc', 0.999),
            RecognizedText('\ub9d0\ub9cc', 0.999),
            RecognizedText('\ub9d0\ub9cc', 0.999),
        )
    )
    original = RecognizedText('\ub9e1\ub9cc', 0.744)

    assert _retry_confirmed_trimmed_two_hangul_word(
        Image.new('RGB', (100, 18)),
        16,
        49,
        54,
        17.6,
        original,
        recognizer,
    ) == original


@pytest.mark.parametrize(
    ('line_height', 'recognized', 'left', 'right', 'following_left'),
    [
        (17.4, RecognizedText('\ub9e1\ub9cc', 0.744), 16, 49, 54),
        (17.6, RecognizedText('\ub9e1\ub9cc\uc774', 0.744), 16, 49, 54),
        (17.6, RecognizedText('\ub9e1\ub9cc', 0.81), 16, 49, 54),
        (17.6, RecognizedText('\ub9e1\ub9cc', 0.744), 10, 43, 48),
        (17.6, RecognizedText('\ub9e1\ub9cc', 0.744), 16, 49, 52),
    ],
)
def test_trimmed_two_hangul_retry_requires_bounded_profile(
    line_height: float,
    recognized: RecognizedText,
    left: int,
    right: int,
    following_left: int,
) -> None:
    recognizer = BinarizedRetryRecognizer(())

    assert _retry_confirmed_trimmed_two_hangul_word(
        Image.new('RGB', (100, 18)),
        left,
        right,
        following_left,
        line_height,
        recognized,
        recognizer,
    ) == recognized
    assert recognizer.sizes == []


def test_large_first_hangul_retry_accepts_five_matching_variants() -> None:
    recognizer = BinarizedRetryRecognizer(
        tuple(
            RecognizedText('\ub808\ub2cc\uc740', confidence)
            for confidence in (0.9986, 0.9985, 0.9984, 0.9983, 0.9982)
        )
    )

    result = _retry_confirmed_large_first_hangul_word(
        Image.new('RGB', (200, 29)),
        70,
        155,
        170,
        28.2,
        RecognizedText('\ub7ec\ub2cc\uc740', 0.985),
        True,
        recognizer,
    )

    assert result == RecognizedText('\ub808\ub2cc\uc740', 0.9982)
    assert recognizer.sizes == [(166, 58), (168, 58), (170, 58), (172, 58), (174, 58)]


def test_large_first_hangul_retry_rejects_disagreement() -> None:
    recognizer = BinarizedRetryRecognizer(
        (
            RecognizedText('\ub808\ub2cc\uc740', 0.999),
            RecognizedText('\ub808\ub2cc\uc740', 0.999),
            RecognizedText('\ub7ec\ub2cc\uc740', 0.999),
            RecognizedText('\ub808\ub2cc\uc740', 0.999),
            RecognizedText('\ub808\ub2cc\uc740', 0.999),
        )
    )
    original = RecognizedText('\ub7ec\ub2cc\uc740', 0.985)

    assert _retry_confirmed_large_first_hangul_word(
        Image.new('RGB', (200, 29)),
        70,
        155,
        170,
        28.2,
        original,
        True,
        recognizer,
    ) == original


@pytest.mark.parametrize(
    (
        'line_height',
        'recognized',
        'left',
        'right',
        'following_left',
        'weak_leading_noise',
    ),
    [
        (28.0, RecognizedText('\ub7ec\ub2cc\uc740', 0.985), 70, 155, 170, True),
        (28.2, RecognizedText('\ub7ec\ub2cc', 0.985), 70, 155, 170, True),
        (28.2, RecognizedText('\ub7ec\ub2cc\uc740', 0.991), 70, 155, 170, True),
        (28.2, RecognizedText('\ub7ec\ub2cc\uc740', 0.985), 60, 145, 160, True),
        (28.2, RecognizedText('\ub7ec\ub2cc\uc740', 0.985), 70, 155, 165, True),
        (28.2, RecognizedText('\ub7ec\ub2cc\uc740', 0.985), 70, 155, 170, False),
    ],
)
def test_large_first_hangul_retry_requires_bounded_profile(
    line_height: float,
    recognized: RecognizedText,
    left: int,
    right: int,
    following_left: int,
    weak_leading_noise: bool,
) -> None:
    recognizer = BinarizedRetryRecognizer(())

    assert _retry_confirmed_large_first_hangul_word(
        Image.new('RGB', (200, 29)),
        left,
        right,
        following_left,
        line_height,
        recognized,
        weak_leading_noise,
        recognizer,
    ) == recognized
    assert recognizer.sizes == []


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


_ISOLATED_THREE_PLUS_FIVE_TARGET = "\uac00\ub098\ub2e4"
_ISOLATED_THREE_PLUS_FIVE_FOLLOWING = "\ub77c\ub9c8\ubc14\uc0ac\uc544"
_ISOLATED_THREE_PLUS_FIVE_TEXT = (
    _ISOLATED_THREE_PLUS_FIVE_TARGET
    + "."
    + _ISOLATED_THREE_PLUS_FIVE_FOLLOWING
)


class ConfirmedIsolatedThreePlusFivePunctuatedRecognizer:
    def __init__(
        self,
        *,
        boundary_text: str = (
            _ISOLATED_THREE_PLUS_FIVE_TARGET + "."
        ),
        boundary_confidence: float = 0.9930,
        enhanced_boundary_confidence: float = 0.9927,
        target_text: str = _ISOLATED_THREE_PLUS_FIVE_TARGET,
        target_confidence: float = 0.9999,
        enhanced_target_confidence: float = 0.9999,
        following_text: str = _ISOLATED_THREE_PLUS_FIVE_FOLLOWING,
        following_confidence: float = 0.9995,
        enhanced_following_confidence: float = 0.9995,
        segments_005: tuple[tuple[int, int], ...] = (
            (0, 15),
            (14, 212),
            (211, 539),
        ),
    ) -> None:
        self.values = (
            RecognizedText(boundary_text, boundary_confidence),
            RecognizedText(boundary_text, enhanced_boundary_confidence),
            *(
                RecognizedText(target_text, target_confidence)
                for _ in range(7)
            ),
            *(
                RecognizedText(target_text, enhanced_target_confidence)
                for _ in range(7)
            ),
            *(
                RecognizedText(following_text, following_confidence)
                for _ in range(7)
            ),
            *(
                RecognizedText(following_text, enhanced_following_confidence)
                for _ in range(7)
            ),
        )
        self.segments_005 = segments_005
        self.recognition_calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if space_threshold == 0.07:
            return ((0, image.width),)
        if space_threshold == 0.005:
            return self.segments_005
        if space_threshold in {0.0005, 0.001, 0.003, 0.01}:
            return ((0, 15), (14, 212), (211, 539))
        if space_threshold in {0.015, 0.02}:
            return ((0, 212), (211, 539))
        assert space_threshold == 0.03
        return ((0, 539),)

    def recognize(self, _image):
        result = self.values[self.recognition_calls]
        self.recognition_calls += 1
        return result


class IsolatedThreePlusFivePunctuatedRecognizer(
    ConfirmedIsolatedThreePlusFivePunctuatedRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.values = (
            RecognizedText(_ISOLATED_THREE_PLUS_FIVE_TEXT, 0.992479),
            *self.values,
        )


class IsolatedThreePlusFivePunctuatedDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                BoundingBox(99.96, 641.54, 637.04, 699.65),
                0.994781,
            ),
        )


def isolated_three_plus_five_punctuated_words(
    *,
    text: str = _ISOLATED_THREE_PLUS_FIVE_TEXT,
    confidence: float = 0.992479,
    box: BoundingBox | None = None,
) -> list[tuple[str, BoundingBox, float]]:
    return [
        (
            text,
            box or BoundingBox(0, 0, 537.08, 58.11),
            confidence,
        )
    ]


def test_confirmed_isolated_three_plus_five_punctuated_recovers() -> None:
    words = isolated_three_plus_five_punctuated_words()

    recovered = _recover_confirmed_isolated_three_plus_five_punctuated_split(
        words,
        Image.new("RGB", (539, 59)),
        BoundingBox(0, 0, 537.08, 58.11),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(),
    )

    assert recovered == [
        (
            _ISOLATED_THREE_PLUS_FIVE_TARGET + ".",
            BoundingBox(20, 0, 252, 58.11),
            0.992479,
        ),
        (
            _ISOLATED_THREE_PLUS_FIVE_FOLLOWING,
            BoundingBox(223, 0, 537.08, 58.11),
            0.992479,
        ),
    ]


@pytest.mark.parametrize(
    "recognizer",
    [
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            boundary_text=_ISOLATED_THREE_PLUS_FIVE_TARGET + "!"
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            boundary_confidence=0.9924
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            enhanced_boundary_confidence=0.9924
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            target_text="\uac00\ub098\ub77c"
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            target_confidence=0.9997
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            enhanced_target_confidence=0.9997
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            following_text="\ub77c\ub9c8\ubc14\uc0ac\uc790"
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            following_confidence=0.9991
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            enhanced_following_confidence=0.9991
        ),
        ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(
            segments_005=((0, 15), (14, 211), (211, 539))
        ),
    ],
)
def test_confirmed_isolated_three_plus_five_requires_crop_evidence(
    recognizer,
) -> None:
    words = isolated_three_plus_five_punctuated_words()

    assert (
        _recover_confirmed_isolated_three_plus_five_punctuated_split(
            words,
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
            recognizer,
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "crop", "line_box"),
    [
        (
            isolated_three_plus_five_punctuated_words(
                text="\uac00\ub098\ub2e4A\ub77c\ub9c8\ubc14\uc0ac\uc544"
            ),
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
        (
            isolated_three_plus_five_punctuated_words(
                text="A\ub098\ub2e4.\ub77c\ub9c8\ubc14\uc0ac\uc544"
            ),
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
        (
            isolated_three_plus_five_punctuated_words(confidence=0.9922),
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
        (
            isolated_three_plus_five_punctuated_words(
                box=BoundingBox(0, 0, 536, 58.11)
            ),
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
        (
            isolated_three_plus_five_punctuated_words(),
            Image.new("RGB", (538, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
        (
            isolated_three_plus_five_punctuated_words() * 2,
            Image.new("RGB", (539, 59)),
            BoundingBox(0, 0, 537.08, 58.11),
        ),
    ],
)
def test_confirmed_isolated_three_plus_five_requires_word_profile(
    words,
    crop,
    line_box,
) -> None:
    assert (
        _recover_confirmed_isolated_three_plus_five_punctuated_split(
            words,
            crop,
            line_box,
            ConfirmedIsolatedThreePlusFivePunctuatedRecognizer(),
        )
        == words
    )


def test_engine_recovers_isolated_three_plus_five_punctuated_segment() -> None:
    engine = PaddleOcrEngine(
        IsolatedThreePlusFivePunctuatedDetector(),
        IsolatedThreePlusFivePunctuatedRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (800, 800)))

    assert document.lines[0].text == (
        _ISOLATED_THREE_PLUS_FIVE_TARGET
        + ". "
        + _ISOLATED_THREE_PLUS_FIVE_FOLLOWING
    )
    assert [word.text for word in document.lines[0].eojeols] == [
        _ISOLATED_THREE_PLUS_FIVE_TARGET,
        _ISOLATED_THREE_PLUS_FIVE_FOLLOWING,
    ]
    target_box, following_box = (
        word.box for word in document.lines[0].eojeols
    )
    assert target_box.left == pytest.approx(119.96)
    assert target_box.right == pytest.approx(293.96)
    assert following_box.left == pytest.approx(322.96)
    assert following_box.right == pytest.approx(637.04)


_ISOLATED_MIXED_PREFIX_TARGET = chr(0xC760) + chr(0xC761)
_ISOLATED_MIXED_PREFIX_TEXT = _ISOLATED_MIXED_PREFIX_TARGET + "/A.1234-B5"
_ISOLATED_MIXED_PREFIX_BOX = BoundingBox(107.24, 631.76, 596.76, 707.48)


class ConfirmedIsolatedMixedPrefixRecognizer:
    def __init__(
        self,
        *,
        enhanced_candidate_text: str = _ISOLATED_MIXED_PREFIX_TEXT,
        enhanced_candidate_confidence: float = 0.9804,
        prefix_direct_text: str | None = None,
        prefix_direct_confidence: float = 0.991,
        prefix_enhanced_text: str | None = None,
        prefix_enhanced_confidence: float = 0.991,
        suffix_direct_text: str | None = None,
        suffix_direct_confidence: float = 0.75,
        suffix_enhanced_text: str | None = None,
        suffix_enhanced_confidence: float = 0.8,
        segments_003: tuple[tuple[int, int], ...] = (
            (0, 174),
            (173, 490),
        ),
    ) -> None:
        self.enhanced_candidate_text = enhanced_candidate_text
        self.enhanced_candidate_confidence = enhanced_candidate_confidence
        self.prefix_direct_text = (
            _ISOLATED_MIXED_PREFIX_TEXT[:3]
            if prefix_direct_text is None
            else prefix_direct_text
        )
        self.prefix_direct_confidence = prefix_direct_confidence
        self.prefix_enhanced_text = (
            _ISOLATED_MIXED_PREFIX_TEXT[:3]
            if prefix_enhanced_text is None
            else prefix_enhanced_text
        )
        self.prefix_enhanced_confidence = prefix_enhanced_confidence
        self.suffix_direct_text = (
            _ISOLATED_MIXED_PREFIX_TEXT[3:]
            if suffix_direct_text is None
            else suffix_direct_text
        )
        self.suffix_direct_confidence = suffix_direct_confidence
        self.suffix_enhanced_text = (
            _ISOLATED_MIXED_PREFIX_TEXT[3:]
            if suffix_enhanced_text is None
            else suffix_enhanced_text
        )
        self.suffix_enhanced_confidence = suffix_enhanced_confidence
        self.segments_003 = segments_003
        self.calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if space_threshold in {0.04, 0.05, 0.07}:
            return ((0, 490),)
        if space_threshold == 0.0001:
            return ((0, 174), (173, 251), (250, 490))
        if space_threshold == 0.003:
            return self.segments_003
        assert space_threshold in {
            0.0003,
            0.0005,
            0.001,
            0.002,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
            0.03,
        }
        return ((0, 174), (173, 490))

    def recognize(self, image):
        self.calls += 1
        width, height = image.size
        if (width, height) == (490, 77):
            return RecognizedText(_ISOLATED_MIXED_PREFIX_TEXT, 0.975795)
        if (width, height) == (980, 154):
            return RecognizedText(
                self.enhanced_candidate_text,
                self.enhanced_candidate_confidence,
            )
        if height == 77 and 170 <= width <= 178:
            return RecognizedText(
                self.prefix_direct_text,
                self.prefix_direct_confidence,
            )
        if height == 154 and 340 <= width <= 356:
            return RecognizedText(
                self.prefix_enhanced_text,
                self.prefix_enhanced_confidence,
            )
        if height == 77 and 313 <= width <= 319:
            return RecognizedText(
                self.suffix_direct_text,
                self.suffix_direct_confidence,
            )
        if height == 154 and 626 <= width <= 638:
            return RecognizedText(
                self.suffix_enhanced_text,
                self.suffix_enhanced_confidence,
            )
        return RecognizedText("", 0.0)


def isolated_mixed_prefix_words(
    *,
    text: str = _ISOLATED_MIXED_PREFIX_TEXT,
    confidence: float = 0.975795,
    box: BoundingBox = _ISOLATED_MIXED_PREFIX_BOX,
) -> list[tuple[str, BoundingBox, float]]:
    return [(text, box, confidence)]


def test_confirmed_isolated_mixed_prefix_recovers() -> None:
    recovered = _recover_confirmed_isolated_mixed_prefix_split(
        isolated_mixed_prefix_words(),
        Image.new("RGB", (490, 77)),
        _ISOLATED_MIXED_PREFIX_BOX,
        ConfirmedIsolatedMixedPrefixRecognizer(),
    )

    assert recovered == [
        (
            _ISOLATED_MIXED_PREFIX_TEXT[:3],
            BoundingBox(107.24, 631.76, 281.24, 707.48),
            0.975795,
        ),
        (
            _ISOLATED_MIXED_PREFIX_TEXT[3:],
            BoundingBox(280.24, 631.76, 596.76, 707.48),
            0.75,
        ),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {
            "enhanced_candidate_text": (
                _ISOLATED_MIXED_PREFIX_TEXT[:-1] + "6"
            )
        },
        {"enhanced_candidate_confidence": 0.9802},
        {"prefix_direct_text": _ISOLATED_MIXED_PREFIX_TARGET + "!"},
        {"prefix_direct_confidence": 0.9906},
        {"prefix_enhanced_text": _ISOLATED_MIXED_PREFIX_TARGET + "!"},
        {"prefix_enhanced_confidence": 0.9906},
        {"suffix_direct_text": "A.1234-B6"},
        {"suffix_direct_confidence": 0.7409},
        {"suffix_enhanced_text": "A.1234-B6"},
        {"suffix_enhanced_confidence": 0.7829},
        {"segments_003": ((0, 173), (173, 490))},
    ],
)
def test_confirmed_isolated_mixed_prefix_requires_crop_evidence(
    changes,
) -> None:
    words = isolated_mixed_prefix_words()

    assert (
        _recover_confirmed_isolated_mixed_prefix_split(
            words,
            Image.new("RGB", (490, 77)),
            _ISOLATED_MIXED_PREFIX_BOX,
            ConfirmedIsolatedMixedPrefixRecognizer(**changes),
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "word-count",
        "length",
        "target-shape",
        "punctuation-shape",
        "ascii-shape",
        "ascii-counts",
        "confidence",
        "box",
        "height",
        "crop",
    ],
)
def test_confirmed_isolated_mixed_prefix_requires_exact_profile(
    case: str,
) -> None:
    words = isolated_mixed_prefix_words()
    crop = Image.new("RGB", (490, 77))
    line_box = _ISOLATED_MIXED_PREFIX_BOX
    if case == "word-count":
        words *= 2
    elif case == "length":
        words = isolated_mixed_prefix_words(
            text=_ISOLATED_MIXED_PREFIX_TEXT[:-1]
        )
    elif case == "target-shape":
        words = isolated_mixed_prefix_words(
            text="A" + _ISOLATED_MIXED_PREFIX_TEXT[1:]
        )
    elif case == "punctuation-shape":
        words = isolated_mixed_prefix_words(
            text=(
                _ISOLATED_MIXED_PREFIX_TARGET
                + "A"
                + _ISOLATED_MIXED_PREFIX_TEXT[3:]
            )
        )
    elif case == "ascii-shape":
        words = isolated_mixed_prefix_words(
            text=(
                _ISOLATED_MIXED_PREFIX_TEXT[:3]
                + chr(0xC762)
                + _ISOLATED_MIXED_PREFIX_TEXT[4:]
            )
        )
    elif case == "ascii-counts":
        words = isolated_mixed_prefix_words(
            text=_ISOLATED_MIXED_PREFIX_TEXT[:-1] + "C"
        )
    elif case == "confidence":
        words = isolated_mixed_prefix_words(confidence=0.9756)
    elif case == "box":
        words = isolated_mixed_prefix_words(
            box=BoundingBox(107.24, 631.76, 596.75, 707.48)
        )
    elif case == "height":
        line_box = BoundingBox(107.24, 631.76, 596.76, 707.49)
        words = isolated_mixed_prefix_words(box=line_box)
    else:
        crop = Image.new("RGB", (489, 77))
    recognizer = ConfirmedIsolatedMixedPrefixRecognizer()

    assert (
        _recover_confirmed_isolated_mixed_prefix_split(
            words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class IsolatedMixedPrefixDetector:
    def detect(self, _image):
        return (DetectedRegion(_ISOLATED_MIXED_PREFIX_BOX, 0.992809),)


def test_engine_recovers_isolated_mixed_prefix_segment() -> None:
    engine = PaddleOcrEngine(
        IsolatedMixedPrefixDetector(),
        ConfirmedIsolatedMixedPrefixRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (800, 800)))

    line = document.lines[0]
    assert line.text == (
        _ISOLATED_MIXED_PREFIX_TEXT[:3]
        + " "
        + _ISOLATED_MIXED_PREFIX_TEXT[3:]
    )
    assert len(line.eojeols) == 1
    assert line.eojeols[0].text == _ISOLATED_MIXED_PREFIX_TARGET
    assert line.eojeols[0].box == BoundingBox(
        107.24,
        631.76,
        223.24,
        707.48,
    )
    assert line.eojeols[0].confidence == 0.975795


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



class RelativeGapThreePlusOneRecognizer:
    def __init__(self, results: tuple[RecognizedText, ...]) -> None:
        self.results = iter(results)
        self.sizes: list[tuple[int, int]] = []

    def recognize(self, image):
        self.sizes.append(image.size)
        return next(self.results)


def _relative_gap_three_plus_one_words() -> list[tuple[str, BoundingBox, float]]:
    return [
        ('\ud574\uc11d\ud558\ub294', BoundingBox(514, 0, 653, 31.7), 0.9992),
        ('\ubcf4\uc218\uc801', BoundingBox(670, 0, 770, 31.7), 0.9977),
        ('\uc778', BoundingBox(781, 0, 808, 31.7), 0.999955),
        ('\uc2e0\ud559\uc790\ub4e4\uc740', BoundingBox(828, 0, 1003, 31.7), 0.973733),
    ]


def test_relative_gap_three_plus_one_pair_merges_confirmed_union() -> None:
    words = _relative_gap_three_plus_one_words()
    recognizer = RelativeGapThreePlusOneRecognizer(
        (
            RecognizedText('\ubcf4\uc218\uc801\uc778', 0.9887),
            RecognizedText('\ubcf4\uc218\uc801\uc778', 0.9952),
        )
    )

    recovered = _recover_relative_gap_three_plus_one_pairs(
        words,
        Image.new('RGB', (1100, 32)),
        BoundingBox(0, 0, 1100, 31.7),
        recognizer,
    )

    assert recovered == [
        words[0],
        ('\ubcf4\uc218\uc801\uc778', BoundingBox(670, 0, 808, 31.7), 0.9887),
        words[3],
    ]
    assert recognizer.sizes == [(138, 32), (276, 64)]


@pytest.mark.parametrize(
    'enhanced',
    [
        RecognizedText('\ubcf4\uc218\uc801\uc778', 0.9949),
        RecognizedText('\ubcf4\uc218\uc778', 0.999),
    ],
)
def test_relative_gap_three_plus_one_pair_requires_enhanced_confirmation(
    enhanced: RecognizedText,
) -> None:
    words = _relative_gap_three_plus_one_words()
    recognizer = RelativeGapThreePlusOneRecognizer(
        (
            RecognizedText('\ubcf4\uc218\uc801\uc778', 0.9887),
            enhanced,
        )
    )

    assert (
        _recover_relative_gap_three_plus_one_pairs(
            words,
            Image.new('RGB', (1100, 32)),
            BoundingBox(0, 0, 1100, 31.7),
            recognizer,
        )
        == words
    )


def test_relative_gap_three_plus_one_pair_rejects_ordinary_gap() -> None:
    words = _relative_gap_three_plus_one_words()
    words[2] = ('\uc778', BoundingBox(782, 0, 809, 31.7), 0.999955)
    recognizer = RelativeGapThreePlusOneRecognizer(())

    assert (
        _recover_relative_gap_three_plus_one_pairs(
            words,
            Image.new('RGB', (1100, 32)),
            BoundingBox(0, 0, 1100, 31.7),
            recognizer,
        )
        == words
    )
    assert recognizer.sizes == []


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


class RelativeWideSplitRecognizer:
    def __init__(
        self,
        candidate_width: int,
        candidate_text: str,
        *,
        candidate_confidence: float = 0.9996,
        competitor_confidence: float = 0.8,
        competitor_text: str = 'competing',
    ) -> None:
        self.candidate_width = candidate_width
        self.candidate_text = candidate_text
        self.candidate_confidence = candidate_confidence
        self.competitor_confidence = competitor_confidence
        self.competitor_text = competitor_text

    def recognize(self, image):
        if image.width == self.candidate_width:
            return RecognizedText(self.candidate_text, self.candidate_confidence)
        return RecognizedText(self.competitor_text, self.competitor_confidence)


@pytest.mark.parametrize(
    ('words', 'candidate_width', 'candidate_text', 'merged_box'),
    [
        (
            [
                ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
                ('말이', BoundingBox(44, 0, 80, 20), 0.9996),
                ('나', BoundingBox(87.2, 0, 104.2, 20), 0.9991),
                ('뒷말', BoundingBox(114.2, 0, 146.2, 20), 0.999),
            ],
            61,
            '말이나',
            BoundingBox(44, 0, 104.2, 20),
        ),
        (
            [
                ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
                ('우리', BoundingBox(40, 0, 77, 20), 0.9999),
                ('나라', BoundingBox(84.2, 0, 122.2, 20), 0.9999),
                ('뒷말', BoundingBox(134.3, 0, 164.3, 20), 0.999),
            ],
            83,
            '우리나라',
            BoundingBox(40, 0, 122.2, 20),
        ),
    ],
)
def test_relative_wide_hangul_pair_merges_exact_union(
    words,
    candidate_width: int,
    candidate_text: str,
    merged_box: BoundingBox,
) -> None:
    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (180, 20)),
        BoundingBox(0, 0, 180, 20),
        RelativeWideSplitRecognizer(candidate_width, candidate_text),
    )

    assert recovered == [
        words[0],
        (candidate_text, merged_box, min(words[1][2], words[2][2], 0.9996)),
        words[3],
    ]


@pytest.mark.parametrize(
    ('words', 'candidate_width', 'candidate_text'),
    [
        (
            [
                ('앞말', BoundingBox(0, 0, 34, 20), 0.999),
                ('말이', BoundingBox(44, 0, 80, 20), 0.9996),
                ('나', BoundingBox(87.2, 0, 104.2, 20), 0.9991),
                ('뒷말', BoundingBox(114.2, 0, 146.2, 20), 0.999),
            ],
            61,
            '말이나',
        ),
        (
            [
                ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
                ('우리', BoundingBox(40, 0, 77, 20), 0.9999),
                ('나라', BoundingBox(84.2, 0, 122.2, 20), 0.9999),
                ('뒷말', BoundingBox(132.2, 0, 164.2, 20), 0.999),
            ],
            83,
            '우리나라',
        ),
    ],
)
def test_relative_wide_hangul_pair_requires_wider_neighbors(
    words,
    candidate_width: int,
    candidate_text: str,
) -> None:
    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new('RGB', (180, 20)),
            BoundingBox(0, 0, 180, 20),
            RelativeWideSplitRecognizer(candidate_width, candidate_text),
        )
        == words
    )


def test_relative_wide_hangul_pair_rejects_strong_competitor() -> None:
    words = [
        ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
        ('말이', BoundingBox(44, 0, 80, 20), 0.9996),
        ('나', BoundingBox(87.2, 0, 104.2, 20), 0.9991),
        ('뒷말', BoundingBox(114.2, 0, 146.2, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new('RGB', (180, 20)),
            BoundingBox(0, 0, 180, 20),
            RelativeWideSplitRecognizer(
                61,
                '말이나',
                competitor_confidence=0.99,
            ),
        )
        == words
    )


def test_relative_wide_two_plus_two_pair_requires_part_confidence() -> None:
    words = [
        ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
        ('우리', BoundingBox(40, 0, 77, 20), 0.9999),
        ('나라', BoundingBox(84.2, 0, 122.2, 20), 0.99979),
        ('뒷말', BoundingBox(134.3, 0, 164.3, 20), 0.999),
    ]

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new('RGB', (180, 20)),
            BoundingBox(0, 0, 180, 20),
            RelativeWideSplitRecognizer(83, '우리나라'),
        )
        == words
    )


def test_relative_wide_hangul_pair_accepts_explicit_boundary_competitor() -> None:
    words = [
        ('앞말', BoundingBox(0, 0, 30, 20), 0.999),
        ('말이', BoundingBox(44, 0, 80, 20), 0.9996),
        ('나', BoundingBox(87.2, 0, 104.2, 20), 0.9991),
        ('뒷말', BoundingBox(114.2, 0, 146.2, 20), 0.999),
    ]

    assert _recover_isolated_close_word_pairs(
        words,
        Image.new('RGB', (180, 20)),
        BoundingBox(0, 0, 180, 20),
        RelativeWideSplitRecognizer(
            61,
            '말이나',
            competitor_confidence=0.999,
            competitor_text='앞말 말이',
        ),
    ) == [
        words[0],
        (
            '말이나',
            BoundingBox(44, 0, 104.2, 20),
            0.9991,
        ),
        words[3],
    ]


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


_WRAPPED_THREE_FOUR_HEIGHT = 26.42
_WRAPPED_THREE_FOUR_LINE = BoundingBox(
    81.68,
    239.67,
    782.32,
    239.67 + _WRAPPED_THREE_FOUR_HEIGHT,
)
_WRAPPED_THREE_FOUR_TARGET = "".join(
    chr(0xC720 + offset) for offset in range(3)
)
_WRAPPED_THREE_FOUR_FOLLOWING = "".join(
    chr(0xC730 + offset) for offset in range(4)
)
_WRAPPED_THREE_FOUR_CANDIDATE = (
    '"'
    + _WRAPPED_THREE_FOUR_TARGET
    + "/"
    + _WRAPPED_THREE_FOUR_FOLLOWING
)


def wrapped_three_four_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(121.68, 239.67, 197.68, 266.09),
        BoundingBox(206.68, 239.67, 306.68, 266.09),
        BoundingBox(314.68, 239.67, 363.68, 266.09),
        BoundingBox(373.68, 239.67, 423.68, 266.09),
        BoundingBox(431.68, 239.67, 648.68, 266.09),
        BoundingBox(656.68, 239.67, 740.68, 266.09),
    )
    texts = (
        "".join(chr(0xC740 + offset) for offset in range(3)),
        "".join(chr(0xC750 + offset) for offset in range(4)),
        "".join(chr(0xC760 + offset) for offset in range(2)),
        "".join(chr(0xC770 + offset) for offset in range(2)),
        _WRAPPED_THREE_FOUR_CANDIDATE,
        "".join(chr(0xC780 + offset) for offset in range(3)) + ".",
    )
    confidences = (
        0.999711,
        0.999339,
        0.999883,
        0.999907,
        0.862854,
        0.994423,
    )
    return list(zip(texts, boxes, confidences, strict=True))


class ConfirmedWrappedThreePlusFourRecognizer:
    def __init__(
        self,
        *,
        enhanced_candidate_text: str = _WRAPPED_THREE_FOUR_CANDIDATE,
        enhanced_candidate_confidence: float = 0.6861,
        target_direct_text: str = _WRAPPED_THREE_FOUR_TARGET,
        target_direct_confidence: float = 0.9995,
        target_enhanced_text: str = _WRAPPED_THREE_FOUR_TARGET,
        target_enhanced_confidence: float = 0.9995,
        following_direct_text: str = _WRAPPED_THREE_FOUR_FOLLOWING,
        following_direct_confidence: float = 0.9989,
        following_enhanced_text: str = _WRAPPED_THREE_FOUR_FOLLOWING,
        following_enhanced_confidence: float = 0.9942,
        segments_005: tuple[tuple[int, int], ...] = (
            (0, 107),
            (115, 217),
        ),
    ) -> None:
        self.enhanced_candidate_text = enhanced_candidate_text
        self.enhanced_candidate_confidence = enhanced_candidate_confidence
        self.target_direct_text = target_direct_text
        self.target_direct_confidence = target_direct_confidence
        self.target_enhanced_text = target_enhanced_text
        self.target_enhanced_confidence = target_enhanced_confidence
        self.following_direct_text = following_direct_text
        self.following_direct_confidence = following_direct_confidence
        self.following_enhanced_text = following_enhanced_text
        self.following_enhanced_confidence = following_enhanced_confidence
        self.segments_005 = segments_005
        self.calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.width == 702:
            return (
                (40, 116),
                (125, 225),
                (233, 282),
                (292, 342),
                (350, 567),
                (575, 659),
            )
        if image.width != 217:
            return ((0, image.width),)
        if space_threshold in {0.03, 0.04, 0.05, 0.07}:
            return ((0, 217),)
        if space_threshold == 0.0001:
            return ((0, 91), (91, 107), (115, 217))
        if space_threshold == 0.0003:
            return ((0, 91), (91, 105), (115, 217))
        if space_threshold in {0.0005, 0.001, 0.002, 0.003}:
            return ((0, 91), (91, 107), (115, 217))
        if space_threshold == 0.005:
            return self.segments_005
        assert space_threshold in {0.007, 0.01, 0.015, 0.02}
        return ((0, 107), (115, 217))

    def recognize(self, image):
        self.calls += 1
        width, height = image.size
        if height == 28:
            raw = {
                76: RecognizedText(
                    wrapped_three_four_raw_words()[0][0],
                    0.999711,
                ),
                100: RecognizedText(
                    wrapped_three_four_raw_words()[1][0],
                    0.999339,
                ),
                49: RecognizedText(
                    wrapped_three_four_raw_words()[2][0],
                    0.999883,
                ),
                50: RecognizedText(
                    wrapped_three_four_raw_words()[3][0],
                    0.999907,
                ),
                217: RecognizedText(
                    _WRAPPED_THREE_FOUR_CANDIDATE,
                    0.862854,
                ),
                84: RecognizedText(
                    wrapped_three_four_raw_words()[5][0],
                    0.994423,
                ),
            }
            if width in raw:
                return raw[width]
            if 80 <= width <= 81:
                return RecognizedText(
                    self.target_direct_text,
                    self.target_direct_confidence,
                )
            if 102 <= width <= 108:
                return RecognizedText(
                    self.following_direct_text,
                    self.following_direct_confidence,
                )
        if (width, height) == (434, 56):
            return RecognizedText(
                self.enhanced_candidate_text,
                self.enhanced_candidate_confidence,
            )
        if height == 56 and 160 <= width <= 162:
            return RecognizedText(
                self.target_enhanced_text,
                self.target_enhanced_confidence,
            )
        if height == 56 and 204 <= width <= 216:
            return RecognizedText(
                self.following_enhanced_text,
                self.following_enhanced_confidence,
            )
        return RecognizedText("", 0.0)


def test_confirmed_wrapped_three_plus_four_recovers() -> None:
    words = wrapped_three_four_raw_words()

    recovered = _recover_confirmed_wrapped_three_plus_four_split(
        words,
        list(words),
        Image.new("RGB", (702, 28)),
        _WRAPPED_THREE_FOUR_LINE,
        ConfirmedWrappedThreePlusFourRecognizer(),
    )

    assert recovered[:4] == words[:4]
    assert [word[0] for word in recovered[4:]] == [
        _WRAPPED_THREE_FOUR_CANDIDATE[:5],
        _WRAPPED_THREE_FOUR_FOLLOWING,
        words[5][0],
    ]
    assert recovered[4][1].left == pytest.approx(431.68)
    assert recovered[4][1].right == pytest.approx(538.68)
    assert recovered[4][1].top == pytest.approx(239.67)
    assert recovered[4][1].bottom == pytest.approx(266.09)
    assert recovered[4][2] == 0.862854
    assert recovered[5][1].left == pytest.approx(546.68)
    assert recovered[5][1].right == pytest.approx(648.68)
    assert recovered[5][1].top == pytest.approx(239.67)
    assert recovered[5][1].bottom == pytest.approx(266.09)
    assert recovered[5][2] == 0.862854
    assert recovered[6] == words[5]


@pytest.mark.parametrize(
    "changes",
    [
        {
            "enhanced_candidate_text": (
                _WRAPPED_THREE_FOUR_CANDIDATE[:-1] + chr(0xC735)
            )
        },
        {"enhanced_candidate_confidence": 0.6859},
        {"target_direct_text": _WRAPPED_THREE_FOUR_TARGET[:-1] + chr(0xC725)},
        {"target_direct_confidence": 0.9993},
        {
            "target_enhanced_text": (
                _WRAPPED_THREE_FOUR_TARGET[:-1] + chr(0xC725)
            )
        },
        {"target_enhanced_confidence": 0.9993},
        {
            "following_direct_text": (
                _WRAPPED_THREE_FOUR_FOLLOWING[:-1] + chr(0xC735)
            )
        },
        {"following_direct_confidence": 0.9987},
        {
            "following_enhanced_text": (
                _WRAPPED_THREE_FOUR_FOLLOWING[:-1] + chr(0xC735)
            )
        },
        {"following_enhanced_confidence": 0.9940},
        {"segments_005": ((0, 106), (115, 217))},
    ],
)
def test_confirmed_wrapped_three_plus_four_requires_crop_evidence(
    changes,
) -> None:
    words = wrapped_three_four_raw_words()

    assert (
        _recover_confirmed_wrapped_three_plus_four_split(
            words,
            list(words),
            Image.new("RGB", (702, 28)),
            _WRAPPED_THREE_FOUR_LINE,
            ConfirmedWrappedThreePlusFourRecognizer(**changes),
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "raw-shape",
        "candidate-relationship",
        "confidence",
        "width",
        "gap",
        "line-height",
        "crop-size",
        "candidate-bounds",
    ],
)
def test_confirmed_wrapped_three_plus_four_requires_exact_profile(
    case: str,
) -> None:
    raw = wrapped_three_four_raw_words()
    words = list(raw)
    crop = Image.new("RGB", (702, 28))
    line_box = _WRAPPED_THREE_FOUR_LINE
    if case == "selected-count":
        words.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        words[0] = (words[0][0], words[0][1], 0.99972)
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        words = list(raw)
    elif case == "candidate-relationship":
        value = raw[4]
        changed = value[0][:4] + '"' + value[0][5:]
        raw[4] = (changed, value[1], value[2])
        words = list(raw)
    elif case == "confidence":
        raw[0] = (raw[0][0], raw[0][1], 0.9996)
        words = list(raw)
    elif case == "width":
        raw[0] = (
            raw[0][0],
            BoundingBox(121.68, 239.67, 198.68, 266.09),
            raw[0][2],
        )
        words = list(raw)
    elif case == "gap":
        raw[1] = (
            raw[1][0],
            BoundingBox(207.68, 239.67, 306.68, 266.09),
            raw[1][2],
        )
        words = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(81.68, 239.67, 782.32, 266.1)
    elif case == "crop-size":
        crop = Image.new("RGB", (701, 28))
    else:
        raw[4] = (
            raw[4][0],
            BoundingBox(432.68, 239.67, 648.68, 266.09),
            raw[4][2],
        )
        words = list(raw)
    recognizer = ConfirmedWrappedThreePlusFourRecognizer()

    assert (
        _recover_confirmed_wrapped_three_plus_four_split(
            words,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class WrappedThreePlusFourDetector:
    def detect(self, _image):
        return (
            DetectedRegion(_WRAPPED_THREE_FOUR_LINE, 0.99254),
        )


def test_engine_recovers_wrapped_three_plus_four_segment() -> None:
    engine = PaddleOcrEngine(
        WrappedThreePlusFourDetector(),
        ConfirmedWrappedThreePlusFourRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (900, 400)))

    line = document.lines[0]
    expected_words = wrapped_three_four_raw_words()
    assert line.text == " ".join(
        (
            *(word[0] for word in expected_words[:4]),
            _WRAPPED_THREE_FOUR_CANDIDATE[:5],
            _WRAPPED_THREE_FOUR_FOLLOWING,
            expected_words[5][0],
        )
    )
    assert [len(word.text) for word in line.eojeols] == [
        3,
        4,
        2,
        2,
        3,
        4,
        3,
    ]
    target = line.eojeols[4]
    assert target.text == _WRAPPED_THREE_FOUR_TARGET
    assert target.box.left == pytest.approx(453.08)
    assert target.box.right == pytest.approx(517.28)
    assert target.box.top == pytest.approx(239.67)
    assert target.box.bottom == pytest.approx(266.09)
    assert target.confidence == 0.862854


_WRAPPED_THREE_FIVE_HEIGHT = 26.413043
_WRAPPED_THREE_FIVE_LINE = BoundingBox(
    79.4,
    156.521739,
    769.6,
    156.521739 + _WRAPPED_THREE_FIVE_HEIGHT,
)
_WRAPPED_THREE_FIVE_TARGET = "".join(
    chr(0xC790 + offset) for offset in range(3)
)
_WRAPPED_THREE_FIVE_FOLLOWING = "".join(
    chr(0xC7A0 + offset) for offset in range(5)
)
_WRAPPED_THREE_FIVE_CANDIDATE = (
    chr(0x2014)
    + _WRAPPED_THREE_FIVE_TARGET
    + "-"
    + _WRAPPED_THREE_FIVE_FOLLOWING
)


def wrapped_three_five_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(121.4, 156.521739, 170.4, 182.934782),
        BoundingBox(177.4, 156.521739, 248.4, 182.934782),
        BoundingBox(258.4, 156.521739, 328.4, 182.934782),
        BoundingBox(340.4, 156.521739, 384.4, 182.934782),
        BoundingBox(396.4, 156.521739, 469.4, 182.934782),
        BoundingBox(476.4, 156.521739, 726.4, 182.934782),
    )
    texts = (
        "".join(chr(0xC7B0 + offset) for offset in range(2)),
        "".join(chr(0xC7C0 + offset) for offset in range(3)),
        "".join(chr(0xC7D0 + offset) for offset in range(3)),
        "".join(chr(0xC7E0 + offset) for offset in range(2)),
        "".join(chr(0xC7F0 + offset) for offset in range(3)),
        _WRAPPED_THREE_FIVE_CANDIDATE,
    )
    confidences = (
        0.999953,
        0.998768,
        0.999834,
        0.999859,
        0.999597,
        0.851829,
    )
    return list(zip(texts, boxes, confidences, strict=True))


class ConfirmedWrappedThreePlusFiveRecognizer:
    def __init__(
        self,
        *,
        enhanced_candidate_text: str = _WRAPPED_THREE_FIVE_CANDIDATE,
        enhanced_candidate_confidence: float = 0.7802,
        target_direct_text: str = _WRAPPED_THREE_FIVE_TARGET,
        target_direct_confidence: float = 0.9977,
        target_enhanced_text: str = _WRAPPED_THREE_FIVE_TARGET,
        target_enhanced_confidence: float = 0.9984,
        following_direct_text: str = _WRAPPED_THREE_FIVE_FOLLOWING,
        following_direct_confidence: float = 0.9998,
        following_enhanced_text: str = _WRAPPED_THREE_FIVE_FOLLOWING,
        following_enhanced_confidence: float = 0.9981,
        segments_0005: tuple[tuple[int, int], ...] = (
            (0, 106),
            (105, 121),
            (129, 156),
            (155, 250),
        ),
    ) -> None:
        self.enhanced_candidate_text = enhanced_candidate_text
        self.enhanced_candidate_confidence = enhanced_candidate_confidence
        self.target_direct_text = target_direct_text
        self.target_direct_confidence = target_direct_confidence
        self.target_enhanced_text = target_enhanced_text
        self.target_enhanced_confidence = target_enhanced_confidence
        self.following_direct_text = following_direct_text
        self.following_direct_confidence = following_direct_confidence
        self.following_enhanced_text = following_enhanced_text
        self.following_enhanced_confidence = following_enhanced_confidence
        self.segments_0005 = segments_0005
        self.calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.width == 691:
            return (
                (42, 91),
                (98, 169),
                (179, 249),
                (261, 305),
                (317, 390),
                (397, 647),
            )
        if image.width != 250:
            return ((0, image.width),)
        if space_threshold in {0.0001, 0.0003}:
            return (
                (0, 106),
                (105, 121),
                (129, 138),
                (137, 156),
                (155, 250),
            )
        if space_threshold == 0.0005:
            return self.segments_0005
        if space_threshold == 0.001:
            return ((0, 106), (105, 250))
        return ((0, 250),)

    def recognize(self, image):
        self.calls += 1
        width, height = image.size
        if height == 27:
            raw = {
                49: RecognizedText(
                    wrapped_three_five_raw_words()[0][0],
                    0.999953,
                ),
                71: RecognizedText(
                    wrapped_three_five_raw_words()[1][0],
                    0.998768,
                ),
                70: RecognizedText(
                    wrapped_three_five_raw_words()[2][0],
                    0.999834,
                ),
                44: RecognizedText(
                    wrapped_three_five_raw_words()[3][0],
                    0.999859,
                ),
                73: RecognizedText(
                    wrapped_three_five_raw_words()[4][0],
                    0.999597,
                ),
                250: RecognizedText(
                    _WRAPPED_THREE_FIVE_CANDIDATE,
                    0.851829,
                ),
            }
            if width in raw:
                return raw[width]
            if 79 <= width <= 81:
                return RecognizedText(
                    self.target_direct_text,
                    self.target_direct_confidence,
                )
            if 118 <= width <= 124:
                return RecognizedText(
                    self.following_direct_text,
                    self.following_direct_confidence,
                )
        if (width, height) == (500, 54):
            return RecognizedText(
                self.enhanced_candidate_text,
                self.enhanced_candidate_confidence,
            )
        if height == 54 and 158 <= width <= 162:
            return RecognizedText(
                self.target_enhanced_text,
                self.target_enhanced_confidence,
            )
        if height == 54 and 236 <= width <= 248:
            return RecognizedText(
                self.following_enhanced_text,
                self.following_enhanced_confidence,
            )
        return RecognizedText("", 0.0)


def test_confirmed_wrapped_three_plus_five_recovers() -> None:
    words = wrapped_three_five_raw_words()

    recovered = _recover_confirmed_leading_dash_three_plus_five_split(
        words,
        list(words),
        Image.new("RGB", (691, 27)),
        _WRAPPED_THREE_FIVE_LINE,
        ConfirmedWrappedThreePlusFiveRecognizer(),
    )

    assert recovered[:5] == words[:5]
    assert [word[0] for word in recovered[5:]] == [
        _WRAPPED_THREE_FIVE_CANDIDATE[:5],
        _WRAPPED_THREE_FIVE_FOLLOWING,
    ]
    assert recovered[5][1].left == pytest.approx(476.4)
    assert recovered[5][1].right == pytest.approx(597.4)
    assert recovered[5][1].top == pytest.approx(156.521739)
    assert recovered[5][1].bottom == pytest.approx(182.934782)
    assert recovered[5][2] == 0.851829
    assert recovered[6][1].left == pytest.approx(605.4)
    assert recovered[6][1].right == pytest.approx(726.4)
    assert recovered[6][1].top == pytest.approx(156.521739)
    assert recovered[6][1].bottom == pytest.approx(182.934782)
    assert recovered[6][2] == 0.851829


@pytest.mark.parametrize(
    "changes",
    [
        {
            "enhanced_candidate_text": (
                _WRAPPED_THREE_FIVE_CANDIDATE[:-1] + chr(0xC7A6)
            )
        },
        {"enhanced_candidate_confidence": 0.7800},
        {"target_direct_text": _WRAPPED_THREE_FIVE_TARGET[:-1] + chr(0xC796)},
        {"target_direct_confidence": 0.9975},
        {
            "target_enhanced_text": (
                _WRAPPED_THREE_FIVE_TARGET[:-1] + chr(0xC796)
            )
        },
        {"target_enhanced_confidence": 0.9982},
        {
            "following_direct_text": (
                _WRAPPED_THREE_FIVE_FOLLOWING[:-1] + chr(0xC7A6)
            )
        },
        {"following_direct_confidence": 0.9996},
        {
            "following_enhanced_text": (
                _WRAPPED_THREE_FIVE_FOLLOWING[:-1] + chr(0xC7A6)
            )
        },
        {"following_enhanced_confidence": 0.9979},
        {
            "segments_0005": (
                (0, 106),
                (105, 121),
                (129, 155),
                (155, 250),
            )
        },
    ],
)
def test_confirmed_wrapped_three_plus_five_requires_crop_evidence(
    changes,
) -> None:
    words = wrapped_three_five_raw_words()

    assert (
        _recover_confirmed_leading_dash_three_plus_five_split(
            words,
            list(words),
            Image.new("RGB", (691, 27)),
            _WRAPPED_THREE_FIVE_LINE,
            ConfirmedWrappedThreePlusFiveRecognizer(**changes),
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "selected-mismatch",
        "raw-shape",
        "candidate-relationship",
        "confidence",
        "width",
        "gap",
        "line-height",
        "crop-size",
        "candidate-bounds",
    ],
)
def test_confirmed_wrapped_three_plus_five_requires_exact_profile(
    case: str,
) -> None:
    raw = wrapped_three_five_raw_words()
    words = list(raw)
    crop = Image.new("RGB", (691, 27))
    line_box = _WRAPPED_THREE_FIVE_LINE
    if case == "selected-count":
        words.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "selected-mismatch":
        words[0] = (words[0][0], words[0][1], 0.99995)
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        words = list(raw)
    elif case == "candidate-relationship":
        value = raw[5]
        changed = value[0][:4] + '"' + value[0][5:]
        raw[5] = (changed, value[1], value[2])
        words = list(raw)
    elif case == "confidence":
        raw[0] = (raw[0][0], raw[0][1], 0.9998)
        words = list(raw)
    elif case == "width":
        raw[0] = (
            raw[0][0],
            BoundingBox(121.4, 156.521739, 171.4, 182.934782),
            raw[0][2],
        )
        words = list(raw)
    elif case == "gap":
        raw[1] = (
            raw[1][0],
            BoundingBox(178.4, 156.521739, 248.4, 182.934782),
            raw[1][2],
        )
        words = list(raw)
    elif case == "line-height":
        line_box = BoundingBox(79.4, 156.521739, 769.6, 182.95)
    elif case == "crop-size":
        crop = Image.new("RGB", (690, 27))
    else:
        raw[5] = (
            raw[5][0],
            BoundingBox(477.4, 156.521739, 727.4, 182.934782),
            raw[5][2],
        )
        words = list(raw)
    recognizer = ConfirmedWrappedThreePlusFiveRecognizer()

    assert (
        _recover_confirmed_leading_dash_three_plus_five_split(
            words,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class WrappedThreePlusFiveDetector:
    def detect(self, _image):
        return (
            DetectedRegion(_WRAPPED_THREE_FIVE_LINE, 0.990693),
        )


def test_engine_recovers_wrapped_three_plus_five_segment() -> None:
    engine = PaddleOcrEngine(
        WrappedThreePlusFiveDetector(),
        ConfirmedWrappedThreePlusFiveRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (900, 400)))

    line = document.lines[0]
    expected_words = wrapped_three_five_raw_words()
    assert line.text == " ".join(
        (
            *(word[0] for word in expected_words[:5]),
            _WRAPPED_THREE_FIVE_CANDIDATE[:5],
            _WRAPPED_THREE_FIVE_FOLLOWING,
        )
    )
    assert [len(word.text) for word in line.eojeols] == [
        2,
        3,
        3,
        2,
        3,
        3,
        5,
    ]
    target = line.eojeols[5]
    assert target.text == _WRAPPED_THREE_FIVE_TARGET
    assert target.box.left == pytest.approx(500.6)
    assert target.box.right == pytest.approx(573.2)
    assert target.box.top == pytest.approx(156.521739)
    assert target.box.bottom == pytest.approx(182.934782)
    assert target.confidence == 0.851829


_WRAPPED_ONE_FOUR_HEIGHT = 44.021739
_WRAPPED_ONE_FOUR_LINE = BoundingBox(
    61.48,
    166.3,
    1156.52,
    166.3 + _WRAPPED_ONE_FOUR_HEIGHT,
)
_WRAPPED_ONE_FOUR_TARGET = chr(0xC760)
_WRAPPED_ONE_FOUR_OTHER = chr(0xC770)
_WRAPPED_ONE_FOUR_FOLLOWING = "".join(
    chr(0xC780 + offset) for offset in range(4)
)
_WRAPPED_ONE_FOUR_CANDIDATE = (
    "/"
    + _WRAPPED_ONE_FOUR_TARGET
    + "/"
    + _WRAPPED_ONE_FOUR_FOLLOWING
)


def _wrapped_one_four_hangul(start: int, length: int) -> str:
    return "".join(chr(start + offset) for offset in range(length))


def wrapped_one_four_raw_words() -> list[tuple[str, BoundingBox, float]]:
    boxes = (
        BoundingBox(124.48, 166.3, 303.48, 166.3 + _WRAPPED_ONE_FOUR_HEIGHT),
        BoundingBox(321.48, 166.3, 451.48, 166.3 + _WRAPPED_ONE_FOUR_HEIGHT),
        BoundingBox(471.48, 166.3, 563.48, 166.3 + _WRAPPED_ONE_FOUR_HEIGHT),
        BoundingBox(576.48, 166.3, 853.48, 166.3 + _WRAPPED_ONE_FOUR_HEIGHT),
        BoundingBox(852.48, 166.3, 1091.48, 166.3 + _WRAPPED_ONE_FOUR_HEIGHT),
    )
    texts = (
        _wrapped_one_four_hangul(0xC790, 4),
        _wrapped_one_four_hangul(0xC794, 3),
        _wrapped_one_four_hangul(0xC797, 2),
        _WRAPPED_ONE_FOUR_CANDIDATE,
        _wrapped_one_four_hangul(0xC799, 5),
    )
    confidences = (0.999693, 0.999734, 0.995145, 0.993989, 0.999719)
    return list(zip(texts, boxes, confidences, strict=True))


def wrapped_one_four_selected_words(
    raw: list[tuple[str, BoundingBox, float]],
) -> list[tuple[str, BoundingBox, float]]:
    return [
        part
        for text_value, box, confidence in raw
        for part in _split_punctuation_wrapped_word(
            text_value,
            box,
            confidence,
        )
    ]


class ConfirmedWrappedSinglePlusFourRecognizer:
    def __init__(
        self,
        *,
        enhanced_candidate_text: str = _WRAPPED_ONE_FOUR_CANDIDATE,
        enhanced_candidate_confidence: float = 0.9958,
        wrapper_direct_text: str | None = None,
        wrapper_direct_first_confidence: float = 0.9272,
        wrapper_enhanced_text: str | None = None,
        wrapper_enhanced_first_confidence: float = 0.9604,
        target_text: str = _WRAPPED_ONE_FOUR_TARGET,
        target_confidence: float = 0.9995,
        following_text: str = _WRAPPED_ONE_FOUR_FOLLOWING,
        following_confidence: float = 0.9998,
        segments_005: tuple[tuple[int, int], ...] = (
            (0, 80),
            (93, 277),
        ),
    ) -> None:
        self.enhanced_candidate_text = enhanced_candidate_text
        self.enhanced_candidate_confidence = enhanced_candidate_confidence
        self.wrapper_direct_text = (
            _WRAPPED_ONE_FOUR_CANDIDATE[:3]
            if wrapper_direct_text is None
            else wrapper_direct_text
        )
        self.wrapper_direct_first_confidence = (
            wrapper_direct_first_confidence
        )
        self.wrapper_enhanced_text = (
            _WRAPPED_ONE_FOUR_CANDIDATE[:3]
            if wrapper_enhanced_text is None
            else wrapper_enhanced_text
        )
        self.wrapper_enhanced_first_confidence = (
            wrapper_enhanced_first_confidence
        )
        self.target_text = target_text
        self.target_confidence = target_confidence
        self.following_text = following_text
        self.following_confidence = following_confidence
        self.segments_005 = segments_005
        self.calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.width == 1096:
            return (
                (63, 242),
                (260, 390),
                (410, 502),
                (515, 792),
                (791, 1030),
            )
        if image.width != 277:
            return ((0, image.width),)
        if space_threshold in {0.04, 0.05, 0.07}:
            return ((0, 277),)
        if space_threshold == 0.0001:
            return ((0, 19), (18, 80), (93, 170), (169, 277))
        if space_threshold == 0.0003:
            return ((0, 19), (18, 80), (93, 277))
        if space_threshold in {0.0005, 0.001}:
            return ((0, 19), (18, 80), (93, 109), (108, 277))
        if space_threshold == 0.002:
            return ((0, 80), (93, 277))
        if space_threshold == 0.003:
            return ((0, 64), (63, 80), (93, 277))
        if space_threshold == 0.005:
            return self.segments_005
        assert space_threshold in {0.007, 0.01, 0.015, 0.02, 0.03}
        return ((0, 80), (93, 277))

    def recognize(self, image):
        self.calls += 1
        width, height = image.size
        if (width, height) == (554, 90):
            return RecognizedText(
                self.enhanced_candidate_text,
                self.enhanced_candidate_confidence,
            )
        if height == 90:
            if width >= 350:
                return RecognizedText(
                    self.following_text,
                    self.following_confidence,
                )
            if 150 <= width <= 170:
                confidence = (
                    self.wrapper_enhanced_first_confidence
                    if width == 156
                    else 0.981
                )
                return RecognizedText(
                    self.wrapper_enhanced_text,
                    confidence,
                )
            if 90 <= width <= 110:
                return RecognizedText(
                    self.target_text,
                    self.target_confidence,
                )
        if height == 45:
            raw_values = {
                179: RecognizedText(
                    _wrapped_one_four_hangul(0xC790, 4),
                    0.999693,
                ),
                130: RecognizedText(
                    _wrapped_one_four_hangul(0xC794, 3),
                    0.999734,
                ),
                92: RecognizedText(
                    _wrapped_one_four_hangul(0xC797, 2),
                    0.995145,
                ),
                277: RecognizedText(
                    _WRAPPED_ONE_FOUR_CANDIDATE,
                    0.993989,
                ),
                239: RecognizedText(
                    _wrapped_one_four_hangul(0xC799, 5),
                    0.999719,
                ),
            }
            if width in raw_values:
                return raw_values[width]
            if width in {177, 181, 182, 184, 186, 187, 189}:
                return RecognizedText(
                    self.following_text,
                    self.following_confidence,
                )
            if 78 <= width <= 84:
                confidence = (
                    self.wrapper_direct_first_confidence
                    if width == 78
                    else 0.984
                )
                return RecognizedText(
                    self.wrapper_direct_text,
                    confidence,
                )
            if 46 <= width <= 52:
                return RecognizedText(
                    self.target_text,
                    self.target_confidence,
                )
        return RecognizedText("", 0.0)


def test_confirmed_wrapped_single_plus_four_geometry_recovers() -> None:
    raw = wrapped_one_four_raw_words()
    words = wrapped_one_four_selected_words(raw)

    recovered = _recover_confirmed_wrapped_single_plus_four_geometry(
        words,
        raw,
        Image.new("RGB", (1096, 45)),
        _WRAPPED_ONE_FOUR_LINE,
        ConfirmedWrappedSinglePlusFourRecognizer(),
    )

    assert recovered[:3] == words[:3]
    assert recovered[3] == (
        _WRAPPED_ONE_FOUR_CANDIDATE[:3],
        BoundingBox(
            576.48,
            166.3,
            656.48,
            166.3 + _WRAPPED_ONE_FOUR_HEIGHT,
        ),
        0.9272,
    )
    assert recovered[4] == (
        _WRAPPED_ONE_FOUR_FOLLOWING,
        BoundingBox(
            669.48,
            166.3,
            853.48,
            166.3 + _WRAPPED_ONE_FOUR_HEIGHT,
        ),
        0.993989,
    )
    assert recovered[5] == words[5]


@pytest.mark.parametrize(
    "changes",
    [
        {"enhanced_candidate_text": _WRAPPED_ONE_FOUR_OTHER},
        {"enhanced_candidate_confidence": 0.9956},
        {"wrapper_direct_text": "/" + _WRAPPED_ONE_FOUR_OTHER + "/"},
        {"wrapper_direct_first_confidence": 0.9269},
        {"wrapper_enhanced_text": "/" + _WRAPPED_ONE_FOUR_OTHER + "/"},
        {"wrapper_enhanced_first_confidence": 0.9599},
        {"target_text": _WRAPPED_ONE_FOUR_OTHER},
        {"target_confidence": 0.9992},
        {
            "following_text": _wrapped_one_four_hangul(0xC7A0, 4),
        },
        {"following_confidence": 0.9996},
        {"segments_005": ((0, 79), (93, 277))},
    ],
)
def test_confirmed_wrapped_single_plus_four_requires_crop_evidence(
    changes,
) -> None:
    raw = wrapped_one_four_raw_words()
    words = wrapped_one_four_selected_words(raw)

    assert (
        _recover_confirmed_wrapped_single_plus_four_geometry(
            words,
            raw,
            Image.new("RGB", (1096, 45)),
            _WRAPPED_ONE_FOUR_LINE,
            ConfirmedWrappedSinglePlusFourRecognizer(**changes),
        )
        == words
    )


@pytest.mark.parametrize(
    "case",
    [
        "selected-count",
        "raw-count",
        "raw-shape",
        "selected-shape",
        "relationship",
        "confidence",
        "width",
        "gap",
        "line-width",
        "crop-size",
        "candidate-bounds",
    ],
)
def test_confirmed_wrapped_single_plus_four_requires_exact_profile(
    case: str,
) -> None:
    raw = wrapped_one_four_raw_words()
    words = wrapped_one_four_selected_words(raw)
    crop = Image.new("RGB", (1096, 45))
    line_box = _WRAPPED_ONE_FOUR_LINE
    if case == "selected-count":
        words.pop()
    elif case == "raw-count":
        raw.pop()
    elif case == "raw-shape":
        raw[0] = ("A" + raw[0][0], raw[0][1], raw[0][2])
        words = wrapped_one_four_selected_words(raw)
    elif case == "selected-shape":
        words[3] = (
            words[3][0] + "A",
            words[3][1],
            words[3][2],
        )
    elif case == "relationship":
        raw[3] = (
            "/" + _WRAPPED_ONE_FOUR_OTHER + "/" + raw[3][0][3:],
            raw[3][1],
            raw[3][2],
        )
    elif case == "confidence":
        raw[0] = (raw[0][0], raw[0][1], 0.9995)
        words = wrapped_one_four_selected_words(raw)
    elif case == "width":
        raw[0] = (
            raw[0][0],
            BoundingBox(
                raw[0][1].left,
                raw[0][1].top,
                raw[0][1].right + 1,
                raw[0][1].bottom,
            ),
            raw[0][2],
        )
        words = wrapped_one_four_selected_words(raw)
    elif case == "gap":
        raw[1] = (
            raw[1][0],
            BoundingBox(
                raw[1][1].left + 1,
                raw[1][1].top,
                raw[1][1].right,
                raw[1][1].bottom,
            ),
            raw[1][2],
        )
        words = wrapped_one_four_selected_words(raw)
    elif case == "line-width":
        line_box = BoundingBox(
            line_box.left,
            line_box.top,
            line_box.right - 1,
            line_box.bottom,
        )
    elif case == "crop-size":
        crop = Image.new("RGB", (1095, 45))
    else:
        raw[3] = (
            raw[3][0],
            BoundingBox(
                raw[3][1].left + 1,
                raw[3][1].top,
                raw[3][1].right,
                raw[3][1].bottom,
            ),
            raw[3][2],
        )
        words = wrapped_one_four_selected_words(raw)
    recognizer = ConfirmedWrappedSinglePlusFourRecognizer()

    assert (
        _recover_confirmed_wrapped_single_plus_four_geometry(
            words,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


class WrappedSinglePlusFourDetector:
    def detect(self, _image):
        return (
            DetectedRegion(_WRAPPED_ONE_FOUR_LINE, 0.994316),
        )


def test_engine_recovers_confirmed_wrapped_single_plus_four_geometry() -> None:
    engine = PaddleOcrEngine(
        WrappedSinglePlusFourDetector(),
        ConfirmedWrappedSinglePlusFourRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (1280, 720)))

    line = document.lines[0]
    target = line.eojeols[3]
    following = line.eojeols[4]
    assert target.text == _WRAPPED_ONE_FOUR_TARGET
    assert target.box.left == pytest.approx(603.1466666666666)
    assert target.box.right == pytest.approx(629.8133333333334)
    assert target.confidence == 0.9272
    assert following.text == _WRAPPED_ONE_FOUR_FOLLOWING
    assert following.box == BoundingBox(
        669.48,
        166.3,
        853.48,
        166.3 + _WRAPPED_ONE_FOUR_HEIGHT,
    )
    assert line.text == " ".join(
        word[0]
        for word in wrapped_one_four_selected_words(
            wrapped_one_four_raw_words()
        )
    )


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

_ISOLATED_FIVE_THREE_PREFIX = "".join(chr(0xC740 + index) for index in range(5))
_ISOLATED_FIVE_THREE_TARGET = "".join(chr(0xD000 + index) for index in range(3))
_ISOLATED_FIVE_THREE_TEXT = (
    _ISOLATED_FIVE_THREE_PREFIX
    + "-"
    + _ISOLATED_FIVE_THREE_TARGET
    + chr(0x2014)
)
_ISOLATED_FIVE_THREE_BOX = BoundingBox(
    106.76, 183.717391, 400.24, 215.413043
)
_ISOLATED_FIVE_THREE_SEGMENTS = {
    0.0001: (
        (0, 69),
        (68, 86),
        (85, 113),
        (112, 130),
        (129, 207),
        (206, 251),
        (250, 266),
    ),
    0.0003: ((0, 86), (85, 113), (112, 130), (129, 251), (250, 266)),
    0.0005: ((0, 113), (112, 130), (129, 251), (250, 266)),
    0.001: ((0, 130), (129, 266)),
    0.002: ((0, 130), (129, 146), (145, 266)),
    0.003: ((0, 130), (129, 266)),
    0.005: ((0, 130), (129, 266)),
    0.007: ((0, 130), (129, 266)),
    0.01: ((0, 130), (129, 266)),
    0.015: ((0, 266),),
    0.02: ((0, 266),),
    0.03: ((0, 266),),
    0.04: ((0, 266),),
    0.05: ((0, 266),),
    0.07: ((0, 266),),
}


class ConfirmedIsolatedFiveThreeRecognizer:
    def __init__(self, **overrides) -> None:
        self.candidate_direct = RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.8528)
        self.candidate_enhanced = RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.8157)
        self.full_enhanced = RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.6637)
        self.prefix_direct = RecognizedText(_ISOLATED_FIVE_THREE_PREFIX, 0.9999)
        self.prefix_enhanced = RecognizedText(_ISOLATED_FIVE_THREE_PREFIX, 0.9999)
        target_text = "-" + _ISOLATED_FIVE_THREE_TARGET
        self.target_direct = RecognizedText(target_text, 0.9377)
        self.target_enhanced = RecognizedText(target_text, 0.9251)
        self.segments_002 = ((0, 130), (129, 146), (145, 266))
        for name, value in overrides.items():
            setattr(self, name, value)
        self.recognition_calls = 0
        self.variant_calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.size == (295, 33) and space_threshold == 0.07:
            return ((14, 280),)
        if space_threshold == 0.002:
            return self.segments_002
        return _ISOLATED_FIVE_THREE_SEGMENTS[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        if image.size == (295, 33):
            return RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.718374)
        if image.size == (266, 33):
            return self.candidate_direct
        if image.size == (532, 66):
            return self.candidate_enhanced
        if image.size == (590, 66):
            return self.full_enhanced
        variant = self.variant_calls
        self.variant_calls += 1
        if variant < 7:
            return self.prefix_direct
        if variant < 14:
            return self.prefix_enhanced
        if variant < 21:
            return self.target_direct
        return self.target_enhanced


class IsolatedFiveThreeDetector:
    def detect(self, _image):
        return (DetectedRegion(_ISOLATED_FIVE_THREE_BOX, 0.98816),)


def isolated_five_three_words(
    *,
    text: str = _ISOLATED_FIVE_THREE_TEXT,
    confidence: float = 0.718374,
    box: BoundingBox = _ISOLATED_FIVE_THREE_BOX,
) -> list[tuple[str, BoundingBox, float]]:
    return [(text, box, confidence)]


def test_confirmed_isolated_five_plus_three_punctuated_recovers() -> None:
    recovered = _recover_confirmed_isolated_five_plus_three_punctuated_split(
        isolated_five_three_words(),
        Image.new("RGB", (295, 33)),
        _ISOLATED_FIVE_THREE_BOX,
        ConfirmedIsolatedFiveThreeRecognizer(),
    )

    assert recovered == [
        (
            _ISOLATED_FIVE_THREE_PREFIX,
            BoundingBox(120.76, 183.717391, 249.76, 215.413043),
            0.6637,
        ),
        (
            "-" + _ISOLATED_FIVE_THREE_TARGET + chr(0x2014),
            BoundingBox(249.76, 183.717391, 386.76, 215.413043),
            0.6637,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_direct": RecognizedText(_ISOLATED_FIVE_THREE_TEXT[:-1], 0.9)},
        {"candidate_direct": RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.8526)},
        {"candidate_enhanced": RecognizedText(_ISOLATED_FIVE_THREE_TEXT[:-1], 0.9)},
        {"candidate_enhanced": RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.8155)},
        {"full_enhanced": RecognizedText(_ISOLATED_FIVE_THREE_TEXT[:-1], 0.9)},
        {"full_enhanced": RecognizedText(_ISOLATED_FIVE_THREE_TEXT, 0.6635)},
        {"prefix_direct": RecognizedText(_ISOLATED_FIVE_THREE_PREFIX[:-1], 1.0)},
        {"prefix_enhanced": RecognizedText(_ISOLATED_FIVE_THREE_PREFIX, 0.9997)},
        {"target_direct": RecognizedText(_ISOLATED_FIVE_THREE_TARGET, 1.0)},
        {
            "target_enhanced": RecognizedText(
                "-" + _ISOLATED_FIVE_THREE_TARGET, 0.9249
            )
        },
        {"segments_002": ((0, 130), (129, 145), (145, 266))},
    ],
)
def test_confirmed_isolated_five_plus_three_requires_crop_evidence(
    overrides,
) -> None:
    words = isolated_five_three_words()

    assert (
        _recover_confirmed_isolated_five_plus_three_punctuated_split(
            words,
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
            ConfirmedIsolatedFiveThreeRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "crop", "line_box"),
    [
        (
            isolated_five_three_words(text="A" + _ISOLATED_FIVE_THREE_TEXT[1:]),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(
                text=_ISOLATED_FIVE_THREE_PREFIX
                + "/"
                + _ISOLATED_FIVE_THREE_TEXT[6:]
            ),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(
                text=_ISOLATED_FIVE_THREE_TEXT[:6]
                + "A"
                + _ISOLATED_FIVE_THREE_TEXT[7:]
            ),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(text=_ISOLATED_FIVE_THREE_TEXT[:-1] + "."),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(confidence=0.7182),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(
                box=BoundingBox(106.76, 183.717391, 399.24, 215.413043)
            ),
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words(),
            Image.new("RGB", (295, 33)),
            BoundingBox(106.76, 183.717391, 400.24, 216.413043),
        ),
        (
            isolated_five_three_words(),
            Image.new("RGB", (294, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
        (
            isolated_five_three_words() * 2,
            Image.new("RGB", (295, 33)),
            _ISOLATED_FIVE_THREE_BOX,
        ),
    ],
)
def test_confirmed_isolated_five_plus_three_requires_exact_profile(
    words,
    crop,
    line_box,
) -> None:
    recognizer = ConfirmedIsolatedFiveThreeRecognizer()

    assert (
        _recover_confirmed_isolated_five_plus_three_punctuated_split(
            words, crop, line_box, recognizer
        )
        == words
    )
    assert recognizer.recognition_calls == 0


def test_engine_recovers_isolated_five_plus_three_punctuated_segment() -> None:
    engine = PaddleOcrEngine(
        IsolatedFiveThreeDetector(),
        ConfirmedIsolatedFiveThreeRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (500, 300)))

    assert document.lines[0].text == (
        _ISOLATED_FIVE_THREE_PREFIX
        + " -"
        + _ISOLATED_FIVE_THREE_TARGET
        + chr(0x2014)
    )
    assert [word.text for word in document.lines[0].eojeols] == [
        _ISOLATED_FIVE_THREE_PREFIX,
        _ISOLATED_FIVE_THREE_TARGET,
    ]
    target_box = document.lines[0].eojeols[1].box
    assert target_box.left == pytest.approx(277.16)
    assert target_box.right == pytest.approx(359.36)

_TERMINAL_WRAPPED_TWO_LINE = BoundingBox(
    89.08, 147.913043, 638.92, 177.847826
)
_TERMINAL_WRAPPED_TWO_SEGMENTS = (
    (33, 103),
    (114, 189),
    (198, 270),
    (282, 328),
    (339, 518),
)
_TERMINAL_WRAPPED_TWO_FIRST = (
    "".join(chr(0xAC10 + index) for index in range(3)),
    "".join(chr(0xAD10 + index) for index in range(3)),
    "".join(chr(0xAE10 + index) for index in range(3)),
    "".join(chr(0xAF10 + index) for index in range(2)),
)
_TERMINAL_WRAPPED_TWO_PREFIX = "".join(
    chr(0xB010 + index) for index in range(3)
)
_TERMINAL_WRAPPED_TWO_TARGET = "".join(
    chr(0xB110 + index) for index in range(2)
)
_TERMINAL_WRAPPED_TWO_CANDIDATE = (
    _TERMINAL_WRAPPED_TWO_PREFIX
    + "-"
    + _TERMINAL_WRAPPED_TWO_TARGET
    + chr(0x2014)
    + chr(0x2026)
)
_TERMINAL_WRAPPED_TWO_CORRECTED = (
    chr(0x201C) + _TERMINAL_WRAPPED_TWO_TARGET + chr(0x201D)
)
_TERMINAL_WRAPPED_TWO_THRESHOLDS = {
    0.0001: ((0, 74), (82, 91), (90, 122), (121, 163), (162, 179)),
    0.0003: ((0, 74), (82, 91), (90, 163), (162, 179)),
    0.0005: ((0, 74), (82, 179)),
    0.001: ((0, 74), (82, 179)),
    0.002: ((0, 74), (82, 179)),
    0.003: ((0, 74), (82, 179)),
    0.005: ((0, 74), (82, 179)),
    0.007: ((0, 74), (82, 179)),
    0.01: ((0, 74), (82, 179)),
    0.015: ((0, 179),),
    0.02: ((0, 179),),
    0.03: ((0, 179),),
    0.04: ((0, 179),),
    0.05: ((0, 179),),
    0.07: ((0, 179),),
}


class ConfirmedTerminalWrappedTwoRecognizer:
    def __init__(self, **overrides) -> None:
        self.default_segments = _TERMINAL_WRAPPED_TWO_SEGMENTS
        self.thresholds = dict(_TERMINAL_WRAPPED_TWO_THRESHOLDS)
        self.candidate_direct = RecognizedText(
            _TERMINAL_WRAPPED_TWO_CANDIDATE, 0.7829
        )
        self.candidate_enhanced = RecognizedText(
            _TERMINAL_WRAPPED_TWO_CANDIDATE, 0.7802
        )
        self.prefix_direct = RecognizedText(
            _TERMINAL_WRAPPED_TWO_PREFIX, 0.9995
        )
        self.prefix_enhanced = RecognizedText(
            _TERMINAL_WRAPPED_TWO_PREFIX, 0.9997
        )
        self.wrapped_direct = RecognizedText(
            _TERMINAL_WRAPPED_TWO_CORRECTED, 0.4334
        )
        self.wrapped_enhanced = RecognizedText(
            _TERMINAL_WRAPPED_TWO_CORRECTED, 0.4570
        )
        self.target_direct = RecognizedText(
            _TERMINAL_WRAPPED_TWO_TARGET, 0.9996
        )
        self.target_enhanced = RecognizedText(
            _TERMINAL_WRAPPED_TWO_TARGET, 0.9997
        )
        for name, value in overrides.items():
            setattr(self, name, value)
        self.recognition_calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.size == (550, 31):
            return self.default_segments
        return self.thresholds[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        width, height = image.size
        if (width, height) == (179, 31):
            return self.candidate_direct
        if (width, height) == (358, 62):
            return self.candidate_enhanced
        if height == 31 and 70 <= width <= 78:
            return self.prefix_direct
        if height == 62 and 140 <= width <= 156:
            return self.prefix_enhanced
        if height == 31 and 92 <= width <= 96:
            return self.wrapped_direct
        if height == 62 and 184 <= width <= 192:
            return self.wrapped_enhanced
        if height == 31 and 46 <= width <= 56:
            return self.target_direct
        if height == 62 and 92 <= width <= 112:
            return self.target_enhanced
        return RecognizedText("", 0.0)


class EngineTerminalWrappedTwoRecognizer(
    ConfirmedTerminalWrappedTwoRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        self.raw_results = (
            *(
                RecognizedText(text, confidence)
                for text, confidence in zip(
                    _TERMINAL_WRAPPED_TWO_FIRST,
                    (0.999743, 0.999512, 0.999788, 0.999941),
                    strict=True,
                )
            ),
            RecognizedText(_TERMINAL_WRAPPED_TWO_CANDIDATE, 0.782912),
        )
        self.raw_index = 0

    def recognize(self, image):
        if self.raw_index < len(self.raw_results):
            result = self.raw_results[self.raw_index]
            self.raw_index += 1
            return result
        return super().recognize(image)


class TerminalWrappedTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(_TERMINAL_WRAPPED_TWO_LINE, 0.992033),)


def terminal_wrapped_two_words(
    *,
    candidate_text: str = _TERMINAL_WRAPPED_TWO_CANDIDATE,
    candidate_confidence: float = 0.782912,
) -> list[tuple[str, BoundingBox, float]]:
    texts = (*_TERMINAL_WRAPPED_TWO_FIRST, candidate_text)
    confidences = (0.999743, 0.999512, 0.999788, 0.999941, candidate_confidence)
    return [
        (
            text,
            BoundingBox(
                _TERMINAL_WRAPPED_TWO_LINE.left + left,
                _TERMINAL_WRAPPED_TWO_LINE.top,
                _TERMINAL_WRAPPED_TWO_LINE.left + right,
                _TERMINAL_WRAPPED_TWO_LINE.bottom,
            ),
            confidence,
        )
        for text, confidence, (left, right) in zip(
            texts, confidences, _TERMINAL_WRAPPED_TWO_SEGMENTS, strict=True
        )
    ]


def test_confirmed_terminal_three_plus_wrapped_two_recovers() -> None:
    words = terminal_wrapped_two_words()

    recovered = _recover_confirmed_terminal_three_plus_wrapped_two_split(
        words,
        Image.new("RGB", (550, 31)),
        _TERMINAL_WRAPPED_TWO_LINE,
        ConfirmedTerminalWrappedTwoRecognizer(),
    )

    candidate_left = _TERMINAL_WRAPPED_TWO_LINE.left + 339
    assert recovered == [
        *words[:-1],
        (
            _TERMINAL_WRAPPED_TWO_PREFIX,
            BoundingBox(
                candidate_left,
                147.913043,
                candidate_left + 74,
                177.847826,
            ),
            0.7802,
        ),
        (
            _TERMINAL_WRAPPED_TWO_CORRECTED,
            BoundingBox(
                candidate_left + 82,
                147.913043,
                candidate_left + 178,
                177.847826,
            ),
            0.4334,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_direct": RecognizedText(
                _TERMINAL_WRAPPED_TWO_CANDIDATE[:-1], 0.9
            )
        },
        {
            "candidate_enhanced": RecognizedText(
                _TERMINAL_WRAPPED_TWO_CANDIDATE, 0.7800
            )
        },
        {
            "prefix_direct": RecognizedText(
                _TERMINAL_WRAPPED_TWO_PREFIX[:-1], 1.0
            )
        },
        {
            "prefix_enhanced": RecognizedText(
                _TERMINAL_WRAPPED_TWO_PREFIX, 0.9995
            )
        },
        {
            "wrapped_direct": RecognizedText(
                chr(0x201C) + _TERMINAL_WRAPPED_TWO_TARGET + '"', 0.9
            )
        },
        {
            "wrapped_enhanced": RecognizedText(
                _TERMINAL_WRAPPED_TWO_CORRECTED, 0.4568
            )
        },
        {
            "target_direct": RecognizedText(
                _TERMINAL_WRAPPED_TWO_TARGET[:-1], 1.0
            )
        },
        {
            "target_enhanced": RecognizedText(
                _TERMINAL_WRAPPED_TWO_TARGET, 0.9995
            )
        },
        {
            "default_segments": (
                (33, 103),
                (114, 189),
                (198, 270),
                (282, 328),
                (339, 517),
            )
        },
    ],
)
def test_confirmed_terminal_three_plus_wrapped_two_requires_evidence(
    overrides,
) -> None:
    words = terminal_wrapped_two_words()
    recognizer = ConfirmedTerminalWrappedTwoRecognizer(**overrides)

    assert (
        _recover_confirmed_terminal_three_plus_wrapped_two_split(
            words,
            Image.new("RGB", (550, 31)),
            _TERMINAL_WRAPPED_TWO_LINE,
            recognizer,
        )
        == words
    )


def test_confirmed_terminal_three_plus_wrapped_two_requires_ctc_profile() -> None:
    words = terminal_wrapped_two_words()
    recognizer = ConfirmedTerminalWrappedTwoRecognizer()
    recognizer.thresholds[0.0003] = (
        (0, 74),
        (82, 91),
        (90, 162),
        (162, 179),
    )

    assert (
        _recover_confirmed_terminal_three_plus_wrapped_two_split(
            words,
            Image.new("RGB", (550, 31)),
            _TERMINAL_WRAPPED_TWO_LINE,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "preceding_text",
        "preceding_confidence",
        "candidate_pattern",
        "candidate_confidence",
        "box",
        "line",
        "crop",
    ],
)
def test_confirmed_terminal_three_plus_wrapped_two_requires_profile(
    case: str,
) -> None:
    words = terminal_wrapped_two_words()
    crop = Image.new("RGB", (550, 31))
    line_box = _TERMINAL_WRAPPED_TWO_LINE
    if case == "count":
        words = words[:-1]
    elif case == "preceding_text":
        words[0] = ("A" + words[0][0][1:], words[0][1], words[0][2])
    elif case == "preceding_confidence":
        words[0] = (words[0][0], words[0][1], 0.9993)
    elif case == "candidate_pattern":
        words = terminal_wrapped_two_words(
            candidate_text=_TERMINAL_WRAPPED_TWO_CANDIDATE[:3]
            + "/"
            + _TERMINAL_WRAPPED_TWO_CANDIDATE[4:]
        )
    elif case == "candidate_confidence":
        words = terminal_wrapped_two_words(candidate_confidence=0.7827)
    elif case == "box":
        text, box, confidence = words[-1]
        words[-1] = (
            text,
            BoundingBox(box.left + 1, box.top, box.right, box.bottom),
            confidence,
        )
    elif case == "line":
        line_box = BoundingBox(89.08, 147.913043, 638.92, 178.847826)
    else:
        crop = Image.new("RGB", (549, 31))
    recognizer = ConfirmedTerminalWrappedTwoRecognizer()

    assert (
        _recover_confirmed_terminal_three_plus_wrapped_two_split(
            words, crop, line_box, recognizer
        )
        == words
    )
    assert recognizer.recognition_calls == 0


def test_engine_recovers_terminal_three_plus_wrapped_two_segment() -> None:
    engine = PaddleOcrEngine(
        TerminalWrappedTwoDetector(),
        EngineTerminalWrappedTwoRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (800, 300)))

    assert document.lines[0].text == " ".join(
        (
            *_TERMINAL_WRAPPED_TWO_FIRST,
            _TERMINAL_WRAPPED_TWO_PREFIX,
            _TERMINAL_WRAPPED_TWO_CORRECTED,
        )
    )
    assert [word.text for word in document.lines[0].eojeols] == [
        *_TERMINAL_WRAPPED_TWO_FIRST,
        _TERMINAL_WRAPPED_TWO_PREFIX,
        _TERMINAL_WRAPPED_TWO_TARGET,
    ]
    target_box = document.lines[0].eojeols[-1].box
    assert target_box.left == pytest.approx(534.08)
    assert target_box.right == pytest.approx(582.08)
_TERMINAL_SINGLE_QUOTE_LINE = BoundingBox(
    47.64, 158.282609, 1227.36, 189.978261
)
_TERMINAL_SINGLE_QUOTE_SEGMENTS = (
    (74, 159),
    (168, 251),
    (261, 347),
    (356, 411),
    (420, 447),
    (457, 540),
    (551, 664),
    (674, 757),
    (768, 823),
    (834, 888),
    (898, 983),
    (993, 1106),
    (1111, 1155),
)
_TERMINAL_SINGLE_QUOTE_LENGTHS = (3, 3, 3, 2, 1, 3, 4, 3, 2, 2, 3)
_TERMINAL_SINGLE_QUOTE_FIRST = tuple(
    "".join(chr(0xAC00 + offset + index) for index in range(length))
    for offset, length in zip(
        range(0, 176, 16),
        _TERMINAL_SINGLE_QUOTE_LENGTHS,
        strict=True,
    )
)
_TERMINAL_SINGLE_QUOTE_TARGET = "".join(
    chr(0xB800 + index) for index in range(2)
)
_TERMINAL_SINGLE_QUOTE_FOLLOWING = chr(0xB900)
_TERMINAL_SINGLE_QUOTE_CANDIDATE = (
    "'" + _TERMINAL_SINGLE_QUOTE_TARGET + "'" + _TERMINAL_SINGLE_QUOTE_FOLLOWING
)
_TERMINAL_SINGLE_QUOTE_CORRECTED = (
    chr(0x2018) + _TERMINAL_SINGLE_QUOTE_TARGET + chr(0x2019)
)
_TERMINAL_SINGLE_QUOTE_CANDIDATE_THRESHOLDS = {
    0.0001: ((0, 14), (13, 75), (85, 113)),
    0.0003: ((0, 75), (85, 113)),
    0.0005: ((0, 75), (85, 113)),
    0.001: ((0, 75), (85, 113)),
    0.002: ((0, 75), (85, 113)),
    0.003: ((0, 75), (85, 113)),
    0.005: ((0, 113),),
    0.007: ((0, 113),),
    0.01: ((0, 113),),
    0.015: ((0, 113),),
    0.02: ((0, 113),),
    0.03: ((0, 113),),
    0.04: ((0, 113),),
    0.05: ((0, 113),),
    0.07: ((0, 113),),
}
_TERMINAL_SINGLE_QUOTE_COMBINED_THRESHOLDS = {
    0.0001: (
        (0, 14),
        (13, 30),
        (29, 62),
        (61, 88),
        (88, 113),
        (120, 162),
    ),
    0.0003: (
        (0, 14),
        (13, 30),
        (29, 62),
        (61, 88),
        (88, 113),
        (120, 162),
    ),
    0.0005: ((0, 14), (13, 30), (29, 88), (88, 113), (120, 162)),
    0.001: ((0, 14), (13, 88), (88, 104), (104, 113), (120, 162)),
    0.002: ((0, 14), (13, 88), (88, 113), (120, 162)),
    0.003: ((0, 88), (88, 113), (120, 162)),
    0.005: ((0, 88), (88, 113), (120, 152), (152, 162)),
    0.007: ((0, 88), (88, 113), (120, 152), (152, 162)),
    0.01: ((0, 88), (88, 113), (120, 162)),
    0.015: ((0, 88), (88, 113), (120, 162)),
    0.02: ((0, 88), (88, 113), (120, 162)),
    0.03: ((0, 88), (88, 113), (120, 162)),
    0.04: ((0, 113), (120, 162)),
    0.05: ((0, 113),),
    0.07: ((0, 113),),
}


class ConfirmedTerminalSingleQuoteRecognizer:
    def __init__(self, **overrides) -> None:
        self.default_segments = _TERMINAL_SINGLE_QUOTE_SEGMENTS
        self.candidate_thresholds = dict(
            _TERMINAL_SINGLE_QUOTE_CANDIDATE_THRESHOLDS
        )
        self.combined_thresholds = dict(
            _TERMINAL_SINGLE_QUOTE_COMBINED_THRESHOLDS
        )
        self.candidate_direct = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CANDIDATE, 0.7583
        )
        self.candidate_enhanced = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CANDIDATE, 0.5895
        )
        self.boundary_direct = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.5606
        )
        self.boundary_enhanced = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.6633
        )
        self.wrapper_direct = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.9992
        )
        self.wrapper_enhanced = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.9993
        )
        self.target_direct = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_TARGET, 0.99989
        )
        self.target_enhanced = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_TARGET, 0.99990
        )
        self.following_direct = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_FOLLOWING, 0.99990
        )
        self.following_enhanced = RecognizedText(
            _TERMINAL_SINGLE_QUOTE_FOLLOWING, 0.99995
        )
        for name, value in overrides.items():
            setattr(self, name, value)
        self.recognition_calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.size == (1181, 32):
            return self.default_segments
        if image.size == (113, 32):
            return self.candidate_thresholds[space_threshold]
        if image.size == (162, 32):
            return self.combined_thresholds[space_threshold]
        raise AssertionError(image.size)

    def recognize(self, image):
        self.recognition_calls += 1
        width, height = image.size
        if (width, height) == (113, 32):
            return self.candidate_direct
        if (width, height) == (226, 64):
            return self.candidate_enhanced
        if (width, height) == (75, 32):
            return self.boundary_direct
        if (width, height) == (150, 64):
            return self.boundary_enhanced
        if height == 32 and 72 <= width <= 82:
            return self.wrapper_direct
        if height == 64 and 144 <= width <= 164:
            return self.wrapper_enhanced
        if height == 32 and 49 <= width <= 57:
            return self.target_direct
        if height == 64 and 98 <= width <= 114:
            return self.target_enhanced
        if height == 32 and 26 <= width <= 29:
            return self.following_direct
        if height == 64 and 52 <= width <= 58:
            return self.following_enhanced
        return RecognizedText("", 0.0)


class EngineTerminalSingleQuoteRecognizer(
    ConfirmedTerminalSingleQuoteRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        confidences = (
            0.998911,
            0.999594,
            0.999767,
            0.999631,
            0.999836,
            0.999076,
            0.999559,
            0.999955,
            0.999742,
            0.998643,
            0.999939,
        )
        self.raw_results = (
            *(
                RecognizedText(text, confidence)
                for text, confidence in zip(
                    _TERMINAL_SINGLE_QUOTE_FIRST,
                    confidences,
                    strict=True,
                )
            ),
            RecognizedText(_TERMINAL_SINGLE_QUOTE_CANDIDATE, 0.758343),
            RecognizedText("", 0.0),
        )
        self.raw_index = 0

    def recognize(self, image):
        if self.raw_index < len(self.raw_results):
            result = self.raw_results[self.raw_index]
            self.raw_index += 1
            return result
        return super().recognize(image)


class TerminalSingleQuoteDetector:
    def detect(self, _image):
        return (DetectedRegion(_TERMINAL_SINGLE_QUOTE_LINE, 0.992273),)


def terminal_single_quote_words(
    *,
    candidate_text: str = _TERMINAL_SINGLE_QUOTE_CANDIDATE,
    candidate_confidence: float = 0.758343,
) -> list[tuple[str, BoundingBox, float]]:
    confidences = (
        0.998911,
        0.999594,
        0.999767,
        0.999631,
        0.999836,
        0.999076,
        0.999559,
        0.999955,
        0.999742,
        0.998643,
        0.999939,
        candidate_confidence,
    )
    texts = (*_TERMINAL_SINGLE_QUOTE_FIRST, candidate_text)
    return [
        (
            text,
            BoundingBox(
                _TERMINAL_SINGLE_QUOTE_LINE.left + left,
                _TERMINAL_SINGLE_QUOTE_LINE.top,
                _TERMINAL_SINGLE_QUOTE_LINE.left + right,
                _TERMINAL_SINGLE_QUOTE_LINE.bottom,
            ),
            confidence,
        )
        for text, confidence, (left, right) in zip(
            texts,
            confidences,
            _TERMINAL_SINGLE_QUOTE_SEGMENTS[:-1],
            strict=True,
        )
    ]


def test_confirmed_terminal_wrapped_two_plus_one_recovers() -> None:
    words = terminal_single_quote_words()

    recovered = _recover_confirmed_terminal_wrapped_two_plus_one_split(
        words,
        Image.new("RGB", (1181, 32)),
        _TERMINAL_SINGLE_QUOTE_LINE,
        ConfirmedTerminalSingleQuoteRecognizer(),
    )

    candidate_left = _TERMINAL_SINGLE_QUOTE_LINE.left + 993
    assert recovered == [
        *words[:-1],
        (
            _TERMINAL_SINGLE_QUOTE_CORRECTED,
            BoundingBox(
                candidate_left,
                _TERMINAL_SINGLE_QUOTE_LINE.top,
                candidate_left + 88,
                _TERMINAL_SINGLE_QUOTE_LINE.bottom,
            ),
            0.5606,
        ),
        (
            _TERMINAL_SINGLE_QUOTE_FOLLOWING,
            BoundingBox(
                candidate_left + 85,
                _TERMINAL_SINGLE_QUOTE_LINE.top,
                candidate_left + 113,
                _TERMINAL_SINGLE_QUOTE_LINE.bottom,
            ),
            0.5895,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_direct": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_CANDIDATE[:-1], 0.9
            )
        },
        {
            "candidate_enhanced": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_CANDIDATE, 0.5893
            )
        },
        {
            "boundary_direct": RecognizedText(
                "'" + _TERMINAL_SINGLE_QUOTE_TARGET + "'", 0.9
            )
        },
        {
            "boundary_enhanced": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.6631
            )
        },
        {
            "wrapper_direct": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.9990
            )
        },
        {
            "wrapper_enhanced": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_CORRECTED, 0.9991
            )
        },
        {
            "target_direct": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_TARGET, 0.99987
            )
        },
        {
            "target_enhanced": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_TARGET, 0.99988
            )
        },
        {
            "following_direct": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_FOLLOWING, 0.99988
            )
        },
        {
            "following_enhanced": RecognizedText(
                _TERMINAL_SINGLE_QUOTE_FOLLOWING, 0.99993
            )
        },
        {
            "default_segments": (
                *_TERMINAL_SINGLE_QUOTE_SEGMENTS[:-1],
                (1111, 1154),
            )
        },
    ],
)
def test_confirmed_terminal_wrapped_two_plus_one_requires_evidence(
    overrides,
) -> None:
    words = terminal_single_quote_words()

    assert (
        _recover_confirmed_terminal_wrapped_two_plus_one_split(
            words,
            Image.new("RGB", (1181, 32)),
            _TERMINAL_SINGLE_QUOTE_LINE,
            ConfirmedTerminalSingleQuoteRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize("profile", ["candidate", "combined"])
def test_confirmed_terminal_wrapped_two_plus_one_requires_ctc_profiles(
    profile: str,
) -> None:
    words = terminal_single_quote_words()
    recognizer = ConfirmedTerminalSingleQuoteRecognizer()
    thresholds = (
        recognizer.candidate_thresholds
        if profile == "candidate"
        else recognizer.combined_thresholds
    )
    thresholds[0.0003] = ((0, 1),)

    assert (
        _recover_confirmed_terminal_wrapped_two_plus_one_split(
            words,
            Image.new("RGB", (1181, 32)),
            _TERMINAL_SINGLE_QUOTE_LINE,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "preceding_text",
        "preceding_confidence",
        "candidate_pattern",
        "candidate_confidence",
        "box",
        "line",
        "crop",
    ],
)
def test_confirmed_terminal_wrapped_two_plus_one_requires_profile(
    case: str,
) -> None:
    words = terminal_single_quote_words()
    crop = Image.new("RGB", (1181, 32))
    line_box = _TERMINAL_SINGLE_QUOTE_LINE
    if case == "count":
        words = words[:-1]
    elif case == "preceding_text":
        text, box, confidence = words[0]
        words[0] = ("A" + text[1:], box, confidence)
    elif case == "preceding_confidence":
        text, box, _ = words[9]
        words[9] = (text, box, 0.9985)
    elif case == "candidate_pattern":
        words = terminal_single_quote_words(
            candidate_text=(
                "/"
                + _TERMINAL_SINGLE_QUOTE_TARGET
                + "'"
                + _TERMINAL_SINGLE_QUOTE_FOLLOWING
            )
        )
    elif case == "candidate_confidence":
        words = terminal_single_quote_words(candidate_confidence=0.7581)
    elif case == "box":
        text, box, confidence = words[-1]
        words[-1] = (
            text,
            BoundingBox(box.left + 1, box.top, box.right, box.bottom),
            confidence,
        )
    elif case == "line":
        line_box = BoundingBox(47.64, 158.282609, 1228.36, 189.978261)
    else:
        crop = Image.new("RGB", (1180, 32))
    recognizer = ConfirmedTerminalSingleQuoteRecognizer()

    assert (
        _recover_confirmed_terminal_wrapped_two_plus_one_split(
            words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


def test_engine_recovers_terminal_wrapped_two_plus_one_segment() -> None:
    engine = PaddleOcrEngine(
        TerminalSingleQuoteDetector(),
        EngineTerminalSingleQuoteRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (1300, 300)))

    assert document.lines[0].text == " ".join(
        (
            *_TERMINAL_SINGLE_QUOTE_FIRST,
            _TERMINAL_SINGLE_QUOTE_CORRECTED,
            _TERMINAL_SINGLE_QUOTE_FOLLOWING,
        )
    )
    assert [word.text for word in document.lines[0].eojeols] == [
        *_TERMINAL_SINGLE_QUOTE_FIRST,
        _TERMINAL_SINGLE_QUOTE_TARGET,
        _TERMINAL_SINGLE_QUOTE_FOLLOWING,
    ]
    target_box = document.lines[0].eojeols[-2].box
    assert target_box.left == pytest.approx(1062.64)
    assert target_box.right == pytest.approx(1106.64)
    following_box = document.lines[0].eojeols[-1].box
    assert following_box.left == pytest.approx(1125.64)
    assert following_box.right == pytest.approx(1153.64)
_TERMINAL_DASH_WRAPPED_TWO_LINE = BoundingBox(
    78.44, 156.130435, 782.56, 184.304348
)
_TERMINAL_DASH_WRAPPED_TWO_SEGMENTS = (
    (44, 115),
    (124, 172),
    (179, 276),
    (287, 334),
    (343, 389),
    (399, 471),
    (479, 501),
    (510, 661),
    (669, 705),
)
_TERMINAL_DASH_WRAPPED_TWO_LENGTHS = (3, 2, 4, 2, 2, 3, 1)
_TERMINAL_DASH_WRAPPED_TWO_FIRST = tuple(
    "".join(chr(0xBA00 + offset + index) for index in range(length))
    for offset, length in zip(
        range(0, 112, 16),
        _TERMINAL_DASH_WRAPPED_TWO_LENGTHS,
        strict=True,
    )
)
_TERMINAL_DASH_WRAPPED_TWO_TARGET = "".join(
    chr(0xBC00 + index) for index in range(2)
)
_TERMINAL_DASH_WRAPPED_TWO_FOLLOWING = "".join(
    chr(0xBD00 + index) for index in range(2)
)
_TERMINAL_DASH_WRAPPED_TWO_CANDIDATE = (
    chr(0x2014)
    + _TERMINAL_DASH_WRAPPED_TWO_TARGET
    + "-"
    + _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING
)
_TERMINAL_DASH_WRAPPED_TWO_CORRECTED = (
    chr(0x2014) + _TERMINAL_DASH_WRAPPED_TWO_TARGET + chr(0x2014)
)
_TERMINAL_DASH_WRAPPED_TWO_THRESHOLDS = {
    0.0001: (
        (0, 27),
        (26, 42),
        (41, 66),
        (65, 96),
        (105, 119),
        (118, 143),
        (142, 151),
    ),
    0.0003: ((0, 66), (65, 96), (105, 151)),
    0.0005: ((0, 96), (105, 151)),
    0.001: ((0, 96), (105, 151)),
    0.002: ((0, 96), (105, 151)),
    0.003: ((0, 96), (105, 151)),
    0.005: ((0, 96), (105, 151)),
    0.007: ((0, 151),),
    0.01: ((0, 151),),
    0.015: ((0, 151),),
    0.02: ((0, 151),),
    0.03: ((0, 151),),
    0.04: ((0, 151),),
    0.05: ((0, 151),),
    0.07: ((0, 151),),
}


class ConfirmedTerminalDashWrappedTwoRecognizer:
    def __init__(self, **overrides) -> None:
        self.default_segments = _TERMINAL_DASH_WRAPPED_TWO_SEGMENTS
        self.thresholds = dict(_TERMINAL_DASH_WRAPPED_TWO_THRESHOLDS)
        self.candidate_direct = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_CANDIDATE, 0.7299
        )
        self.candidate_enhanced = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_CANDIDATE, 0.7058
        )
        self.wrapper_direct = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_CORRECTED, 0.5738
        )
        self.wrapper_enhanced = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_CORRECTED, 0.5083
        )
        self.boundary_direct = RecognizedText(chr(0x2014), 0.6654)
        self.boundary_enhanced = RecognizedText(chr(0x2014), 0.6928)
        self.target_direct = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_TARGET, 0.99992
        )
        self.target_enhanced = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_TARGET, 0.99993
        )
        self.following_direct = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING, 0.99990
        )
        self.following_enhanced = RecognizedText(
            _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING, 0.99991
        )
        for name, value in overrides.items():
            setattr(self, name, value)
        self.recognition_calls = 0

    def word_boxes(
        self,
        image,
        space_threshold: float = 0.07,
    ) -> tuple[tuple[int, int], ...]:
        if image.size == (705, 29):
            return self.default_segments
        if image.size == (151, 29):
            return self.thresholds[space_threshold]
        return ((0, image.width),)

    def recognize(self, image):
        index = self.recognition_calls
        if index == 0:
            expected_sizes = {(151, 29)}
            result = self.candidate_direct
        elif index == 1:
            expected_sizes = {(302, 58)}
            result = self.candidate_enhanced
        elif index < 5:
            expected_sizes = {(101, 29), (104, 29), (106, 29)}
            result = self.wrapper_direct
        elif index < 8:
            expected_sizes = {(202, 58), (208, 58), (212, 58)}
            result = self.wrapper_enhanced
        elif index < 15:
            expected_sizes = {(width, 29) for width in range(27, 33)}
            result = self.boundary_direct
        elif index < 22:
            expected_sizes = {(width, 58) for width in range(54, 65, 2)}
            result = self.boundary_enhanced
        elif index < 29:
            expected_sizes = {(width, 29) for width in range(44, 49)}
            result = self.target_direct
        elif index < 36:
            expected_sizes = {(width, 58) for width in range(88, 97, 2)}
            result = self.target_enhanced
        elif index < 43:
            expected_sizes = {(width, 29) for width in range(41, 47)}
            result = self.following_direct
        else:
            expected_sizes = {(width, 58) for width in range(82, 93, 2)}
            result = self.following_enhanced
        if image.size not in expected_sizes:
            return RecognizedText("", 0.0)
        self.recognition_calls += 1
        return result


class EngineTerminalDashWrappedTwoRecognizer(
    ConfirmedTerminalDashWrappedTwoRecognizer
):
    def __init__(self) -> None:
        super().__init__()
        confidences = (
            0.999951,
            0.999709,
            0.999802,
            0.999976,
            0.999891,
            0.999977,
            0.999919,
        )
        self.raw_results = (
            *(
                RecognizedText(text, confidence)
                for text, confidence in zip(
                    _TERMINAL_DASH_WRAPPED_TWO_FIRST,
                    confidences,
                    strict=True,
                )
            ),
            RecognizedText(_TERMINAL_DASH_WRAPPED_TWO_CANDIDATE, 0.729867),
            RecognizedText("", 0.0),
        )
        self.raw_index = 0

    def recognize(self, image):
        if self.raw_index < len(self.raw_results):
            result = self.raw_results[self.raw_index]
            self.raw_index += 1
            return result
        return super().recognize(image)


class TerminalDashWrappedTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(_TERMINAL_DASH_WRAPPED_TWO_LINE, 0.988354),)


def terminal_dash_wrapped_two_words(
    *,
    candidate_text: str = _TERMINAL_DASH_WRAPPED_TWO_CANDIDATE,
    candidate_confidence: float = 0.729867,
) -> list[tuple[str, BoundingBox, float]]:
    confidences = (
        0.999951,
        0.999709,
        0.999802,
        0.999976,
        0.999891,
        0.999977,
        0.999919,
        candidate_confidence,
    )
    texts = (*_TERMINAL_DASH_WRAPPED_TWO_FIRST, candidate_text)
    return [
        (
            text,
            BoundingBox(
                _TERMINAL_DASH_WRAPPED_TWO_LINE.left + left,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.top,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.left + right,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.bottom,
            ),
            confidence,
        )
        for text, confidence, (left, right) in zip(
            texts,
            confidences,
            _TERMINAL_DASH_WRAPPED_TWO_SEGMENTS[:-1],
            strict=True,
        )
    ]


def test_confirmed_terminal_dash_wrapped_two_plus_two_recovers() -> None:
    words = terminal_dash_wrapped_two_words()

    recovered = _recover_confirmed_terminal_dash_wrapped_two_plus_two_split(
        words,
        Image.new("RGB", (705, 29)),
        _TERMINAL_DASH_WRAPPED_TWO_LINE,
        ConfirmedTerminalDashWrappedTwoRecognizer(),
    )

    candidate_left = _TERMINAL_DASH_WRAPPED_TWO_LINE.left + 510
    assert recovered == [
        *words[:-1],
        (
            _TERMINAL_DASH_WRAPPED_TWO_CORRECTED,
            BoundingBox(
                candidate_left,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.top,
                candidate_left + 96,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.bottom,
            ),
            0.5083,
        ),
        (
            _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING,
            BoundingBox(
                candidate_left + 105,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.top,
                candidate_left + 151,
                _TERMINAL_DASH_WRAPPED_TWO_LINE.bottom,
            ),
            0.7058,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_direct": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_CANDIDATE[:-1], 0.9
            )
        },
        {
            "candidate_enhanced": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_CANDIDATE, 0.7055
            )
        },
        {
            "wrapper_direct": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_CORRECTED, 0.5736
            )
        },
        {
            "wrapper_enhanced": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_CORRECTED, 0.5081
            )
        },
        {"boundary_direct": RecognizedText("-", 0.9)},
        {"boundary_enhanced": RecognizedText(chr(0x2014), 0.6926)},
        {
            "target_direct": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_TARGET, 0.99990
            )
        },
        {
            "target_enhanced": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_TARGET, 0.99991
            )
        },
        {
            "following_direct": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING, 0.99988
            )
        },
        {
            "following_enhanced": RecognizedText(
                _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING, 0.99989
            )
        },
        {
            "default_segments": (
                *_TERMINAL_DASH_WRAPPED_TWO_SEGMENTS[:-1],
                (669, 704),
            )
        },
    ],
)
def test_confirmed_terminal_dash_wrapped_two_plus_two_requires_evidence(
    overrides,
) -> None:
    words = terminal_dash_wrapped_two_words()

    assert (
        _recover_confirmed_terminal_dash_wrapped_two_plus_two_split(
            words,
            Image.new("RGB", (705, 29)),
            _TERMINAL_DASH_WRAPPED_TWO_LINE,
            ConfirmedTerminalDashWrappedTwoRecognizer(**overrides),
        )
        == words
    )


def test_confirmed_terminal_dash_wrapped_two_plus_two_requires_ctc_profile() -> None:
    words = terminal_dash_wrapped_two_words()
    recognizer = ConfirmedTerminalDashWrappedTwoRecognizer()
    recognizer.thresholds[0.0003] = ((0, 1),)

    assert (
        _recover_confirmed_terminal_dash_wrapped_two_plus_two_split(
            words,
            Image.new("RGB", (705, 29)),
            _TERMINAL_DASH_WRAPPED_TWO_LINE,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "preceding_text",
        "preceding_confidence",
        "candidate_pattern",
        "candidate_confidence",
        "box",
        "line",
        "crop",
    ],
)
def test_confirmed_terminal_dash_wrapped_two_plus_two_requires_profile(
    case: str,
) -> None:
    words = terminal_dash_wrapped_two_words()
    crop = Image.new("RGB", (705, 29))
    line_box = _TERMINAL_DASH_WRAPPED_TWO_LINE
    if case == "count":
        words = words[:-1]
    elif case == "preceding_text":
        text, box, confidence = words[0]
        words[0] = ("A" + text[1:], box, confidence)
    elif case == "preceding_confidence":
        text, box, _ = words[1]
        words[1] = (text, box, 0.9996)
    elif case == "candidate_pattern":
        words = terminal_dash_wrapped_two_words(
            candidate_text=(
                "/"
                + _TERMINAL_DASH_WRAPPED_TWO_TARGET
                + "-"
                + _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING
            )
        )
    elif case == "candidate_confidence":
        words = terminal_dash_wrapped_two_words(candidate_confidence=0.7296)
    elif case == "box":
        text, box, confidence = words[-1]
        words[-1] = (
            text,
            BoundingBox(box.left + 1, box.top, box.right, box.bottom),
            confidence,
        )
    elif case == "line":
        line_box = BoundingBox(78.44, 156.130435, 783.56, 184.304348)
    else:
        crop = Image.new("RGB", (704, 29))
    recognizer = ConfirmedTerminalDashWrappedTwoRecognizer()

    assert (
        _recover_confirmed_terminal_dash_wrapped_two_plus_two_split(
            words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


def test_engine_recovers_terminal_dash_wrapped_two_plus_two_segment() -> None:
    engine = PaddleOcrEngine(
        TerminalDashWrappedTwoDetector(),
        EngineTerminalDashWrappedTwoRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (900, 300)))

    assert document.lines[0].text == " ".join(
        (
            *_TERMINAL_DASH_WRAPPED_TWO_FIRST,
            _TERMINAL_DASH_WRAPPED_TWO_CORRECTED,
            _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING,
        )
    )
    assert [word.text for word in document.lines[0].eojeols] == [
        *_TERMINAL_DASH_WRAPPED_TWO_FIRST,
        _TERMINAL_DASH_WRAPPED_TWO_TARGET,
        _TERMINAL_DASH_WRAPPED_TWO_FOLLOWING,
    ]
    target_box = document.lines[0].eojeols[-2].box
    assert target_box.left == pytest.approx(612.44)
    assert target_box.right == pytest.approx(660.44)
    following_box = document.lines[0].eojeols[-1].box
    assert following_box.left == pytest.approx(693.44)
    assert following_box.right == pytest.approx(739.44)


_ISOLATED_DASH_FOUR_SEVEN_BOX = BoundingBox(
    79.08, 165.717391, 918.92, 225.586957
)
_ISOLATED_DASH_FOUR_SEVEN_TARGET = "".join(
    chr(0xC700 + index) for index in range(4)
)
_ISOLATED_DASH_FOUR_SEVEN_FOLLOWING = "".join(
    chr(0xC720 + index) for index in range(7)
)
_ISOLATED_DASH_FOUR_SEVEN_WHOLE = (
    chr(0x2018)
    + "-"
    + _ISOLATED_DASH_FOUR_SEVEN_TARGET
    + "-"
    + _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING
    + chr(0x2019)
)
_ISOLATED_DASH_FOUR_SEVEN_CANDIDATE = (
    chr(0x2014)
    + _ISOLATED_DASH_FOUR_SEVEN_TARGET
    + "-"
    + _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING
)
_ISOLATED_DASH_FOUR_SEVEN_SEGMENTS = {
    0.0001: (
        (0, 67),
        (66, 117),
        (116, 158),
        (157, 239),
        (238, 331),
        (330, 423),
        (422, 484),
        (483, 514),
        (513, 534),
        (533, 595),
        (594, 636),
        (635, 667),
        (666, 687),
        (686, 750),
    ),
    0.0003: (
        (0, 67),
        (66, 158),
        (157, 239),
        (238, 260),
        (259, 331),
        (330, 534),
        (533, 595),
        (594, 687),
        (686, 750),
    ),
    0.0005: (
        (0, 67),
        (66, 158),
        (157, 239),
        (238, 260),
        (259, 331),
        (330, 595),
        (594, 687),
        (686, 750),
    ),
    0.001: (
        (0, 260),
        (259, 331),
        (330, 372),
        (371, 595),
        (594, 687),
        (686, 750),
    ),
    0.002: ((0, 300), (300, 331), (330, 687), (686, 750)),
    0.003: ((0, 300), (300, 331), (330, 687), (686, 750)),
    0.005: ((0, 300), (300, 331), (330, 687), (686, 750)),
    0.007: ((0, 331), (330, 687), (686, 750)),
    0.01: ((0, 331), (330, 687), (686, 750)),
    0.015: ((0, 331), (330, 750)),
    0.02: ((0, 331), (330, 750)),
    0.03: ((0, 750),),
    0.04: ((0, 750),),
    0.05: ((0, 750),),
    0.07: ((0, 750),),
}


class ConfirmedIsolatedDashFourSevenRecognizer:
    def __init__(
        self,
        *,
        whole_direct: RecognizedText | None = None,
        whole_enhanced: RecognizedText | None = None,
        candidate_direct: RecognizedText | None = None,
        candidate_enhanced: RecognizedText | None = None,
        boundary_direct: RecognizedText | None = None,
        boundary_enhanced: RecognizedText | None = None,
        target_direct: RecognizedText | None = None,
        target_enhanced: RecognizedText | None = None,
        following_direct: RecognizedText | None = None,
        following_enhanced: RecognizedText | None = None,
        target_boundary_direct: RecognizedText | None = None,
        target_boundary_enhanced: RecognizedText | None = None,
        ctc_override: tuple[float, tuple[tuple[int, int], ...]] | None = None,
    ) -> None:
        self.whole_direct = whole_direct or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_WHOLE, 0.6944
        )
        self.whole_enhanced = whole_enhanced or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_WHOLE, 0.7859
        )
        self.candidate_direct = candidate_direct or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_CANDIDATE, 0.6919
        )
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_CANDIDATE, 0.6744
        )
        self.boundary_direct = boundary_direct or RecognizedText(
            chr(0x2014), 0.5198
        )
        self.boundary_enhanced = boundary_enhanced or RecognizedText(
            chr(0x2014), 0.5164
        )
        self.target_direct = target_direct or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_TARGET, 0.99999
        )
        self.target_enhanced = target_enhanced or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_TARGET, 0.99999
        )
        self.following_direct = following_direct or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING, 0.9998
        )
        self.following_enhanced = following_enhanced or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING, 0.9998
        )
        self.target_boundary_direct = target_boundary_direct or RecognizedText(
            _ISOLATED_DASH_FOUR_SEVEN_TARGET + chr(0x2014), 0.744
        )
        self.target_boundary_enhanced = (
            target_boundary_enhanced
            or RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_TARGET + chr(0x2014), 0.638
            )
        )
        self.ctc_override = ctc_override
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (840, 61) and space_threshold is None:
            return ((44, 794),)
        if image.size != (750, 61) or space_threshold is None:
            return ()
        if self.ctc_override is not None and space_threshold == self.ctc_override[0]:
            return self.ctc_override[1]
        return _ISOLATED_DASH_FOUR_SEVEN_SEGMENTS[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        if image.size == (840, 61):
            return self.whole_direct
        if image.size == (1680, 122):
            return self.whole_enhanced
        if image.size == (750, 61):
            return self.candidate_direct
        if image.size == (1500, 122):
            return self.candidate_enhanced
        original_width = image.width if image.height == 61 else image.width // 2
        enhanced = image.height == 122
        if original_width <= 60:
            return self.boundary_enhanced if enhanced else self.boundary_direct
        if 230 <= original_width <= 240:
            return self.target_enhanced if enhanced else self.target_direct
        if 285 <= original_width <= 300:
            return (
                self.target_boundary_enhanced
                if enhanced
                else self.target_boundary_direct
            )
        if 395 <= original_width <= 405:
            return (
                self.following_enhanced if enhanced else self.following_direct
            )
        return RecognizedText("", 0.0)


def isolated_dash_four_seven_words(
    *,
    text: str = _ISOLATED_DASH_FOUR_SEVEN_WHOLE,
    confidence: float = 0.7859,
    box: BoundingBox = _ISOLATED_DASH_FOUR_SEVEN_BOX,
) -> list[tuple[str, BoundingBox, float]]:
    return [(text, box, confidence)]


def test_confirmed_isolated_dash_wrapped_four_plus_seven_recovers() -> None:
    recovered = (
        _recover_confirmed_isolated_dash_wrapped_four_plus_seven_split(
            isolated_dash_four_seven_words(),
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
            ConfirmedIsolatedDashFourSevenRecognizer(),
        )
    )

    assert recovered == [
        (
            chr(0x2014)
            + _ISOLATED_DASH_FOUR_SEVEN_TARGET
            + chr(0x2014),
            BoundingBox(123.08, 165.717391, 463.08, 225.586957),
            0.5164,
        ),
        (
            _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING,
            BoundingBox(472.08, 165.717391, 873.08, 225.586957),
            0.6744,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "whole_direct": RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_WHOLE, 0.6942
            )
        },
        {
            "candidate_enhanced": RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_CANDIDATE[:-1], 0.9
            )
        },
        {"boundary_direct": RecognizedText(chr(0x2014), 0.5196)},
        {
            "target_enhanced": RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_TARGET[:-1], 1.0
            )
        },
        {
            "following_direct": RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING, 0.9997
            )
        },
        {
            "target_boundary_enhanced": RecognizedText(
                _ISOLATED_DASH_FOUR_SEVEN_TARGET + chr(0x2014), 0.6371
            )
        },
        {"ctc_override": (0.015, ((0, 750),))},
    ],
)
def test_confirmed_isolated_dash_four_plus_seven_requires_crop_evidence(
    overrides,
) -> None:
    words = isolated_dash_four_seven_words()

    assert (
        _recover_confirmed_isolated_dash_wrapped_four_plus_seven_split(
            words,
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
            ConfirmedIsolatedDashFourSevenRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "crop", "line_box"),
    [
        (
            isolated_dash_four_seven_words(
                text="A" + _ISOLATED_DASH_FOUR_SEVEN_WHOLE[1:]
            ),
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
        ),
        (
            isolated_dash_four_seven_words(confidence=0.7857),
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
        ),
        (
            isolated_dash_four_seven_words(
                box=BoundingBox(79.08, 165.717391, 918.91, 225.586957)
            ),
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
        ),
        (
            isolated_dash_four_seven_words(),
            Image.new("RGB", (839, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
        ),
        (
            isolated_dash_four_seven_words() * 2,
            Image.new("RGB", (840, 61)),
            _ISOLATED_DASH_FOUR_SEVEN_BOX,
        ),
    ],
)
def test_confirmed_isolated_dash_four_plus_seven_requires_exact_profile(
    words,
    crop,
    line_box,
) -> None:
    recognizer = ConfirmedIsolatedDashFourSevenRecognizer()

    assert (
        _recover_confirmed_isolated_dash_wrapped_four_plus_seven_split(
            words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class IsolatedDashFourSevenDetector:
    def detect(self, _image):
        return (
            DetectedRegion(_ISOLATED_DASH_FOUR_SEVEN_BOX, 0.99382),
        )


def test_engine_recovers_isolated_dash_wrapped_four_plus_seven_segment() -> None:
    engine = PaddleOcrEngine(
        IsolatedDashFourSevenDetector(),
        ConfirmedIsolatedDashFourSevenRecognizer(),
    )

    document = engine.recognize(Image.new("RGB", (1200, 400)))

    line = document.lines[0]
    assert line.text == (
        chr(0x2014)
        + _ISOLATED_DASH_FOUR_SEVEN_TARGET
        + chr(0x2014)
        + " "
        + _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING
    )
    assert [word.text for word in line.eojeols] == [
        _ISOLATED_DASH_FOUR_SEVEN_TARGET,
        _ISOLATED_DASH_FOUR_SEVEN_FOLLOWING,
    ]
    target_box = line.eojeols[0].box
    assert target_box.left == pytest.approx(179.746667)
    assert target_box.right == pytest.approx(406.413333)
    following_box = line.eojeols[1].box
    assert following_box.left == pytest.approx(472.08)
    assert following_box.right == pytest.approx(873.08)


_INTERNAL_DASH_TWO_LINE = BoundingBox(86.16, 151.043478, 750.84, 184.5)
_INTERNAL_DASH_TWO_SEGMENTS = (
    (35, 67),
    (81, 146),
    (157, 398),
    (409, 475),
    (488, 550),
    (567, 627),
)
_INTERNAL_DASH_TWO_PREFIX = "".join(chr(0xC800 + index) for index in range(3))
_INTERNAL_DASH_TWO_TARGET = "".join(chr(0xC820 + index) for index in range(2))
_INTERNAL_DASH_TWO_RAW_TEXTS = (
    chr(0xC840),
    "".join(chr(0xC850 + index) for index in range(2)),
    _INTERNAL_DASH_TWO_PREFIX
    + "-"
    + _INTERNAL_DASH_TWO_TARGET
    + chr(0x2014),
    "".join(chr(0xC860 + index) for index in range(2)),
    "2" + chr(0xC870) + ",",
    "".join(chr(0xC880 + index) for index in range(2)),
)
_INTERNAL_DASH_TWO_CONFIDENCES = (
    0.999883,
    0.999979,
    0.668947,
    0.99893,
    0.972564,
    0.999967,
)
_INTERNAL_DASH_TWO_CTC = {
    0.0001: (
        (0, 100),
        (111, 122),
        (121, 156),
        (155, 179),
        (178, 206),
        (210, 219),
        (218, 241),
    ),
    0.0003: ((0, 100), (111, 122), (121, 179), (178, 241)),
    0.0005: ((0, 100), (111, 179), (178, 241)),
    0.001: ((0, 100), (111, 179), (178, 241)),
    0.002: ((0, 100), (111, 179), (178, 241)),
    0.003: ((0, 100), (111, 179), (178, 241)),
    0.005: ((0, 100), (111, 179), (178, 241)),
    0.007: ((0, 100), (111, 179), (178, 241)),
    0.01: ((0, 100), (111, 241)),
    0.015: ((0, 100), (111, 241)),
    0.02: ((0, 100), (111, 241)),
    0.03: ((0, 241),),
    0.04: ((0, 241),),
    0.05: ((0, 241),),
    0.07: ((0, 241),),
}


def internal_dash_two_crop() -> Image.Image:
    crop = Image.new("RGB", (665, 34))
    for (left, right), intensity in zip(
        _INTERNAL_DASH_TWO_SEGMENTS,
        (10, 20, 100, 30, 40, 50),
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 34))
    return crop


def internal_dash_two_words(
    *,
    candidate_text: str = _INTERNAL_DASH_TWO_RAW_TEXTS[2],
    candidate_confidence: float = _INTERNAL_DASH_TWO_CONFIDENCES[2],
    candidate_box: BoundingBox | None = None,
    fourth_text: str = _INTERNAL_DASH_TWO_RAW_TEXTS[4],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _INTERNAL_DASH_TWO_SEGMENTS,
            _INTERNAL_DASH_TWO_RAW_TEXTS,
            _INTERNAL_DASH_TWO_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 2:
            text = candidate_text
            confidence = candidate_confidence
        elif index == 4:
            text = fourth_text
        box = BoundingBox(
            _INTERNAL_DASH_TWO_LINE.left + left,
            _INTERNAL_DASH_TWO_LINE.top,
            _INTERNAL_DASH_TWO_LINE.left + right,
            _INTERNAL_DASH_TWO_LINE.bottom,
        )
        if index == 2 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedInternalDashTwoRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = _INTERNAL_DASH_TWO_SEGMENTS,
        ctc_override: tuple[float, tuple[tuple[int, int], ...]] | None = None,
        candidate_direct: RecognizedText | None = None,
        candidate_enhanced: RecognizedText | None = None,
        prefix_variant: RecognizedText | None = None,
        ctc_wrapper_direct: RecognizedText | None = None,
        ctc_wrapper_enhanced: RecognizedText | None = None,
        wrapper_direct: RecognizedText | None = None,
        wrapper_enhanced: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        boundary_variant: RecognizedText | None = None,
    ) -> None:
        wrapper = (
            chr(0x2014)
            + _INTERNAL_DASH_TWO_TARGET
            + chr(0x2014)
        )
        enhanced_wrapper = chr(0x2014) + _INTERNAL_DASH_TWO_TARGET + "-"
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_direct = candidate_direct or RecognizedText(
            _INTERNAL_DASH_TWO_RAW_TEXTS[2],
            0.66895,
        )
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _INTERNAL_DASH_TWO_RAW_TEXTS[2][:-1] + "-",
            0.59685,
        )
        self.prefix_variant = prefix_variant or RecognizedText(
            _INTERNAL_DASH_TWO_PREFIX,
            0.99995,
        )
        self.ctc_wrapper_direct = ctc_wrapper_direct or RecognizedText(
            wrapper,
            0.7736,
        )
        self.ctc_wrapper_enhanced = ctc_wrapper_enhanced or RecognizedText(
            "-" + _INTERNAL_DASH_TWO_TARGET + "-",
            0.67,
        )
        self.wrapper_direct = wrapper_direct or RecognizedText(wrapper, 0.9)
        self.wrapper_enhanced = wrapper_enhanced or RecognizedText(
            enhanced_wrapper,
            0.8,
        )
        self.target_variant = target_variant or RecognizedText(
            _INTERNAL_DASH_TWO_TARGET,
            0.9999,
        )
        self.boundary_variant = boundary_variant or RecognizedText(
            chr(0x2014),
            0.9,
        )
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (665, 34) and space_threshold is None:
            return self.default_segments
        if image.size != (241, 34) or space_threshold is None:
            return ()
        if self.ctc_override is not None and space_threshold == self.ctc_override[0]:
            return self.ctc_override[1]
        return _INTERNAL_DASH_TWO_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 68
        segment_reads = {
            10: (
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[0], 0.999883),
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[0], 0.999428),
            ),
            20: (
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[1], 0.999979),
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[1], 0.99998),
            ),
            30: (
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[3], 0.99893),
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[3], 0.999098),
            ),
            40: (
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[4], 0.972564),
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[4], 0.985471),
            ),
            50: (
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[5], 0.999967),
                RecognizedText(_INTERNAL_DASH_TWO_RAW_TEXTS[5], 0.999975),
            ),
        }
        if intensity in segment_reads:
            return segment_reads[intensity][1 if enhanced else 0]
        if intensity != 100:
            return RecognizedText("", 0.0)
        if image.size == (241, 34):
            return self.candidate_direct
        if image.size == (482, 68):
            return self.candidate_enhanced
        original_width = image.width // 2 if enhanced else image.width
        if 95 <= original_width <= 103:
            return self.prefix_variant
        if original_width == 130:
            return (
                self.ctc_wrapper_enhanced
                if enhanced
                else self.ctc_wrapper_direct
            )
        if 125 <= original_width <= 145:
            return self.wrapper_enhanced if enhanced else self.wrapper_direct
        if 60 <= original_width <= 75:
            return self.target_variant
        if 30 <= original_width <= 45:
            return self.boundary_variant
        return RecognizedText("", 0.0)


def test_confirmed_internal_dash_wrapped_two_recovers() -> None:
    words = internal_dash_two_words()
    recovered = _recover_confirmed_internal_dash_wrapped_two_split(
        words,
        words.copy(),
        internal_dash_two_crop(),
        _INTERNAL_DASH_TWO_LINE,
        ConfirmedInternalDashTwoRecognizer(),
    )

    assert recovered[:2] == words[:2]
    assert recovered[4:] == words[3:]
    assert recovered[2] == (
        _INTERNAL_DASH_TWO_PREFIX,
        BoundingBox(
            _INTERNAL_DASH_TWO_LINE.left + 157,
            _INTERNAL_DASH_TWO_LINE.top,
            _INTERNAL_DASH_TWO_LINE.left + 257,
            _INTERNAL_DASH_TWO_LINE.bottom,
        ),
        0.59685,
    )
    assert recovered[3] == (
        chr(0x2014) + _INTERNAL_DASH_TWO_TARGET + chr(0x2014),
        BoundingBox(
            _INTERNAL_DASH_TWO_LINE.left + 268,
            _INTERNAL_DASH_TWO_LINE.top,
            _INTERNAL_DASH_TWO_LINE.left + 398,
            _INTERNAL_DASH_TWO_LINE.bottom,
        ),
        0.59685,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _INTERNAL_DASH_TWO_RAW_TEXTS[2],
                0.9,
            )
        },
        {
            "ctc_wrapper_direct": RecognizedText(
                "-" + _INTERNAL_DASH_TWO_TARGET + "-",
                0.9,
            )
        },
        {
            "wrapper_enhanced": RecognizedText(
                "-" + _INTERNAL_DASH_TWO_TARGET + "-",
                0.9,
            )
        },
        {
            "target_variant": RecognizedText(
                _INTERNAL_DASH_TWO_TARGET,
                0.9998,
            )
        },
        {"boundary_variant": RecognizedText("-", 0.9)},
        {
            "prefix_variant": RecognizedText(
                _INTERNAL_DASH_TWO_PREFIX[:-1],
                1.0,
            )
        },
        {"ctc_override": (0.01, ((0, 241),))},
    ],
)
def test_confirmed_internal_dash_wrapped_two_requires_crop_evidence(
    overrides,
) -> None:
    words = internal_dash_two_words()

    assert (
        _recover_confirmed_internal_dash_wrapped_two_split(
            words,
            words.copy(),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            internal_dash_two_words() * 2,
            internal_dash_two_words(),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(candidate_text="A" + _INTERNAL_DASH_TWO_RAW_TEXTS[2][1:]),
            internal_dash_two_words(candidate_text="A" + _INTERNAL_DASH_TWO_RAW_TEXTS[2][1:]),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(candidate_confidence=0.6688),
            internal_dash_two_words(candidate_confidence=0.6688),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(fourth_text="A2,"),
            internal_dash_two_words(fourth_text="A2,"),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(
                candidate_box=BoundingBox(243.16, 151.043478, 483.16, 184.5)
            ),
            internal_dash_two_words(
                candidate_box=BoundingBox(243.16, 151.043478, 483.16, 184.5)
            ),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(),
            internal_dash_two_words(),
            Image.new("RGB", (664, 34)),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(),
            internal_dash_two_words(),
            internal_dash_two_crop(),
            BoundingBox(86.16, 151.043478, 750.86, 184.5),
            ConfirmedInternalDashTwoRecognizer(),
        ),
        (
            internal_dash_two_words(),
            internal_dash_two_words(),
            internal_dash_two_crop(),
            _INTERNAL_DASH_TWO_LINE,
            ConfirmedInternalDashTwoRecognizer(
                default_segments=_INTERNAL_DASH_TWO_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_internal_dash_wrapped_two_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_internal_dash_wrapped_two_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class InternalDashTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(_INTERNAL_DASH_TWO_LINE, 0.9927),)


def test_engine_recovers_internal_dash_wrapped_two_segment() -> None:
    image = Image.new("RGB", (800, 350))
    image.paste(internal_dash_two_crop(), (86, 151))
    engine = PaddleOcrEngine(
        InternalDashTwoDetector(),
        ConfirmedInternalDashTwoRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 23
    assert [len(word.text) for word in line.eojeols] == [1, 2, 3, 2, 2, 2, 2]
    assert line.eojeols[3].text == _INTERNAL_DASH_TWO_TARGET
    assert line.eojeols[3].box.left == pytest.approx(386.66)
    assert line.eojeols[3].box.right == pytest.approx(451.66)


_INTERNAL_PAIRED_TWO_LINE = BoundingBox(62.16, 235.76, 1132.84, 279.78)
_INTERNAL_PAIRED_TWO_SEGMENTS = (
    (59, 194),
    (211, 303),
    (315, 602),
    (617, 755),
    (767, 853),
    (871, 1010),
)
_INTERNAL_PAIRED_TWO_PREFIX = "".join(
    chr(value) for value in (0xAC00, 0xB098, 0xB2E4)
)
_INTERNAL_PAIRED_TWO_TARGET = "".join(chr(value) for value in (0xC5C4, 0xC120))
_INTERNAL_PAIRED_TWO_WRAPPER = (
    chr(0x201C) + _INTERNAL_PAIRED_TWO_TARGET + chr(0x201D)
)
_INTERNAL_PAIRED_TWO_RAW_TEXTS = (
    "".join(chr(value) for value in (0xB77C, 0xB9C8, 0xBC14)),
    "".join(chr(value) for value in (0xC0AC, 0xC544)),
    _INTERNAL_PAIRED_TWO_PREFIX
    + chr(0x201C)
    + _INTERNAL_PAIRED_TWO_TARGET
    + '"',
    "".join(chr(value) for value in (0xC790, 0xCC28, 0xCE74)),
    "".join(chr(value) for value in (0xD0C0, 0xD30C)),
    "".join(chr(value) for value in (0xD558, 0xAC70, 0xB108)),
)
_INTERNAL_PAIRED_TWO_CONFIDENCES = (
    0.999944,
    0.999965,
    0.63105,
    0.999903,
    0.99945,
    0.99999,
)
_INTERNAL_PAIRED_TWO_CTC = {
    **{
        threshold: ((0, 132), (131, 287))
        for threshold in (
            0.0001,
            0.0003,
            0.0005,
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
        )
    },
    **{
        threshold: ((0, 287),)
        for threshold in (0.03, 0.04, 0.05, 0.07)
    },
}


def internal_paired_two_crop() -> Image.Image:
    crop = Image.new("RGB", (1071, 45))
    for intensity, (left, right) in zip(
        (10, 20, 100, 30, 40, 50),
        _INTERNAL_PAIRED_TWO_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 45))
    return crop


def internal_paired_two_words(
    *,
    candidate_text: str = _INTERNAL_PAIRED_TWO_RAW_TEXTS[2],
    candidate_confidence: float = _INTERNAL_PAIRED_TWO_CONFIDENCES[2],
    candidate_box: BoundingBox | None = None,
    fourth_text: str = _INTERNAL_PAIRED_TWO_RAW_TEXTS[4],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _INTERNAL_PAIRED_TWO_SEGMENTS,
            _INTERNAL_PAIRED_TWO_RAW_TEXTS,
            _INTERNAL_PAIRED_TWO_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 2:
            text = candidate_text
            confidence = candidate_confidence
        elif index == 4:
            text = fourth_text
        box = BoundingBox(
            _INTERNAL_PAIRED_TWO_LINE.left + left,
            _INTERNAL_PAIRED_TWO_LINE.top,
            _INTERNAL_PAIRED_TWO_LINE.left + right,
            _INTERNAL_PAIRED_TWO_LINE.bottom,
        )
        if index == 2 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedInternalPairedTwoRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _INTERNAL_PAIRED_TWO_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _INTERNAL_PAIRED_TWO_RAW_TEXTS[2],
            0.59,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _INTERNAL_PAIRED_TWO_WRAPPER,
            0.8,
        )
        self.target_variant = target_variant or RecognizedText(
            _INTERNAL_PAIRED_TWO_TARGET,
            0.9995,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.8)
        self.closing_variant = closing_variant or RecognizedText('"', 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (1071, 45) and space_threshold is None:
            return self.default_segments
        if image.size != (287, 45) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _INTERNAL_PAIRED_TWO_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 90
        segment_reads = {
            10: RecognizedText(_INTERNAL_PAIRED_TWO_RAW_TEXTS[0], 0.999944),
            20: RecognizedText(_INTERNAL_PAIRED_TWO_RAW_TEXTS[1], 0.999965),
            30: RecognizedText(_INTERNAL_PAIRED_TWO_RAW_TEXTS[3], 0.999903),
            40: RecognizedText(_INTERNAL_PAIRED_TWO_RAW_TEXTS[4], 0.99945),
            50: RecognizedText(_INTERNAL_PAIRED_TWO_RAW_TEXTS[5], 0.99999),
        }
        if intensity in segment_reads:
            return segment_reads[intensity]
        if intensity != 100:
            return RecognizedText("", 0.0)
        if image.size == (287, 45):
            return RecognizedText(
                _INTERNAL_PAIRED_TWO_RAW_TEXTS[2],
                0.63105,
            )
        if image.size == (574, 90):
            return self.candidate_enhanced
        original_width = image.width // 2 if enhanced else image.width
        if 128 <= original_width <= 136:
            return RecognizedText(_INTERNAL_PAIRED_TWO_PREFIX, 0.9998)
        if 154 <= original_width <= 159:
            return self.wrapper_variant
        if 91 <= original_width <= 97:
            return self.target_variant
        if 34 <= original_width <= 46:
            return self.opening_variant
        if 20 <= original_width <= 27:
            return self.closing_variant
        return RecognizedText("", 0.0)


def test_confirmed_internal_paired_wrapped_two_recovers() -> None:
    words = internal_paired_two_words()
    recovered = _recover_confirmed_internal_paired_wrapped_two_split(
        words,
        words.copy(),
        internal_paired_two_crop(),
        _INTERNAL_PAIRED_TWO_LINE,
        ConfirmedInternalPairedTwoRecognizer(),
    )

    assert recovered[:2] == words[:2]
    assert recovered[4:] == words[3:]
    assert recovered[2] == (
        _INTERNAL_PAIRED_TWO_PREFIX,
        BoundingBox(
            _INTERNAL_PAIRED_TWO_LINE.left + 315,
            235.76,
            _INTERNAL_PAIRED_TWO_LINE.left + 447,
            279.78,
        ),
        0.59,
    )
    assert recovered[3] == (
        _INTERNAL_PAIRED_TWO_WRAPPER,
        BoundingBox(
            _INTERNAL_PAIRED_TWO_LINE.left + 446,
            235.76,
            _INTERNAL_PAIRED_TWO_LINE.left + 602,
            279.78,
        ),
        0.59,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _INTERNAL_PAIRED_TWO_RAW_TEXTS[2][:-2]
                + "?"
                + _INTERNAL_PAIRED_TWO_RAW_TEXTS[2][-1],
                0.9,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _INTERNAL_PAIRED_TWO_TARGET + '"',
                0.9,
            )
        },
        {
            "target_variant": RecognizedText(
                _INTERNAL_PAIRED_TWO_TARGET[:1],
                1.0,
            )
        },
        {"opening_variant": RecognizedText(chr(0x201C), 0.9)},
        {"closing_variant": RecognizedText('"', 0.57)},
        {"ctc_override": (0.01, ((0, 287),))},
    ],
)
def test_confirmed_internal_paired_two_requires_crop_evidence(
    overrides,
) -> None:
    words = internal_paired_two_words()

    assert (
        _recover_confirmed_internal_paired_wrapped_two_split(
            words,
            words.copy(),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            internal_paired_two_words() * 2,
            internal_paired_two_words(),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(
                candidate_text=(
                    "A" + _INTERNAL_PAIRED_TWO_RAW_TEXTS[2][1:]
                )
            ),
            internal_paired_two_words(
                candidate_text=(
                    "A" + _INTERNAL_PAIRED_TWO_RAW_TEXTS[2][1:]
                )
            ),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(candidate_confidence=0.6309),
            internal_paired_two_words(candidate_confidence=0.6309),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(fourth_text="A?"),
            internal_paired_two_words(fourth_text="A?"),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(
                candidate_box=BoundingBox(377.16, 235.76, 603.16, 279.78)
            ),
            internal_paired_two_words(
                candidate_box=BoundingBox(377.16, 235.76, 603.16, 279.78)
            ),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(),
            internal_paired_two_words(),
            Image.new("RGB", (1070, 45)),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(),
            internal_paired_two_words(),
            internal_paired_two_crop(),
            BoundingBox(62.16, 235.76, 1132.86, 279.78),
            ConfirmedInternalPairedTwoRecognizer(),
        ),
        (
            internal_paired_two_words(),
            internal_paired_two_words(),
            internal_paired_two_crop(),
            _INTERNAL_PAIRED_TWO_LINE,
            ConfirmedInternalPairedTwoRecognizer(
                default_segments=_INTERNAL_PAIRED_TWO_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_internal_paired_two_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_internal_paired_wrapped_two_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class InternalPairedTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(_INTERNAL_PAIRED_TWO_LINE, 0.9921),)


def test_engine_recovers_internal_paired_wrapped_two_segment() -> None:
    image = Image.new("RGB", (1200, 400))
    image.paste(internal_paired_two_crop(), (62, 235))
    engine = PaddleOcrEngine(
        InternalPairedTwoDetector(),
        ConfirmedInternalPairedTwoRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 26
    assert [len(word.text) for word in line.eojeols] == [3, 2, 3, 2, 3, 2, 3]
    assert line.eojeols[3].text == _INTERNAL_PAIRED_TWO_TARGET
    assert line.eojeols[3].box.left == pytest.approx(547.16)
    assert line.eojeols[3].box.right == pytest.approx(625.16)

_INTERNAL_PAIRED_THREE_LINE = BoundingBox(86.2, 153.39, 706.8, 185.09)
_INTERNAL_PAIRED_THREE_SEGMENTS = (
    (37, 59),
    (70, 94),
    (105, 234),
    (244, 325),
    (334, 522),
    (533, 585),
)
_INTERNAL_PAIRED_THREE_PREFIX = "".join(
    chr(value) for value in (0xAC00, 0xB098, 0xB2E4)
)
_INTERNAL_PAIRED_THREE_TARGET = "".join(
    chr(value) for value in (0xC5C4, 0xC120, 0xD0DD)
)
_INTERNAL_PAIRED_THREE_WRAPPER = (
    chr(0x201C) + _INTERNAL_PAIRED_THREE_TARGET + chr(0x201D)
)
_INTERNAL_PAIRED_THREE_RAW_TEXTS = (
    chr(0xB77C),
    chr(0xB9C8),
    "".join(chr(value) for value in (0xBC14, 0xC0AC, 0xC544, 0xC790, 0xCC28)),
    "".join(chr(value) for value in (0xCE74, 0xD0C0, 0xD30C)),
    _INTERNAL_PAIRED_THREE_PREFIX
    + chr(0x201C)
    + _INTERNAL_PAIRED_THREE_TARGET
    + '"',
    "".join(chr(value) for value in (0xD558, 0xAC70)),
)
_INTERNAL_PAIRED_THREE_CONFIDENCES = (
    0.999926,
    0.9999,
    0.998613,
    0.998778,
    0.60985,
    0.999917,
)
_INTERNAL_PAIRED_THREE_CTC = {
    **{
        threshold: ((0, 78), (87, 188))
        for threshold in (
            0.0001,
            0.0003,
            0.0005,
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
        )
    },
    **{
        threshold: ((0, 188),)
        for threshold in (0.03, 0.04, 0.05, 0.07)
    },
}


def internal_paired_three_crop() -> Image.Image:
    crop = Image.new("RGB", (621, 33))
    for intensity, (left, right) in zip(
        (10, 20, 30, 40, 100, 50),
        _INTERNAL_PAIRED_THREE_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 33))
    candidate_left = _INTERNAL_PAIRED_THREE_SEGMENTS[4][0]
    crop.paste((90, 90, 90), (candidate_left + 83, 0, candidate_left + 97, 33))
    crop.paste((110, 110, 110), (candidate_left + 101, 0, candidate_left + 175, 33))
    crop.paste((120, 120, 120), (candidate_left + 179, 0, candidate_left + 188, 33))
    return crop


def internal_paired_three_words(
    *,
    selected: bool,
    candidate_text: str = _INTERNAL_PAIRED_THREE_RAW_TEXTS[4],
    candidate_confidence: float | None = None,
    candidate_box: BoundingBox | None = None,
    fourth_text: str = _INTERNAL_PAIRED_THREE_RAW_TEXTS[3],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _INTERNAL_PAIRED_THREE_SEGMENTS,
            _INTERNAL_PAIRED_THREE_RAW_TEXTS,
            _INTERNAL_PAIRED_THREE_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 4:
            text = candidate_text
            confidence = (
                candidate_confidence
                if candidate_confidence is not None
                else (0.68085 if selected else confidence)
            )
        elif index == 3:
            text = fourth_text
        box = BoundingBox(
            _INTERNAL_PAIRED_THREE_LINE.left + left,
            _INTERNAL_PAIRED_THREE_LINE.top,
            _INTERNAL_PAIRED_THREE_LINE.left + right,
            _INTERNAL_PAIRED_THREE_LINE.bottom,
        )
        if index == 4 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedInternalPairedThreeRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _INTERNAL_PAIRED_THREE_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _INTERNAL_PAIRED_THREE_RAW_TEXTS[4],
            0.68085,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _INTERNAL_PAIRED_THREE_WRAPPER,
            0.82,
        )
        self.target_variant = target_variant or RecognizedText(
            _INTERNAL_PAIRED_THREE_TARGET,
            0.9998,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.95)
        self.closing_variant = closing_variant or RecognizedText('"', 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (621, 33) and space_threshold is None:
            return self.default_segments
        if image.size != (188, 33) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _INTERNAL_PAIRED_THREE_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 66
        segment_reads = {
            10: RecognizedText(_INTERNAL_PAIRED_THREE_RAW_TEXTS[0], 0.999926),
            20: RecognizedText(_INTERNAL_PAIRED_THREE_RAW_TEXTS[1], 0.9999),
            30: RecognizedText(_INTERNAL_PAIRED_THREE_RAW_TEXTS[2], 0.998613),
            40: RecognizedText(_INTERNAL_PAIRED_THREE_RAW_TEXTS[3], 0.998778),
            50: RecognizedText(_INTERNAL_PAIRED_THREE_RAW_TEXTS[5], 0.999917),
        }
        if intensity in segment_reads:
            return segment_reads[intensity]
        if image.size == (188, 33):
            return RecognizedText(
                _INTERNAL_PAIRED_THREE_RAW_TEXTS[4],
                0.60985,
            )
        if image.size == (376, 66):
            return self.candidate_enhanced
        original_width = image.width // 2 if enhanced else image.width
        if 99 <= original_width <= 104:
            return self.wrapper_variant
        if 74 <= original_width <= 84:
            if intensity > (100 if enhanced else 105):
                return self.target_variant
            return RecognizedText(_INTERNAL_PAIRED_THREE_PREFIX, 0.9997)
        if 9 <= original_width <= 18:
            if intensity > (100 if enhanced else 115):
                return self.closing_variant
            return self.opening_variant
        return RecognizedText("", 0.0)


def test_confirmed_internal_paired_wrapped_three_recovers() -> None:
    words = internal_paired_three_words(selected=True)
    raw_words = internal_paired_three_words(selected=False)
    recovered = _recover_confirmed_internal_paired_wrapped_three_split(
        words,
        raw_words,
        internal_paired_three_crop(),
        _INTERNAL_PAIRED_THREE_LINE,
        ConfirmedInternalPairedThreeRecognizer(),
    )

    assert recovered[:4] == words[:4]
    assert recovered[6:] == words[5:]
    assert recovered[4] == (
        _INTERNAL_PAIRED_THREE_PREFIX,
        BoundingBox(
            _INTERNAL_PAIRED_THREE_LINE.left + 334,
            153.39,
            _INTERNAL_PAIRED_THREE_LINE.left + 412,
            185.09,
        ),
        0.60985,
    )
    assert recovered[5] == (
        _INTERNAL_PAIRED_THREE_WRAPPER,
        BoundingBox(
            _INTERNAL_PAIRED_THREE_LINE.left + 421,
            153.39,
            _INTERNAL_PAIRED_THREE_LINE.left + 522,
            185.09,
        ),
        0.60985,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _INTERNAL_PAIRED_THREE_RAW_TEXTS[4][:-2]
                + "?"
                + _INTERNAL_PAIRED_THREE_RAW_TEXTS[4][-1],
                0.9,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _INTERNAL_PAIRED_THREE_TARGET + '"',
                0.9,
            )
        },
        {
            "target_variant": RecognizedText(
                _INTERNAL_PAIRED_THREE_TARGET[:2],
                1.0,
            )
        },
        {"opening_variant": RecognizedText(chr(0x201C), 0.99)},
        {"closing_variant": RecognizedText('"', 0.55)},
        {"ctc_override": (0.01, ((0, 188),))},
    ],
)
def test_confirmed_internal_paired_three_requires_crop_evidence(
    overrides,
) -> None:
    words = internal_paired_three_words(selected=True)

    assert (
        _recover_confirmed_internal_paired_wrapped_three_split(
            words,
            internal_paired_three_words(selected=False),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            internal_paired_three_words(selected=True) * 2,
            internal_paired_three_words(selected=False),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(
                selected=True,
                candidate_text=(
                    "A" + _INTERNAL_PAIRED_THREE_RAW_TEXTS[4][1:]
                ),
            ),
            internal_paired_three_words(
                selected=False,
                candidate_text=(
                    "A" + _INTERNAL_PAIRED_THREE_RAW_TEXTS[4][1:]
                ),
            ),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(
                selected=True,
                candidate_confidence=0.6807,
            ),
            internal_paired_three_words(selected=False),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(selected=True),
            internal_paired_three_words(
                selected=False,
                candidate_confidence=0.6097,
            ),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(selected=True, fourth_text="A?"),
            internal_paired_three_words(selected=False, fourth_text="A?"),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(
                selected=True,
                candidate_box=BoundingBox(420.2, 153.39, 607.2, 185.09),
            ),
            internal_paired_three_words(
                selected=False,
                candidate_box=BoundingBox(420.2, 153.39, 607.2, 185.09),
            ),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(selected=True),
            internal_paired_three_words(selected=False),
            Image.new("RGB", (620, 33)),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(selected=True),
            internal_paired_three_words(selected=False),
            internal_paired_three_crop(),
            BoundingBox(86.2, 153.39, 706.82, 185.09),
            ConfirmedInternalPairedThreeRecognizer(),
        ),
        (
            internal_paired_three_words(selected=True),
            internal_paired_three_words(selected=False),
            internal_paired_three_crop(),
            _INTERNAL_PAIRED_THREE_LINE,
            ConfirmedInternalPairedThreeRecognizer(
                default_segments=_INTERNAL_PAIRED_THREE_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_internal_paired_three_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_internal_paired_wrapped_three_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class InternalPairedThreeDetector:
    def detect(self, _image):
        return (DetectedRegion(_INTERNAL_PAIRED_THREE_LINE, 0.9902),)


def test_engine_recovers_internal_paired_wrapped_three_segment() -> None:
    image = Image.new("RGB", (800, 300))
    image.paste(internal_paired_three_crop(), (86, 153))
    engine = PaddleOcrEngine(
        InternalPairedThreeDetector(),
        ConfirmedInternalPairedThreeRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 26
    assert [len(word.text) for word in line.eojeols] == [1, 1, 5, 3, 3, 3, 2]
    assert line.eojeols[5].text == _INTERNAL_PAIRED_THREE_TARGET
    assert line.eojeols[5].box.left == pytest.approx(527.4)
    assert line.eojeols[5].box.right == pytest.approx(588.0)


_LEADING_PAIRED_THREE_LINE = BoundingBox(
    97.32,
    165.71739130434784,
    701.68,
    234.3913043478261,
)
_LEADING_PAIRED_THREE_SEGMENTS = ((24, 378), (397, 581))
_LEADING_PAIRED_THREE_PREFIX = "".join(map(chr, (0xAC00, 0xB098)))
_LEADING_PAIRED_THREE_TARGET = "".join(map(chr, (0xB2E4, 0xB77C, 0xB9C8)))
_LEADING_PAIRED_THREE_NEIGHBOR = "".join(map(chr, (0xBC14, 0xC0AC, 0xC544)))
_LEADING_PAIRED_THREE_WRAPPER = (
    chr(0x2018) + _LEADING_PAIRED_THREE_TARGET + chr(0x2019)
)
_LEADING_PAIRED_THREE_RAW = (
    _LEADING_PAIRED_THREE_PREFIX
    + chr(0x2018)
    + _LEADING_PAIRED_THREE_TARGET
    + "'"
)
_LEADING_PAIRED_THREE_RETRY = (
    _LEADING_PAIRED_THREE_PREFIX + "'" + _LEADING_PAIRED_THREE_TARGET + "'"
)
_LEADING_PAIRED_THREE_CTC = {
    **{
        threshold: ((0, 123), (141, 354))
        for threshold in (
            0.0001,
            0.0003,
            0.0005,
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
        )
    },
    **{
        threshold: ((0, 354),)
        for threshold in (0.02, 0.03, 0.04, 0.05, 0.07)
    },
}


def leading_paired_three_crop() -> Image.Image:
    crop = Image.new("RGB", (605, 70))
    crop.paste((90, 90, 90), (24, 0, 378, 70))
    crop.paste((50, 50, 50), (397, 0, 581, 70))
    crop.paste((100, 100, 100), (24 + 132, 0, 24 + 154, 70))
    crop.paste((110, 110, 110), (24 + 154, 0, 24 + 340, 70))
    crop.paste((120, 120, 120), (24 + 340, 0, 24 + 354, 70))
    return crop


def leading_paired_three_words(
    *,
    candidate_text: str = _LEADING_PAIRED_THREE_RAW,
    candidate_confidence: float = 0.54925,
    candidate_box: BoundingBox | None = None,
    neighbor_text: str = _LEADING_PAIRED_THREE_NEIGHBOR,
) -> list[tuple[str, BoundingBox, float]]:
    candidate_box = candidate_box or BoundingBox(
        _LEADING_PAIRED_THREE_LINE.left + 24,
        _LEADING_PAIRED_THREE_LINE.top,
        _LEADING_PAIRED_THREE_LINE.left + 378,
        _LEADING_PAIRED_THREE_LINE.bottom,
    )
    return [
        (candidate_text, candidate_box, candidate_confidence),
        (
            neighbor_text,
            BoundingBox(
                _LEADING_PAIRED_THREE_LINE.left + 397,
                _LEADING_PAIRED_THREE_LINE.top,
                _LEADING_PAIRED_THREE_LINE.left + 581,
                _LEADING_PAIRED_THREE_LINE.bottom,
            ),
            0.99995,
        ),
    ]


class ConfirmedLeadingPairedThreeRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _LEADING_PAIRED_THREE_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _LEADING_PAIRED_THREE_RETRY,
            0.51575,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _LEADING_PAIRED_THREE_WRAPPER,
            0.9,
        )
        self.target_variant = target_variant or RecognizedText(
            _LEADING_PAIRED_THREE_TARGET,
            0.9999,
        )
        self.opening_variant = opening_variant or RecognizedText(
            chr(0x2018),
            0.7,
        )
        self.closing_variant = closing_variant or RecognizedText("'", 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (605, 70) and space_threshold is None:
            return self.default_segments
        if image.size != (354, 70) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _LEADING_PAIRED_THREE_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 140
        if image.size == (354, 70):
            return RecognizedText(_LEADING_PAIRED_THREE_RAW, 0.54925)
        if image.size == (708, 140):
            return self.candidate_enhanced
        original_width = image.width // 2 if enhanced else image.width
        if 211 <= original_width <= 216:
            return self.wrapper_variant
        if 180 <= original_width <= 190:
            return self.target_variant
        if 119 <= original_width <= 127:
            return RecognizedText(_LEADING_PAIRED_THREE_PREFIX, 0.99995)
        if 11 <= original_width <= 22:
            if intensity >= 115:
                return self.closing_variant
            return self.opening_variant
        if image.size == (184, 70):
            return RecognizedText(_LEADING_PAIRED_THREE_NEIGHBOR, 0.99995)
        return RecognizedText("", 0.0)


def test_confirmed_leading_paired_wrapped_three_recovers() -> None:
    words = leading_paired_three_words()
    recovered = _recover_confirmed_leading_paired_wrapped_three_split(
        words,
        leading_paired_three_words(),
        leading_paired_three_crop(),
        _LEADING_PAIRED_THREE_LINE,
        ConfirmedLeadingPairedThreeRecognizer(),
    )

    assert recovered == [
        (
            _LEADING_PAIRED_THREE_PREFIX,
            BoundingBox(
                _LEADING_PAIRED_THREE_LINE.left + 24,
                _LEADING_PAIRED_THREE_LINE.top,
                _LEADING_PAIRED_THREE_LINE.left + 147,
                _LEADING_PAIRED_THREE_LINE.bottom,
            ),
            0.51575,
        ),
        (
            _LEADING_PAIRED_THREE_WRAPPER,
            BoundingBox(
                _LEADING_PAIRED_THREE_LINE.left + 165,
                _LEADING_PAIRED_THREE_LINE.top,
                _LEADING_PAIRED_THREE_LINE.left + 378,
                _LEADING_PAIRED_THREE_LINE.bottom,
            ),
            0.51575,
        ),
        words[1],
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _LEADING_PAIRED_THREE_RETRY[:-2] + "?" + "'",
                0.9,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x2018) + _LEADING_PAIRED_THREE_TARGET + "'",
                0.9,
            )
        },
        {
            "target_variant": RecognizedText(
                _LEADING_PAIRED_THREE_TARGET[:2],
                1.0,
            )
        },
        {"opening_variant": RecognizedText("'", 0.99)},
        {"closing_variant": RecognizedText("'", 0.47)},
        {"ctc_override": (0.01, ((0, 354),))},
    ],
)
def test_confirmed_leading_paired_three_requires_crop_evidence(
    overrides,
) -> None:
    words = leading_paired_three_words()

    assert (
        _recover_confirmed_leading_paired_wrapped_three_split(
            words,
            leading_paired_three_words(),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            leading_paired_three_words() * 2,
            leading_paired_three_words(),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(
                candidate_text="A" + _LEADING_PAIRED_THREE_RAW[1:]
            ),
            leading_paired_three_words(
                candidate_text="A" + _LEADING_PAIRED_THREE_RAW[1:]
            ),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(candidate_confidence=0.5491),
            leading_paired_three_words(candidate_confidence=0.5491),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(),
            leading_paired_three_words(candidate_confidence=0.5491),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(neighbor_text="A12"),
            leading_paired_three_words(neighbor_text="A12"),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(
                candidate_box=BoundingBox(
                    122.32,
                    _LEADING_PAIRED_THREE_LINE.top,
                    475.32,
                    _LEADING_PAIRED_THREE_LINE.bottom,
                )
            ),
            leading_paired_three_words(
                candidate_box=BoundingBox(
                    122.32,
                    _LEADING_PAIRED_THREE_LINE.top,
                    475.32,
                    _LEADING_PAIRED_THREE_LINE.bottom,
                )
            ),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(),
            leading_paired_three_words(),
            Image.new("RGB", (604, 70)),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(),
            leading_paired_three_words(),
            leading_paired_three_crop(),
            BoundingBox(97.32, 165.71739130434784, 701.7, 234.3913043478261),
            ConfirmedLeadingPairedThreeRecognizer(),
        ),
        (
            leading_paired_three_words(),
            leading_paired_three_words(),
            leading_paired_three_crop(),
            _LEADING_PAIRED_THREE_LINE,
            ConfirmedLeadingPairedThreeRecognizer(
                default_segments=_LEADING_PAIRED_THREE_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_leading_paired_three_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_leading_paired_wrapped_three_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class LeadingPairedThreeDetector:
    def detect(self, _image):
        return (DetectedRegion(_LEADING_PAIRED_THREE_LINE, 0.9936),)


def test_engine_recovers_leading_paired_wrapped_three_segment() -> None:
    image = Image.new("RGB", (800, 300))
    image.paste(leading_paired_three_crop(), (97, 165))
    engine = PaddleOcrEngine(
        LeadingPairedThreeDetector(),
        ConfirmedLeadingPairedThreeRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 12
    assert [len(word.text) for word in line.eojeols] == [2, 3, 3]
    assert line.eojeols[1].text == _LEADING_PAIRED_THREE_TARGET
    assert line.eojeols[1].box.left == pytest.approx(304.92)
    assert line.eojeols[1].box.right == pytest.approx(432.72)


_TERMINAL_PAIRED_SINGLE_LINE = BoundingBox(
    85.28,
    234.0,
    733.72,
    263.9347826086957,
)
_TERMINAL_PAIRED_SINGLE_SEGMENTS = (
    (36, 175),
    (184, 238),
    (248, 350),
    (359, 439),
    (447, 611),
)
_TERMINAL_PAIRED_SINGLE_PREFIX = "".join(
    map(chr, (0xAC00, 0xB098, 0xB2E4, 0xB77C))
)
_TERMINAL_PAIRED_SINGLE_TARGET = chr(0xB9C8)
_TERMINAL_PAIRED_SINGLE_WRAPPER = (
    chr(0x201C) + _TERMINAL_PAIRED_SINGLE_TARGET + chr(0x201D)
)
_TERMINAL_PAIRED_SINGLE_RAW = (
    _TERMINAL_PAIRED_SINGLE_PREFIX
    + chr(0x201C)
    + _TERMINAL_PAIRED_SINGLE_TARGET
    + '"'
)
_TERMINAL_PAIRED_SINGLE_RETRY = (
    _TERMINAL_PAIRED_SINGLE_PREFIX + _TERMINAL_PAIRED_SINGLE_WRAPPER
)
_TERMINAL_PAIRED_SINGLE_TEXTS = (
    "".join(map(chr, (0xBC14, 0xC0AC, 0xC544, 0xC790, 0xCC28))) + ",",
    "".join(map(chr, (0xCE74, 0xD0C0))),
    "".join(map(chr, (0xD30C, 0xD558, 0xAC70, 0xB108))),
    "".join(map(chr, (0xB2E4, 0xB77C, 0xB9C8))),
    _TERMINAL_PAIRED_SINGLE_RAW,
)
_TERMINAL_PAIRED_SINGLE_CONFIDENCES = (
    0.99495,
    0.99975,
    0.99995,
    0.99995,
    0.67815,
)
_TERMINAL_PAIRED_SINGLE_CTC = {
    0.0001: ((0, 88), (87, 107), (115, 148), (147, 164)),
    0.0003: ((0, 107), (115, 148), (147, 164)),
    0.0005: ((0, 98), (97, 107), (115, 164)),
    **{
        threshold: ((0, 107), (115, 164))
        for threshold in (
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
        )
    },
    **{
        threshold: ((0, 164),)
        for threshold in (0.02, 0.03, 0.04, 0.05, 0.07)
    },
}


def terminal_paired_single_crop() -> Image.Image:
    crop = Image.new("RGB", (649, 30))
    for intensity, (left, right) in zip(
        (10, 20, 30, 40, 90),
        _TERMINAL_PAIRED_SINGLE_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 30))
    candidate_left = _TERMINAL_PAIRED_SINGLE_SEGMENTS[4][0]
    crop.paste(
        (100, 100, 100),
        (candidate_left + 103, 0, candidate_left + 126, 30),
    )
    crop.paste(
        (110, 110, 110),
        (candidate_left + 126, 0, candidate_left + 153, 30),
    )
    crop.paste(
        (120, 120, 120),
        (candidate_left + 153, 0, candidate_left + 164, 30),
    )
    return crop


def terminal_paired_single_words(
    *,
    candidate_text: str = _TERMINAL_PAIRED_SINGLE_RAW,
    candidate_confidence: float = 0.67815,
    candidate_box: BoundingBox | None = None,
    fourth_text: str = _TERMINAL_PAIRED_SINGLE_TEXTS[3],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _TERMINAL_PAIRED_SINGLE_SEGMENTS,
            _TERMINAL_PAIRED_SINGLE_TEXTS,
            _TERMINAL_PAIRED_SINGLE_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 4:
            text = candidate_text
            confidence = candidate_confidence
        elif index == 3:
            text = fourth_text
        box = BoundingBox(
            _TERMINAL_PAIRED_SINGLE_LINE.left + left,
            _TERMINAL_PAIRED_SINGLE_LINE.top,
            _TERMINAL_PAIRED_SINGLE_LINE.left + right,
            _TERMINAL_PAIRED_SINGLE_LINE.bottom,
        )
        if index == 4 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedTerminalPairedSingleRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _TERMINAL_PAIRED_SINGLE_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _TERMINAL_PAIRED_SINGLE_RETRY,
            0.56725,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _TERMINAL_PAIRED_SINGLE_WRAPPER,
            0.8,
        )
        self.target_variant = target_variant or RecognizedText(
            _TERMINAL_PAIRED_SINGLE_TARGET,
            0.9999,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.8)
        self.closing_variant = closing_variant or RecognizedText('"', 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (649, 30) and space_threshold is None:
            return self.default_segments
        if image.size != (164, 30) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _TERMINAL_PAIRED_SINGLE_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 60
        segment_reads = {
            10: RecognizedText(_TERMINAL_PAIRED_SINGLE_TEXTS[0], 0.99495),
            20: RecognizedText(_TERMINAL_PAIRED_SINGLE_TEXTS[1], 0.99975),
            30: RecognizedText(_TERMINAL_PAIRED_SINGLE_TEXTS[2], 0.99995),
            40: RecognizedText(_TERMINAL_PAIRED_SINGLE_TEXTS[3], 0.99995),
        }
        if image.height == 30 and intensity in segment_reads:
            return segment_reads[intensity]
        if image.size == (164, 30):
            return RecognizedText(_TERMINAL_PAIRED_SINGLE_RAW, 0.67815)
        if image.size == (328, 60):
            return self.candidate_enhanced
        original_width = image.width // 2 if enhanced else image.width
        if 55 <= original_width <= 60:
            return self.wrapper_variant
        if 100 <= original_width <= 106:
            return RecognizedText(_TERMINAL_PAIRED_SINGLE_PREFIX, 0.99995)
        if 9 <= original_width <= 31:
            if enhanced:
                if original_width <= 15:
                    return self.closing_variant
                if (
                    intensity == 100
                    or (original_width, intensity) in ((27, 127), (20, 0))
                ):
                    return self.opening_variant
                return self.target_variant
            if intensity >= 115:
                return self.closing_variant
            if intensity <= 105:
                return self.opening_variant
            return self.target_variant
        return RecognizedText("", 0.0)


def test_confirmed_terminal_paired_wrapped_single_recovers() -> None:
    words = terminal_paired_single_words()
    recovered = _recover_confirmed_terminal_paired_wrapped_single_split(
        words,
        terminal_paired_single_words(),
        terminal_paired_single_crop(),
        _TERMINAL_PAIRED_SINGLE_LINE,
        ConfirmedTerminalPairedSingleRecognizer(),
    )

    assert recovered == [
        *words[:4],
        (
            _TERMINAL_PAIRED_SINGLE_PREFIX,
            BoundingBox(
                _TERMINAL_PAIRED_SINGLE_LINE.left + 447,
                _TERMINAL_PAIRED_SINGLE_LINE.top,
                _TERMINAL_PAIRED_SINGLE_LINE.left + 549,
                _TERMINAL_PAIRED_SINGLE_LINE.bottom,
            ),
            0.56725,
        ),
        (
            _TERMINAL_PAIRED_SINGLE_WRAPPER,
            BoundingBox(
                _TERMINAL_PAIRED_SINGLE_LINE.left + 554,
                _TERMINAL_PAIRED_SINGLE_LINE.top,
                _TERMINAL_PAIRED_SINGLE_LINE.left + 611,
                _TERMINAL_PAIRED_SINGLE_LINE.bottom,
            ),
            0.56725,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _TERMINAL_PAIRED_SINGLE_RAW,
                0.9,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _TERMINAL_PAIRED_SINGLE_TARGET + '"',
                0.9,
            )
        },
        {"target_variant": RecognizedText(chr(0xC790), 1.0)},
        {"opening_variant": RecognizedText("'", 0.99)},
        {"closing_variant": RecognizedText('"', 0.48)},
        {"ctc_override": (0.01, ((0, 164),))},
    ],
)
def test_confirmed_terminal_paired_single_requires_crop_evidence(
    overrides,
) -> None:
    words = terminal_paired_single_words()

    assert (
        _recover_confirmed_terminal_paired_wrapped_single_split(
            words,
            terminal_paired_single_words(),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(**overrides),
        )
        == words
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            terminal_paired_single_words() * 2,
            terminal_paired_single_words(),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(
                candidate_text="A" + _TERMINAL_PAIRED_SINGLE_RAW[1:]
            ),
            terminal_paired_single_words(
                candidate_text="A" + _TERMINAL_PAIRED_SINGLE_RAW[1:]
            ),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(candidate_confidence=0.6780),
            terminal_paired_single_words(candidate_confidence=0.6780),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(),
            terminal_paired_single_words(candidate_confidence=0.6780),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(fourth_text="A12"),
            terminal_paired_single_words(fourth_text="A12"),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(
                candidate_box=BoundingBox(
                    _TERMINAL_PAIRED_SINGLE_LINE.left + 448,
                    _TERMINAL_PAIRED_SINGLE_LINE.top,
                    _TERMINAL_PAIRED_SINGLE_LINE.left + 611,
                    _TERMINAL_PAIRED_SINGLE_LINE.bottom,
                )
            ),
            terminal_paired_single_words(
                candidate_box=BoundingBox(
                    _TERMINAL_PAIRED_SINGLE_LINE.left + 448,
                    _TERMINAL_PAIRED_SINGLE_LINE.top,
                    _TERMINAL_PAIRED_SINGLE_LINE.left + 611,
                    _TERMINAL_PAIRED_SINGLE_LINE.bottom,
                )
            ),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(),
            terminal_paired_single_words(),
            Image.new("RGB", (648, 30)),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(),
            terminal_paired_single_words(),
            terminal_paired_single_crop(),
            BoundingBox(85.28, 234.0, 733.74, 263.9347826086957),
            ConfirmedTerminalPairedSingleRecognizer(),
        ),
        (
            terminal_paired_single_words(),
            terminal_paired_single_words(),
            terminal_paired_single_crop(),
            _TERMINAL_PAIRED_SINGLE_LINE,
            ConfirmedTerminalPairedSingleRecognizer(
                default_segments=_TERMINAL_PAIRED_SINGLE_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_terminal_paired_single_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_terminal_paired_wrapped_single_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class TerminalPairedSingleDetector:
    def detect(self, _image):
        return (DetectedRegion(_TERMINAL_PAIRED_SINGLE_LINE, 0.9904),)


def test_engine_recovers_terminal_paired_wrapped_single_segment() -> None:
    image = Image.new("RGB", (800, 350))
    image.paste(terminal_paired_single_crop(), (85, 234))
    engine = PaddleOcrEngine(
        TerminalPairedSingleDetector(),
        ConfirmedTerminalPairedSingleRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 27
    assert [len(word.text) for word in line.eojeols] == [5, 2, 4, 3, 4, 1]
    assert line.eojeols[5].text == _TERMINAL_PAIRED_SINGLE_TARGET
    assert line.eojeols[5].box.left == pytest.approx(658.28)
    assert line.eojeols[5].box.right == pytest.approx(677.28)


_MISMATCHED_CURLY_THREE_LINE = BoundingBox(
    67.8,
    235.76,
    1071.2,
    279.7817391304348,
)
_MISMATCHED_CURLY_THREE_SEGMENTS = (
    (57, 193),
    (206, 403),
    (420, 748),
    (767, 945),
)
_MISMATCHED_CURLY_THREE_TARGET = "".join(map(chr, (0xC544, 0xC790, 0xCC28)))
_MISMATCHED_CURLY_THREE_FOLLOWING = "".join(
    map(chr, (0xCE74, 0xD0C0, 0xD30C))
)
_MISMATCHED_CURLY_THREE_WRAPPER = (
    chr(0x201C) + _MISMATCHED_CURLY_THREE_TARGET + chr(0x201D)
)
_MISMATCHED_CURLY_THREE_RAW = (
    chr(0x201C)
    + _MISMATCHED_CURLY_THREE_TARGET
    + '"'
    + _MISMATCHED_CURLY_THREE_FOLLOWING
)
_MISMATCHED_CURLY_THREE_TEXTS = (
    "".join(map(chr, (0xAC00, 0xB098, 0xB2E4))),
    "".join(map(chr, (0xB77C, 0xB9C8, 0xBC14, 0xC0AC))) + ",",
    _MISMATCHED_CURLY_THREE_RAW,
    "".join(map(chr, (0xD558, 0xAC70, 0xB108, 0xB354))),
)
_MISMATCHED_CURLY_THREE_RAW_CONFIDENCES = (
    0.99995,
    0.9927,
    0.57315,
    0.98895,
)
_MISMATCHED_CURLY_THREE_CTC = {
    0.0001: ((0, 154), (153, 181), (196, 252), (251, 328)),
    **{
        threshold: ((0, 154), (153, 181), (196, 328))
        for threshold in (0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005)
    },
    **{
        threshold: ((0, 181), (196, 328))
        for threshold in (0.007, 0.01, 0.015, 0.02)
    },
    **{
        threshold: ((0, 328),)
        for threshold in (0.03, 0.04, 0.05, 0.07)
    },
}


def mismatched_curly_three_crop() -> Image.Image:
    crop = Image.new("RGB", (1005, 45))
    for intensity, (left, right) in zip(
        (10, 20, 90, 40),
        _MISMATCHED_CURLY_THREE_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 45))
    candidate_left = _MISMATCHED_CURLY_THREE_SEGMENTS[2][0]
    crop.paste(
        (100, 100, 100),
        (candidate_left, 0, candidate_left + 19, 45),
    )
    crop.paste(
        (110, 110, 110),
        (candidate_left + 19, 0, candidate_left + 153, 45),
    )
    crop.paste(
        (120, 120, 120),
        (candidate_left + 153, 0, candidate_left + 181, 45),
    )
    crop.paste(
        (130, 130, 130),
        (candidate_left + 196, 0, candidate_left + 328, 45),
    )
    return crop


def mismatched_curly_three_words(
    *,
    selected: bool,
    candidate_text: str = _MISMATCHED_CURLY_THREE_RAW,
    candidate_confidence: float | None = None,
    candidate_box: BoundingBox | None = None,
    second_text: str = _MISMATCHED_CURLY_THREE_TEXTS[1],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _MISMATCHED_CURLY_THREE_SEGMENTS,
            _MISMATCHED_CURLY_THREE_TEXTS,
            _MISMATCHED_CURLY_THREE_RAW_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 2:
            text = candidate_text
            confidence = (
                0.64565
                if selected and candidate_confidence is None
                else confidence
                if candidate_confidence is None
                else candidate_confidence
            )
        elif index == 1:
            text = second_text
        box = BoundingBox(
            _MISMATCHED_CURLY_THREE_LINE.left + left,
            _MISMATCHED_CURLY_THREE_LINE.top,
            _MISMATCHED_CURLY_THREE_LINE.left + right,
            _MISMATCHED_CURLY_THREE_LINE.bottom,
        )
        if index == 2 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedMismatchedCurlyThreeRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _MISMATCHED_CURLY_THREE_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        following_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _MISMATCHED_CURLY_THREE_RAW,
            0.64565,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _MISMATCHED_CURLY_THREE_WRAPPER,
            0.71,
        )
        self.target_variant = target_variant or RecognizedText(
            _MISMATCHED_CURLY_THREE_TARGET,
            0.9999,
        )
        self.following_variant = following_variant or RecognizedText(
            _MISMATCHED_CURLY_THREE_FOLLOWING,
            0.9998,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.95)
        self.closing_variant = closing_variant or RecognizedText('"', 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (1005, 45) and space_threshold is None:
            return self.default_segments
        if image.size != (328, 45) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _MISMATCHED_CURLY_THREE_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        enhanced = image.height == 90
        original_width = image.width // 2 if enhanced else image.width
        segment_reads = {
            (136, 10): RecognizedText(
                _MISMATCHED_CURLY_THREE_TEXTS[0],
                0.99995,
            ),
            (197, 20): RecognizedText(
                _MISMATCHED_CURLY_THREE_TEXTS[1],
                0.9927,
            ),
            (178, 40): RecognizedText(
                _MISMATCHED_CURLY_THREE_TEXTS[3],
                0.98895,
            ),
        }
        if not enhanced and (original_width, intensity) in segment_reads:
            return segment_reads[(original_width, intensity)]
        if image.size == (328, 45):
            return RecognizedText(_MISMATCHED_CURLY_THREE_RAW, 0.57315)
        if image.size == (656, 90):
            return self.candidate_enhanced
        if 179 <= original_width <= 187:
            confidence = 0.56675 if enhanced else self.wrapper_variant.confidence
            return RecognizedText(self.wrapper_variant.text, confidence)
        if original_width == 154:
            return RecognizedText(
                '"' + _MISMATCHED_CURLY_THREE_TARGET,
                0.9,
            )
        if 130 <= original_width <= 142 and intensity >= 125:
            return self.following_variant
        if enhanced and 127 <= original_width <= 143 and intensity == 0:
            return self.target_variant
        if 127 <= original_width <= 143 and 105 <= intensity <= 115:
            return self.target_variant
        if 18 <= original_width <= 26 and intensity <= 105:
            return self.opening_variant
        if 26 <= original_width <= 34 and intensity >= 115:
            confidence = 0.85 if original_width == 28 else self.closing_variant.confidence
            return RecognizedText(self.closing_variant.text, confidence)
        if original_width == 132:
            return self.following_variant
        return RecognizedText("", 0.0)


def test_confirmed_mismatched_curly_three_plus_three_recovers() -> None:
    selected = mismatched_curly_three_words(selected=True)
    recovered = _recover_confirmed_mismatched_curly_three_plus_three_split(
        selected,
        mismatched_curly_three_words(selected=False),
        mismatched_curly_three_crop(),
        _MISMATCHED_CURLY_THREE_LINE,
        ConfirmedMismatchedCurlyThreeRecognizer(),
    )

    assert recovered == [
        *selected[:2],
        (
            _MISMATCHED_CURLY_THREE_WRAPPER,
            BoundingBox(
                _MISMATCHED_CURLY_THREE_LINE.left + 420,
                _MISMATCHED_CURLY_THREE_LINE.top,
                _MISMATCHED_CURLY_THREE_LINE.left + 601,
                _MISMATCHED_CURLY_THREE_LINE.bottom,
            ),
            0.56675,
        ),
        (
            _MISMATCHED_CURLY_THREE_FOLLOWING,
            BoundingBox(
                _MISMATCHED_CURLY_THREE_LINE.left + 616,
                _MISMATCHED_CURLY_THREE_LINE.top,
                _MISMATCHED_CURLY_THREE_LINE.left + 748,
                _MISMATCHED_CURLY_THREE_LINE.bottom,
            ),
            0.57315,
        ),
        selected[3],
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _MISMATCHED_CURLY_THREE_RAW,
                0.6455,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _MISMATCHED_CURLY_THREE_TARGET + '"',
                0.9,
            )
        },
        {"target_variant": RecognizedText(chr(0xC790) * 3, 1.0)},
        {"following_variant": RecognizedText(chr(0xD558) * 3, 1.0)},
        {"opening_variant": RecognizedText("'", 0.99)},
        {"closing_variant": RecognizedText('"', 0.65)},
        {"ctc_override": (0.01, ((0, 328),))},
    ],
)
def test_confirmed_mismatched_curly_three_requires_crop_evidence(
    overrides,
) -> None:
    selected = mismatched_curly_three_words(selected=True)

    assert (
        _recover_confirmed_mismatched_curly_three_plus_three_split(
            selected,
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(**overrides),
        )
        == selected
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            mismatched_curly_three_words(selected=True) * 2,
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(
                selected=True,
                candidate_text="A" + _MISMATCHED_CURLY_THREE_RAW[1:],
            ),
            mismatched_curly_three_words(
                selected=False,
                candidate_text="A" + _MISMATCHED_CURLY_THREE_RAW[1:],
            ),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(
                selected=True,
                candidate_confidence=0.6455,
            ),
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(selected=True),
            mismatched_curly_three_words(
                selected=False,
                candidate_confidence=0.5730,
            ),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(
                selected=True,
                candidate_text=_MISMATCHED_CURLY_THREE_RAW[:-1] + chr(0xD558),
            ),
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(
                selected=True,
                second_text="A123,",
            ),
            mismatched_curly_three_words(
                selected=False,
                second_text="A123,",
            ),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(
                selected=True,
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_THREE_LINE.left + 421,
                    _MISMATCHED_CURLY_THREE_LINE.top,
                    _MISMATCHED_CURLY_THREE_LINE.left + 748,
                    _MISMATCHED_CURLY_THREE_LINE.bottom,
                ),
            ),
            mismatched_curly_three_words(
                selected=False,
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_THREE_LINE.left + 421,
                    _MISMATCHED_CURLY_THREE_LINE.top,
                    _MISMATCHED_CURLY_THREE_LINE.left + 748,
                    _MISMATCHED_CURLY_THREE_LINE.bottom,
                ),
            ),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(selected=True),
            mismatched_curly_three_words(selected=False),
            Image.new("RGB", (1004, 45)),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(selected=True),
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            BoundingBox(67.8, 235.76, 1071.22, 279.7817391304348),
            ConfirmedMismatchedCurlyThreeRecognizer(),
        ),
        (
            mismatched_curly_three_words(selected=True),
            mismatched_curly_three_words(selected=False),
            mismatched_curly_three_crop(),
            _MISMATCHED_CURLY_THREE_LINE,
            ConfirmedMismatchedCurlyThreeRecognizer(
                default_segments=_MISMATCHED_CURLY_THREE_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_mismatched_curly_three_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_mismatched_curly_three_plus_three_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class MismatchedCurlyThreeDetector:
    def detect(self, _image):
        return (DetectedRegion(_MISMATCHED_CURLY_THREE_LINE, 0.994),)


def test_engine_recovers_mismatched_curly_three_plus_three_segment() -> None:
    image = Image.new("RGB", (1280, 720))
    image.paste(mismatched_curly_three_crop(), (67, 235))
    engine = PaddleOcrEngine(
        MismatchedCurlyThreeDetector(),
        ConfirmedMismatchedCurlyThreeRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 24
    assert [len(word.text) for word in line.eojeols] == [3, 4, 3, 3, 4]
    assert line.eojeols[2].text == _MISMATCHED_CURLY_THREE_TARGET
    assert line.eojeols[2].box.left == pytest.approx(524.0)
    assert line.eojeols[2].box.right == pytest.approx(632.6)
_MISMATCHED_CURLY_TWO_LINE = BoundingBox(
    355.4,
    298.17391304347825,
    668.6,
    347.4782608695652,
)
_MISMATCHED_CURLY_TWO_SEGMENTS = ((6, 187), (204, 303))
_MISMATCHED_CURLY_TWO_TARGET = "".join(map(chr, (0xD53C, 0xD560)))
_MISMATCHED_CURLY_TWO_FOLLOWING = chr(0xC218)
_MISMATCHED_CURLY_TWO_WRAPPER = (
    chr(0x201C) + _MISMATCHED_CURLY_TWO_TARGET + chr(0x201D)
)
_MISMATCHED_CURLY_TWO_RAW = (
    '"'
    + _MISMATCHED_CURLY_TWO_TARGET
    + chr(0x201D)
    + _MISMATCHED_CURLY_TWO_FOLLOWING
)
_MISMATCHED_CURLY_TWO_SECOND = "".join(map(chr, (0xC5C6, 0xB2E4))) + "."
_MISMATCHED_CURLY_TWO_CTC = {
    0.0001: ((0, 55), (54, 96), (95, 138), (137, 181)),
    **{
        threshold: ((0, 96), (95, 138), (137, 181))
        for threshold in (0.0003, 0.0005)
    },
    **{
        threshold: ((0, 138), (137, 181))
        for threshold in (
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
            0.03,
        )
    },
    **{
        threshold: ((0, 181),)
        for threshold in (0.04, 0.05, 0.07)
    },
}


def mismatched_curly_two_crop() -> Image.Image:
    crop = Image.new("RGB", (314, 50))
    crop.paste((90, 90, 90), (6, 0, 187, 50))
    crop.paste((40, 40, 40), (204, 0, 303, 50))
    crop.paste((100, 100, 100), (6, 0, 30, 50))
    crop.paste((110, 110, 110), (18, 0, 108, 50))
    crop.paste((120, 120, 120), (99, 0, 150, 50))
    crop.paste((130, 130, 130), (139, 0, 187, 50))
    return crop


def mismatched_curly_two_words(
    *,
    selected: bool,
    candidate_text: str = _MISMATCHED_CURLY_TWO_RAW,
    candidate_confidence: float | None = None,
    candidate_box: BoundingBox | None = None,
    second_text: str = _MISMATCHED_CURLY_TWO_SECOND,
) -> list[tuple[str, BoundingBox, float]]:
    first_confidence = (
        0.51235
        if selected and candidate_confidence is None
        else 0.509356
        if candidate_confidence is None
        else candidate_confidence
    )
    values = [
        (
            candidate_text,
            BoundingBox(
                _MISMATCHED_CURLY_TWO_LINE.left + 6,
                _MISMATCHED_CURLY_TWO_LINE.top,
                _MISMATCHED_CURLY_TWO_LINE.left + 187,
                _MISMATCHED_CURLY_TWO_LINE.bottom,
            ),
            first_confidence,
        ),
        (
            second_text,
            BoundingBox(
                _MISMATCHED_CURLY_TWO_LINE.left + 204,
                _MISMATCHED_CURLY_TWO_LINE.top,
                _MISMATCHED_CURLY_TWO_LINE.left + 303,
                _MISMATCHED_CURLY_TWO_LINE.bottom,
            ),
            0.994093,
        ),
    ]
    if candidate_box is not None:
        values[0] = (candidate_text, candidate_box, first_confidence)
    return values


class ConfirmedMismatchedCurlyTwoRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _MISMATCHED_CURLY_TWO_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        following_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _MISMATCHED_CURLY_TWO_RAW,
            0.51235,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _MISMATCHED_CURLY_TWO_WRAPPER,
            0.76,
        )
        self.target_variant = target_variant or RecognizedText(
            _MISMATCHED_CURLY_TWO_TARGET,
            1.0,
        )
        self.following_variant = following_variant or RecognizedText(
            _MISMATCHED_CURLY_TWO_FOLLOWING,
            1.0,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.99)
        self.closing_variant = closing_variant or RecognizedText('"', 0.56)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (314, 50) and space_threshold is None:
            return self.default_segments
        if image.size != (181, 50) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _MISMATCHED_CURLY_TWO_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        enhanced = image.height == 100
        original_width = image.width // 2 if enhanced else image.width
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        if image.size == (181, 50):
            return RecognizedText(_MISMATCHED_CURLY_TWO_RAW, 0.509356)
        if image.size == (362, 100):
            return self.candidate_enhanced
        if image.size == (99, 50):
            return RecognizedText(_MISMATCHED_CURLY_TWO_SECOND, 0.994093)
        if 136 <= original_width <= 144:
            confidence = 0.78 if enhanced else self.wrapper_variant.confidence
            return RecognizedText(self.wrapper_variant.text, confidence)
        if 80 <= original_width <= 90:
            return self.target_variant
        if 16 <= original_width <= 24:
            return self.opening_variant
        if 40 <= original_width <= 49:
            if intensity >= 129:
                return self.following_variant
            confidence = 0.57 if enhanced else self.closing_variant.confidence
            return RecognizedText(self.closing_variant.text, confidence)
        return RecognizedText("", 0.0)


def test_confirmed_mismatched_curly_two_plus_one_recovers() -> None:
    selected = mismatched_curly_two_words(selected=True)
    recovered = _recover_confirmed_mismatched_curly_two_plus_one_split(
        selected,
        mismatched_curly_two_words(selected=False),
        mismatched_curly_two_crop(),
        _MISMATCHED_CURLY_TWO_LINE,
        ConfirmedMismatchedCurlyTwoRecognizer(),
    )

    assert recovered == [
        (
            _MISMATCHED_CURLY_TWO_WRAPPER,
            BoundingBox(
                _MISMATCHED_CURLY_TWO_LINE.left + 6,
                _MISMATCHED_CURLY_TWO_LINE.top,
                _MISMATCHED_CURLY_TWO_LINE.left + 144,
                _MISMATCHED_CURLY_TWO_LINE.bottom,
            ),
            0.509356,
        ),
        (
            _MISMATCHED_CURLY_TWO_FOLLOWING,
            BoundingBox(
                _MISMATCHED_CURLY_TWO_LINE.left + 143,
                _MISMATCHED_CURLY_TWO_LINE.top,
                _MISMATCHED_CURLY_TWO_LINE.left + 187,
                _MISMATCHED_CURLY_TWO_LINE.bottom,
            ),
            0.509356,
        ),
        selected[1],
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _MISMATCHED_CURLY_TWO_RAW,
                0.5122,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _MISMATCHED_CURLY_TWO_TARGET + '"',
                0.9,
            )
        },
        {"target_variant": RecognizedText(chr(0xC790) * 2, 1.0)},
        {"following_variant": RecognizedText(chr(0xD558), 1.0)},
        {"opening_variant": RecognizedText("'", 0.99)},
        {"closing_variant": RecognizedText(chr(0x201D), 0.9)},
        {"ctc_override": (0.01, ((0, 181),))},
    ],
)
def test_confirmed_mismatched_curly_two_requires_crop_evidence(
    overrides,
) -> None:
    selected = mismatched_curly_two_words(selected=True)

    assert (
        _recover_confirmed_mismatched_curly_two_plus_one_split(
            selected,
            mismatched_curly_two_words(selected=False),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(**overrides),
        )
        == selected
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            mismatched_curly_two_words(selected=True) * 2,
            mismatched_curly_two_words(selected=False),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(
                selected=True,
                candidate_text="'" + _MISMATCHED_CURLY_TWO_RAW[1:],
            ),
            mismatched_curly_two_words(
                selected=False,
                candidate_text="'" + _MISMATCHED_CURLY_TWO_RAW[1:],
            ),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(
                selected=True,
                candidate_confidence=0.5122,
            ),
            mismatched_curly_two_words(selected=False),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(selected=True),
            mismatched_curly_two_words(
                selected=False,
                candidate_confidence=0.5092,
            ),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(
                selected=True,
                second_text="A1.",
            ),
            mismatched_curly_two_words(
                selected=False,
                second_text="A1.",
            ),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(
                selected=True,
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_TWO_LINE.left + 7,
                    _MISMATCHED_CURLY_TWO_LINE.top,
                    _MISMATCHED_CURLY_TWO_LINE.left + 187,
                    _MISMATCHED_CURLY_TWO_LINE.bottom,
                ),
            ),
            mismatched_curly_two_words(
                selected=False,
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_TWO_LINE.left + 7,
                    _MISMATCHED_CURLY_TWO_LINE.top,
                    _MISMATCHED_CURLY_TWO_LINE.left + 187,
                    _MISMATCHED_CURLY_TWO_LINE.bottom,
                ),
            ),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(selected=True),
            mismatched_curly_two_words(selected=False),
            Image.new("RGB", (313, 50)),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(selected=True),
            mismatched_curly_two_words(selected=False),
            mismatched_curly_two_crop(),
            BoundingBox(
                355.4,
                298.17391304347825,
                668.62,
                347.4782608695652,
            ),
            ConfirmedMismatchedCurlyTwoRecognizer(),
        ),
        (
            mismatched_curly_two_words(selected=True),
            mismatched_curly_two_words(selected=False),
            mismatched_curly_two_crop(),
            _MISMATCHED_CURLY_TWO_LINE,
            ConfirmedMismatchedCurlyTwoRecognizer(
                default_segments=(_MISMATCHED_CURLY_TWO_SEGMENTS[0],)
            ),
        ),
    ],
)
def test_confirmed_mismatched_curly_two_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_mismatched_curly_two_plus_one_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class MismatchedCurlyTwoDetector:
    def detect(self, _image):
        return (DetectedRegion(_MISMATCHED_CURLY_TWO_LINE, 0.9891),)


def test_engine_recovers_mismatched_curly_two_plus_one_segment() -> None:
    image = Image.new("RGB", (1280, 720))
    image.paste(mismatched_curly_two_crop(), (355, 298))
    engine = PaddleOcrEngine(
        MismatchedCurlyTwoDetector(),
        ConfirmedMismatchedCurlyTwoRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 10
    assert [len(word.text) for word in line.eojeols] == [2, 1, 2]
    assert line.eojeols[0].text == _MISMATCHED_CURLY_TWO_TARGET
    assert line.eojeols[0].box.left == pytest.approx(395.9)
    assert line.eojeols[0].box.right == pytest.approx(464.9)
_MISMATCHED_CURLY_FOUR_LINE = BoundingBox(
    81.64,
    199.56521739130434,
    797.36,
    225.97826086956522,
)
_MISMATCHED_CURLY_FOUR_SEGMENTS = (
    (42, 145),
    (156, 236),
    (243, 298),
    (305, 551),
    (561, 613),
    (623, 673),
)
_MISMATCHED_CURLY_FOUR_TARGET = "".join(
    map(chr, (0xCC45, 0xD45C, 0xC9C0, 0xB97C))
)
_MISMATCHED_CURLY_FOUR_FOLLOWING = "".join(
    map(chr, (0xC0AC, 0xC6A9, 0xD558, 0xBA70))
)
_MISMATCHED_CURLY_FOUR_WRAPPER = (
    chr(0x201C) + _MISMATCHED_CURLY_FOUR_TARGET + chr(0x201D)
)
_MISMATCHED_CURLY_FOUR_RAW = (
    chr(0x201C)
    + _MISMATCHED_CURLY_FOUR_TARGET
    + '"'
    + _MISMATCHED_CURLY_FOUR_FOLLOWING
)
_MISMATCHED_CURLY_FOUR_TEXTS = (
    "".join(map(chr, (0xC5FC, 0xAC00, 0xD310, 0xC758))),
    "".join(map(chr, (0xBE68, 0xAC1B, 0xACE0))),
    "".join(map(chr, (0xD558, 0xC580))),
    _MISMATCHED_CURLY_FOUR_RAW,
    "".join(map(chr, (0xD3AD, 0xADC4))),
    "".join(map(chr, (0xCC45, 0xC758))),
)
_MISMATCHED_CURLY_FOUR_CONFIDENCES = (
    0.995056,
    0.908196,
    0.999961,
    0.623883,
    0.999944,
    0.999903,
)
_MISMATCHED_CURLY_FOUR_CTC = {
    **{
        threshold: ((0, 115), (114, 142), (141, 246))
        for threshold in (0.0001, 0.0003, 0.0005)
    },
    **{
        threshold: ((0, 142), (141, 246))
        for threshold in (
            0.001,
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
        )
    },
    **{
        threshold: ((0, 246),)
        for threshold in (0.03, 0.04, 0.05, 0.07)
    },
}


def mismatched_curly_four_crop() -> Image.Image:
    crop = Image.new("RGB", (717, 27))
    for intensity, (left, right) in zip(
        (10, 20, 30, 90, 40, 50),
        _MISMATCHED_CURLY_FOUR_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 27))
    candidate_left = _MISMATCHED_CURLY_FOUR_SEGMENTS[3][0]
    crop.paste(
        (100, 100, 100),
        (candidate_left, 0, candidate_left + 22, 27),
    )
    crop.paste(
        (110, 110, 110),
        (candidate_left + 11, 0, candidate_left + 119, 27),
    )
    crop.paste(
        (120, 120, 120),
        (candidate_left + 112, 0, candidate_left + 148, 27),
    )
    crop.paste(
        (130, 130, 130),
        (candidate_left + 135, 0, candidate_left + 246, 27),
    )
    return crop


def mismatched_curly_four_words(
    *,
    candidate_text: str = _MISMATCHED_CURLY_FOUR_RAW,
    candidate_confidence: float = 0.623883,
    candidate_box: BoundingBox | None = None,
    first_text: str = _MISMATCHED_CURLY_FOUR_TEXTS[0],
) -> list[tuple[str, BoundingBox, float]]:
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _MISMATCHED_CURLY_FOUR_SEGMENTS,
            _MISMATCHED_CURLY_FOUR_TEXTS,
            _MISMATCHED_CURLY_FOUR_CONFIDENCES,
            strict=True,
        )
    ):
        if index == 0:
            text = first_text
        if index == 3:
            text = candidate_text
            confidence = candidate_confidence
        box = BoundingBox(
            _MISMATCHED_CURLY_FOUR_LINE.left + left,
            _MISMATCHED_CURLY_FOUR_LINE.top,
            _MISMATCHED_CURLY_FOUR_LINE.left + right,
            _MISMATCHED_CURLY_FOUR_LINE.bottom,
        )
        if index == 3 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedMismatchedCurlyFourRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _MISMATCHED_CURLY_FOUR_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        following_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            chr(0x201C)
            + _MISMATCHED_CURLY_FOUR_TARGET
            + chr(0x201D)
            + _MISMATCHED_CURLY_FOUR_FOLLOWING,
            0.610357,
        )
        self.wrapper_variant = wrapper_variant or RecognizedText(
            _MISMATCHED_CURLY_FOUR_WRAPPER,
            0.72,
        )
        self.target_variant = target_variant or RecognizedText(
            _MISMATCHED_CURLY_FOUR_TARGET,
            0.999,
        )
        self.following_variant = following_variant or RecognizedText(
            _MISMATCHED_CURLY_FOUR_FOLLOWING,
            0.9998,
        )
        self.opening_variant = opening_variant or RecognizedText('"', 0.9)
        self.closing_variant = closing_variant or RecognizedText('"', 0.8)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (717, 27) and space_threshold is None:
            return self.default_segments
        if image.size != (246, 27) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _MISMATCHED_CURLY_FOUR_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        enhanced = image.height == 54
        original_width = image.width // 2 if enhanced else image.width
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        segment_reads = {
            (103, 10): RecognizedText(
                _MISMATCHED_CURLY_FOUR_TEXTS[0],
                0.995056,
            ),
            (80, 20): RecognizedText(
                _MISMATCHED_CURLY_FOUR_TEXTS[1],
                0.908196,
            ),
            (55, 30): RecognizedText(
                _MISMATCHED_CURLY_FOUR_TEXTS[2],
                0.999961,
            ),
            (52, 40): RecognizedText(
                _MISMATCHED_CURLY_FOUR_TEXTS[4],
                0.999944,
            ),
            (50, 50): RecognizedText(
                _MISMATCHED_CURLY_FOUR_TEXTS[5],
                0.999903,
            ),
        }
        if not enhanced and (original_width, intensity) in segment_reads:
            return segment_reads[(original_width, intensity)]
        if image.size == (246, 27):
            return RecognizedText(_MISMATCHED_CURLY_FOUR_RAW, 0.623883)
        if image.size == (492, 54):
            return self.candidate_enhanced
        if original_width == 115:
            confidence = 0.88 if enhanced else 0.84
            return RecognizedText(
                '"' + _MISMATCHED_CURLY_FOUR_TARGET,
                confidence,
            )
        if 140 <= original_width <= 148:
            confidence = (
                0.62 if enhanced and original_width == 142 else 0.46
            ) if enhanced else self.wrapper_variant.confidence
            return RecognizedText(self.wrapper_variant.text, confidence)
        if 14 <= original_width <= 22:
            return self.opening_variant
        if 26 <= original_width <= 34:
            confidence = 0.6 if enhanced else self.closing_variant.confidence
            return RecognizedText(self.closing_variant.text, confidence)
        if 96 <= original_width <= 109:
            if intensity >= 129:
                return self.following_variant
            return self.target_variant
        return RecognizedText("", 0.0)


def test_confirmed_mismatched_curly_four_plus_four_recovers() -> None:
    selected = mismatched_curly_four_words()
    recovered = _recover_confirmed_mismatched_curly_four_plus_four_split(
        selected,
        mismatched_curly_four_words(),
        mismatched_curly_four_crop(),
        _MISMATCHED_CURLY_FOUR_LINE,
        ConfirmedMismatchedCurlyFourRecognizer(),
    )

    assert recovered == [
        *selected[:3],
        (
            _MISMATCHED_CURLY_FOUR_WRAPPER,
            BoundingBox(
                _MISMATCHED_CURLY_FOUR_LINE.left + 305,
                _MISMATCHED_CURLY_FOUR_LINE.top,
                _MISMATCHED_CURLY_FOUR_LINE.left + 447,
                _MISMATCHED_CURLY_FOUR_LINE.bottom,
            ),
            0.46,
        ),
        (
            _MISMATCHED_CURLY_FOUR_FOLLOWING,
            BoundingBox(
                _MISMATCHED_CURLY_FOUR_LINE.left + 446,
                _MISMATCHED_CURLY_FOUR_LINE.top,
                _MISMATCHED_CURLY_FOUR_LINE.left + 551,
                _MISMATCHED_CURLY_FOUR_LINE.bottom,
            ),
            0.610357,
        ),
        *selected[4:],
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _MISMATCHED_CURLY_FOUR_RAW,
                0.7,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                chr(0x201C) + _MISMATCHED_CURLY_FOUR_TARGET + '"',
                0.9,
            )
        },
        {"target_variant": RecognizedText(chr(0xC790) * 4, 1.0)},
        {"following_variant": RecognizedText(chr(0xD558) * 4, 1.0)},
        {"opening_variant": RecognizedText("'", 0.99)},
        {"closing_variant": RecognizedText(chr(0x201D), 0.9)},
        {"ctc_override": (0.01, ((0, 246),))},
    ],
)
def test_confirmed_mismatched_curly_four_requires_crop_evidence(
    overrides,
) -> None:
    selected = mismatched_curly_four_words()

    assert (
        _recover_confirmed_mismatched_curly_four_plus_four_split(
            selected,
            mismatched_curly_four_words(),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(**overrides),
        )
        == selected
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            mismatched_curly_four_words() * 2,
            mismatched_curly_four_words(),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(
                candidate_text="'" + _MISMATCHED_CURLY_FOUR_RAW[1:],
            ),
            mismatched_curly_four_words(
                candidate_text="'" + _MISMATCHED_CURLY_FOUR_RAW[1:],
            ),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(candidate_confidence=0.6237),
            mismatched_curly_four_words(candidate_confidence=0.6237),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(candidate_confidence=0.6239),
            mismatched_curly_four_words(),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(first_text="A123"),
            mismatched_curly_four_words(first_text="A123"),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_FOUR_LINE.left + 306,
                    _MISMATCHED_CURLY_FOUR_LINE.top,
                    _MISMATCHED_CURLY_FOUR_LINE.left + 551,
                    _MISMATCHED_CURLY_FOUR_LINE.bottom,
                ),
            ),
            mismatched_curly_four_words(
                candidate_box=BoundingBox(
                    _MISMATCHED_CURLY_FOUR_LINE.left + 306,
                    _MISMATCHED_CURLY_FOUR_LINE.top,
                    _MISMATCHED_CURLY_FOUR_LINE.left + 551,
                    _MISMATCHED_CURLY_FOUR_LINE.bottom,
                ),
            ),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(),
            mismatched_curly_four_words(),
            Image.new("RGB", (716, 27)),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(),
            mismatched_curly_four_words(),
            mismatched_curly_four_crop(),
            BoundingBox(
                81.64,
                199.56521739130434,
                797.38,
                225.97826086956522,
            ),
            ConfirmedMismatchedCurlyFourRecognizer(),
        ),
        (
            mismatched_curly_four_words(),
            mismatched_curly_four_words(),
            mismatched_curly_four_crop(),
            _MISMATCHED_CURLY_FOUR_LINE,
            ConfirmedMismatchedCurlyFourRecognizer(
                default_segments=_MISMATCHED_CURLY_FOUR_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_mismatched_curly_four_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_mismatched_curly_four_plus_four_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class MismatchedCurlyFourDetector:
    def detect(self, _image):
        return (DetectedRegion(_MISMATCHED_CURLY_FOUR_LINE, 0.9941),)


def test_engine_recovers_mismatched_curly_four_plus_four_segment() -> None:
    image = Image.new("RGB", (1280, 720))
    image.paste(mismatched_curly_four_crop(), (81, 199))
    engine = PaddleOcrEngine(
        MismatchedCurlyFourDetector(),
        ConfirmedMismatchedCurlyFourRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 29
    assert [len(word.text) for word in line.eojeols] == [4, 3, 2, 4, 4, 2, 2]
    assert line.eojeols[3].text == _MISMATCHED_CURLY_FOUR_TARGET
    assert line.eojeols[3].box.left == pytest.approx(410.3066666666667)
    assert line.eojeols[3].box.right == pytest.approx(504.97333333333336)

_MISPLACED_CURLY_STRUCTURED_LINE = BoundingBox(
    95.64,
    638.804347826087,
    724.36,
    700.4347826086956,
)
_MISPLACED_CURLY_STRUCTURED_SEGMENTS = (
    (0, 174),
    (173, 352),
    (351, 630),
)
_MISPLACED_CURLY_STRUCTURED_TARGET = chr(0xACFC)
_MISPLACED_CURLY_STRUCTURED_FOLLOWING = "<8" + chr(0xD56D)
_MISPLACED_CURLY_STRUCTURED_WRAPPER = (
    chr(0x2018) + _MISPLACED_CURLY_STRUCTURED_TARGET + chr(0x2019)
)
_MISPLACED_CURLY_STRUCTURED_ASCII_WRAPPER = (
    "'" + _MISPLACED_CURLY_STRUCTURED_TARGET + "'"
)
_MISPLACED_CURLY_STRUCTURED_RAW = (
    chr(0x2018)
    + _MISPLACED_CURLY_STRUCTURED_TARGET
    + "'"
    + _MISPLACED_CURLY_STRUCTURED_FOLLOWING
    + chr(0x2019)
)
_MISPLACED_CURLY_STRUCTURED_TEXTS = (
    "<3" + chr(0xB300),
    chr(0xADDC) + chr(0xC728) + ">",
    _MISPLACED_CURLY_STRUCTURED_RAW,
)
_MISPLACED_CURLY_STRUCTURED_CTC = {
    **{
        threshold: ((0, 121), (120, 279))
        for threshold in (0.0001, 0.0003, 0.0005, 0.001)
    },
    **{
        threshold: ((0, 279),)
        for threshold in (
            0.002,
            0.003,
            0.005,
            0.007,
            0.01,
            0.015,
            0.02,
            0.03,
            0.04,
            0.05,
            0.07,
        )
    },
}


def misplaced_curly_structured_crop() -> Image.Image:
    crop = Image.new("RGB", (630, 63))
    for intensity, (left, right) in zip(
        (10, 20, 90),
        _MISPLACED_CURLY_STRUCTURED_SEGMENTS,
        strict=True,
    ):
        crop.paste((intensity, intensity, intensity), (left, 0, right, 63))
    candidate_left = _MISPLACED_CURLY_STRUCTURED_SEGMENTS[2][0]
    crop.paste((100, 100, 100), (candidate_left, 0, candidate_left + 36, 63))
    crop.paste((110, 110, 110), (candidate_left + 15, 0, candidate_left + 90, 63))
    crop.paste((120, 120, 120), (candidate_left + 75, 0, candidate_left + 125, 63))
    crop.paste((130, 130, 130), (candidate_left + 115, 0, candidate_left + 279, 63))
    return crop


def misplaced_curly_structured_words(
    *,
    selected: bool = True,
    candidate_text: str = _MISPLACED_CURLY_STRUCTURED_RAW,
    candidate_confidence: float | None = None,
    candidate_box: BoundingBox | None = None,
    first_text: str = _MISPLACED_CURLY_STRUCTURED_TEXTS[0],
) -> list[tuple[str, BoundingBox, float]]:
    if candidate_confidence is None:
        candidate_confidence = 0.508923 if selected else 0.495151
    confidences = (0.995817, 0.995666, candidate_confidence)
    values = []
    for index, ((left, right), text, confidence) in enumerate(
        zip(
            _MISPLACED_CURLY_STRUCTURED_SEGMENTS,
            _MISPLACED_CURLY_STRUCTURED_TEXTS,
            confidences,
            strict=True,
        )
    ):
        if index == 0:
            text = first_text
        if index == 2:
            text = candidate_text
        box = BoundingBox(
            _MISPLACED_CURLY_STRUCTURED_LINE.left + left,
            _MISPLACED_CURLY_STRUCTURED_LINE.top,
            _MISPLACED_CURLY_STRUCTURED_LINE.left + right,
            _MISPLACED_CURLY_STRUCTURED_LINE.bottom,
        )
        if index == 2 and candidate_box is not None:
            box = candidate_box
        values.append((text, box, confidence))
    return values


class ConfirmedMisplacedCurlyStructuredRecognizer:
    def __init__(
        self,
        *,
        default_segments: tuple[tuple[int, int], ...] = (
            _MISPLACED_CURLY_STRUCTURED_SEGMENTS
        ),
        ctc_override: tuple[
            float, tuple[tuple[int, int], ...]
        ] | None = None,
        candidate_enhanced: RecognizedText | None = None,
        wrapper_variant: RecognizedText | None = None,
        target_variant: RecognizedText | None = None,
        following_variant: RecognizedText | None = None,
        opening_variant: RecognizedText | None = None,
        closing_variant: RecognizedText | None = None,
    ) -> None:
        self.default_segments = default_segments
        self.ctc_override = ctc_override
        self.candidate_enhanced = candidate_enhanced or RecognizedText(
            _MISPLACED_CURLY_STRUCTURED_RAW,
            0.508923,
        )
        self.wrapper_variant = wrapper_variant
        self.target_variant = target_variant or RecognizedText(
            _MISPLACED_CURLY_STRUCTURED_TARGET,
            0.9998,
        )
        self.following_variant = following_variant or RecognizedText(
            _MISPLACED_CURLY_STRUCTURED_FOLLOWING,
            0.998,
        )
        self.opening_variant = opening_variant or RecognizedText("'", 0.8)
        self.closing_variant = closing_variant or RecognizedText("'", 0.99)
        self.recognition_calls = 0

    def word_boxes(self, image, *, space_threshold=None):
        if image.size == (630, 63) and space_threshold is None:
            return self.default_segments
        if image.size != (279, 63) or space_threshold is None:
            return ()
        if (
            self.ctc_override is not None
            and space_threshold == self.ctc_override[0]
        ):
            return self.ctc_override[1]
        return _MISPLACED_CURLY_STRUCTURED_CTC[space_threshold]

    def recognize(self, image):
        self.recognition_calls += 1
        enhanced = image.height == 126
        original_width = image.width // 2 if enhanced else image.width
        pixel = image.getpixel((image.width // 2, image.height // 2))
        intensity = pixel[0] if isinstance(pixel, tuple) else pixel
        segment_reads = {
            (174, 10): RecognizedText(
                _MISPLACED_CURLY_STRUCTURED_TEXTS[0],
                0.995817,
            ),
            (179, 20): RecognizedText(
                _MISPLACED_CURLY_STRUCTURED_TEXTS[1],
                0.995666,
            ),
        }
        if not enhanced and (original_width, intensity) in segment_reads:
            return segment_reads[(original_width, intensity)]
        if image.size == (279, 63):
            return RecognizedText(_MISPLACED_CURLY_STRUCTURED_RAW, 0.495151)
        if image.size == (558, 126):
            return self.candidate_enhanced
        if 115 <= original_width <= 125:
            if self.wrapper_variant is not None:
                return self.wrapper_variant
            if original_width in (123, 125) or (
                original_width == 121 and not enhanced
            ):
                text = _MISPLACED_CURLY_STRUCTURED_WRAPPER
            else:
                text = _MISPLACED_CURLY_STRUCTURED_ASCII_WRAPPER
            confidence = 0.99 if original_width == 121 else 0.94
            return RecognizedText(text, confidence)
        if 63 <= original_width <= 68:
            return self.target_variant
        if 20 <= original_width <= 36:
            return self.opening_variant
        if 40 <= original_width <= 47:
            return self.closing_variant
        if 154 <= original_width <= 164:
            return self.following_variant
        return RecognizedText("", 0.0)


def test_confirmed_misplaced_curly_structured_split_recovers() -> None:
    selected = misplaced_curly_structured_words()
    recovered = _recover_confirmed_misplaced_curly_single_plus_structured_split(
        selected,
        misplaced_curly_structured_words(selected=False),
        misplaced_curly_structured_crop(),
        _MISPLACED_CURLY_STRUCTURED_LINE,
        ConfirmedMisplacedCurlyStructuredRecognizer(),
    )

    assert recovered == [
        *selected[:2],
        (
            _MISPLACED_CURLY_STRUCTURED_WRAPPER,
            BoundingBox(
                _MISPLACED_CURLY_STRUCTURED_LINE.left + 351,
                _MISPLACED_CURLY_STRUCTURED_LINE.top,
                _MISPLACED_CURLY_STRUCTURED_LINE.left + 472,
                _MISPLACED_CURLY_STRUCTURED_LINE.bottom,
            ),
            0.495151,
        ),
        (
            _MISPLACED_CURLY_STRUCTURED_FOLLOWING,
            BoundingBox(
                _MISPLACED_CURLY_STRUCTURED_LINE.left + 471,
                _MISPLACED_CURLY_STRUCTURED_LINE.top,
                _MISPLACED_CURLY_STRUCTURED_LINE.left + 630,
                _MISPLACED_CURLY_STRUCTURED_LINE.bottom,
            ),
            0.495151,
        ),
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "candidate_enhanced": RecognizedText(
                _MISPLACED_CURLY_STRUCTURED_RAW[:-1],
                0.9,
            )
        },
        {
            "wrapper_variant": RecognizedText(
                '"' + _MISPLACED_CURLY_STRUCTURED_TARGET + '"',
                0.99,
            )
        },
        {"target_variant": RecognizedText(chr(0xC790), 1.0)},
        {"following_variant": RecognizedText("<7" + chr(0xD56D), 1.0)},
        {"opening_variant": RecognizedText('"', 1.0)},
        {"closing_variant": RecognizedText('"', 1.0)},
        {"ctc_override": (0.001, ((0, 279),))},
    ],
)
def test_confirmed_misplaced_curly_structured_requires_crop_evidence(
    overrides,
) -> None:
    selected = misplaced_curly_structured_words()

    assert (
        _recover_confirmed_misplaced_curly_single_plus_structured_split(
            selected,
            misplaced_curly_structured_words(selected=False),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(**overrides),
        )
        == selected
    )


@pytest.mark.parametrize(
    ("words", "raw_words", "crop", "line_box", "recognizer"),
    [
        (
            misplaced_curly_structured_words() * 2,
            misplaced_curly_structured_words(selected=False),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(
                candidate_text='"'
                + _MISPLACED_CURLY_STRUCTURED_RAW[1:],
            ),
            misplaced_curly_structured_words(
                selected=False,
                candidate_text='"'
                + _MISPLACED_CURLY_STRUCTURED_RAW[1:],
            ),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(candidate_confidence=0.5088),
            misplaced_curly_structured_words(selected=False),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(),
            misplaced_curly_structured_words(
                selected=False,
                candidate_confidence=0.4950,
            ),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(
                first_text="A3" + chr(0xB300),
            ),
            misplaced_curly_structured_words(
                selected=False,
                first_text="A3" + chr(0xB300),
            ),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(
                candidate_box=BoundingBox(
                    _MISPLACED_CURLY_STRUCTURED_LINE.left + 352,
                    _MISPLACED_CURLY_STRUCTURED_LINE.top,
                    _MISPLACED_CURLY_STRUCTURED_LINE.left + 630,
                    _MISPLACED_CURLY_STRUCTURED_LINE.bottom,
                )
            ),
            misplaced_curly_structured_words(
                selected=False,
                candidate_box=BoundingBox(
                    _MISPLACED_CURLY_STRUCTURED_LINE.left + 352,
                    _MISPLACED_CURLY_STRUCTURED_LINE.top,
                    _MISPLACED_CURLY_STRUCTURED_LINE.left + 630,
                    _MISPLACED_CURLY_STRUCTURED_LINE.bottom,
                ),
            ),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(),
            misplaced_curly_structured_words(selected=False),
            Image.new("RGB", (629, 63)),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(),
            misplaced_curly_structured_words(selected=False),
            misplaced_curly_structured_crop(),
            BoundingBox(
                95.64,
                638.804347826087,
                724.38,
                700.4347826086956,
            ),
            ConfirmedMisplacedCurlyStructuredRecognizer(),
        ),
        (
            misplaced_curly_structured_words(),
            misplaced_curly_structured_words(selected=False),
            misplaced_curly_structured_crop(),
            _MISPLACED_CURLY_STRUCTURED_LINE,
            ConfirmedMisplacedCurlyStructuredRecognizer(
                default_segments=_MISPLACED_CURLY_STRUCTURED_SEGMENTS[:-1]
            ),
        ),
    ],
)
def test_confirmed_misplaced_curly_structured_requires_exact_profile(
    words,
    raw_words,
    crop,
    line_box,
    recognizer,
) -> None:
    assert (
        _recover_confirmed_misplaced_curly_single_plus_structured_split(
            words,
            raw_words,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.recognition_calls == 0


class MisplacedCurlyStructuredDetector:
    def detect(self, _image):
        return (
            DetectedRegion(
                _MISPLACED_CURLY_STRUCTURED_LINE,
                0.991306,
            ),
        )


def test_engine_recovers_misplaced_curly_structured_segment() -> None:
    image = Image.new("RGB", (1280, 720))
    image.paste(misplaced_curly_structured_crop(), (95, 638))
    engine = PaddleOcrEngine(
        MisplacedCurlyStructuredDetector(),
        ConfirmedMisplacedCurlyStructuredRecognizer(),
    )

    document = engine.recognize(image)

    line = document.lines[0]
    assert len(line.text) == 15
    assert [len(word.text) for word in line.eojeols] == [3, 3, 1, 3]
    assert line.eojeols[2].text == _MISPLACED_CURLY_STRUCTURED_TARGET
    assert line.eojeols[2].box.left == pytest.approx(486.97333333333336)
    assert line.eojeols[2].box.right == pytest.approx(527.3066666666666)


_ELLIPSIS_FIRST = chr(0xB300) + chr(0xAE30)
_ELLIPSIS_LAST = chr(0xC5C5) + chr(0x2026)
_ELLIPSIS_CORE = _ELLIPSIS_FIRST + _ELLIPSIS_LAST[0]


def terminal_ellipsis_words(
    *,
    first_text: str = _ELLIPSIS_FIRST,
    last_text: str = _ELLIPSIS_LAST,
    first_confidence: float = 0.9997,
    last_confidence: float = 0.9989,
    previous_right: float = 40.0,
    first_left: float = 52.4,
    last_left: float = 94.62,
    last_right: float = 126.62,
    following_left: float = 138.0,
):
    previous_text = chr(0xB098) + chr(0xB77C)
    following_text = chr(0xB178) + chr(0xB3D9)
    return [
        (
            previous_text,
            BoundingBox(0, 0, previous_right, 20),
            0.999,
        ),
        (
            first_text,
            BoundingBox(first_left, 0, 87.4, 20),
            first_confidence,
        ),
        (
            last_text,
            BoundingBox(last_left, 0, last_right, 20),
            last_confidence,
        ),
        (
            following_text,
            BoundingBox(following_left, 0, 198, 20),
            0.999,
        ),
    ]


class TerminalEllipsisPairRecognizer:
    def __init__(
        self,
        *,
        full: RecognizedText | None = None,
        core_direct: RecognizedText | None = None,
        core_enhanced: RecognizedText | None = None,
    ) -> None:
        self.full = full or RecognizedText(
            _ELLIPSIS_CORE + chr(0x2026),
            0.9998,
        )
        self.core_direct = core_direct or RecognizedText(
            _ELLIPSIS_CORE,
            0.9998,
        )
        self.core_enhanced = core_enhanced or RecognizedText(
            _ELLIPSIS_CORE,
            0.9999,
        )
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        if image.size == (75, 20):
            return self.full
        if image.size == (59, 20):
            return self.core_direct
        if image.size == (118, 40):
            return self.core_enhanced
        return RecognizedText("", 0.0)


def test_two_plus_terminal_ellipsis_pair_merges() -> None:
    words = terminal_ellipsis_words()

    recovered = _recover_isolated_close_word_pairs(
        words,
        Image.new("RGB", (220, 20)),
        BoundingBox(0, 0, 220, 20),
        TerminalEllipsisPairRecognizer(),
    )

    assert recovered == [
        words[0],
        (
            _ELLIPSIS_CORE + chr(0x2026),
            BoundingBox(52.4, 0, 126.62, 20),
            0.9989,
        ),
        words[3],
    ]


@pytest.mark.parametrize(
    "profile",
    [
        {"first_text": chr(0xB300)},
        {"last_text": chr(0xC5C5) + "?"},
        {"first_confidence": 0.9996},
        {"last_confidence": 0.9987},
        {"last_left": 94.58},
        {"previous_right": 40.3},
        {"following_left": 137.5},
        {
            "last_right": 134.62,
            "following_left": 146.0,
        },
    ],
)
def test_terminal_ellipsis_pair_requires_profile(profile) -> None:
    words = terminal_ellipsis_words(**profile)
    recognizer = TerminalEllipsisPairRecognizer()

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (220, 20)),
            BoundingBox(0, 0, 220, 20),
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


@pytest.mark.parametrize(
    "recognizer",
    [
        TerminalEllipsisPairRecognizer(
            full=RecognizedText(_ELLIPSIS_CORE, 1.0),
        ),
        TerminalEllipsisPairRecognizer(
            full=RecognizedText(
                _ELLIPSIS_CORE + chr(0x2026),
                0.9996,
            ),
        ),
        TerminalEllipsisPairRecognizer(
            core_direct=RecognizedText(chr(0xB300), 1.0),
        ),
        TerminalEllipsisPairRecognizer(
            core_direct=RecognizedText(
                _ELLIPSIS_CORE,
                0.9996,
            ),
        ),
        TerminalEllipsisPairRecognizer(
            core_enhanced=RecognizedText(
                chr(0xB300),
                1.0,
            ),
        ),
        TerminalEllipsisPairRecognizer(
            core_enhanced=RecognizedText(
                _ELLIPSIS_CORE,
                0.9997,
            ),
        ),
    ],
)
def test_terminal_ellipsis_pair_requires_consensus(
    recognizer,
) -> None:
    words = terminal_ellipsis_words()

    assert (
        _recover_isolated_close_word_pairs(
            words,
            Image.new("RGB", (220, 20)),
            BoundingBox(0, 0, 220, 20),
            recognizer,
        )
        == words
    )


_SYMBOL_JAMO_LINE = BoundingBox(55.68, 201.72, 1162.32, 231.65)
_SYMBOL_JAMO_SELECTED = (0, 1, 3, 4, 5, 6, 7, 8, 9)
_SYMBOL_JAMO_RECOVERED = chr(0xC591)


def symbol_jamo_words():
    texts = (
        chr(0xAC00) + chr(0xB294),
        "".join(
            chr(value)
            for value in (0xD604, 0xC2E4, 0xC5D0, 0xC11C)
        )
        + ",",
        "%",
        chr(0x3151) + "?",
        "".join(
            chr(value)
            for value in (0xBD80, 0xCC98, 0xC640)
        ),
        "".join(
            chr(value)
            for value in (
                0xAC8C,
                0xC784,
                0xC0B0,
                0xC5C5,
                0xACC4,
                0xC758,
            )
        ),
        "".join(
            chr(value)
            for value in (0xACC4, 0xC18D, 0xB418, 0xB294)
        ),
        "".join(
            chr(value)
            for value in (0xC785, 0xC7A5, 0xCC28, 0xB85C)
        ),
        chr(0xC778) + chr(0xD574),
        "".join(
            chr(value)
            for value in (0xBC95, 0xC548, 0xC774)
        ),
        "2",
    )
    bounds = (
        (129.68, 180.68),
        (193.68, 317.68),
        (331.68, 347.68),
        (346.68, 372.68),
        (384.68, 473.68),
        (485.68, 660.68),
        (675.68, 793.68),
        (808.68, 924.68),
        (939.68, 992.68),
        (1010.68, 1092.68),
        (1106.68, 1163.68),
    )
    confidences = (
        0.999887,
        0.979528,
        0.491437,
        0.860453,
        0.999892,
        0.999797,
        0.998883,
        0.999785,
        0.999891,
        0.999423,
        0.266216,
    )
    raw = [
        (
            text,
            BoundingBox(
                left,
                _SYMBOL_JAMO_LINE.top,
                right,
                _SYMBOL_JAMO_LINE.bottom,
            ),
            confidence,
        )
        for text, (left, right), confidence in zip(
            texts,
            bounds,
            confidences,
            strict=True,
        )
    ]
    return [raw[index] for index in _SYMBOL_JAMO_SELECTED], raw


class SymbolJamoRecognizer:
    def __init__(
        self,
        *,
        full_direct: RecognizedText | None = None,
        full_enhanced: RecognizedText | None = None,
        core_direct: RecognizedText | None = None,
        core_enhanced: RecognizedText | None = None,
    ) -> None:
        self.full_direct = full_direct or RecognizedText(
            _SYMBOL_JAMO_RECOVERED + "?",
            0.99937,
        )
        self.full_enhanced = full_enhanced or RecognizedText(
            _SYMBOL_JAMO_RECOVERED + "?",
            0.9991,
        )
        self.core_direct = core_direct or RecognizedText(
            _SYMBOL_JAMO_RECOVERED,
            0.9978,
        )
        self.core_enhanced = core_enhanced or RecognizedText(
            _SYMBOL_JAMO_RECOVERED,
            0.9978,
        )
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        readings = {
            (41, 31): self.full_direct,
            (82, 62): self.full_enhanced,
            (28, 31): self.core_direct,
            (56, 62): self.core_enhanced,
        }
        return readings.get(
            image.size,
            RecognizedText("", 0.0),
        )


def test_overlapping_symbol_jamo_single_recovers() -> None:
    words, raw = symbol_jamo_words()

    recovered = _recover_confirmed_overlapping_symbol_jamo_single(
        words,
        raw,
        Image.new("RGB", (1108, 31)),
        _SYMBOL_JAMO_LINE,
        SymbolJamoRecognizer(),
    )

    assert recovered == [
        *words[:2],
        (
            _SYMBOL_JAMO_RECOVERED + "?",
            BoundingBox(331.68, 201.72, 372.68, 231.65),
            0.491437,
        ),
        *words[3:],
    ]


@pytest.mark.parametrize(
    "case",
    [
        "word_count",
        "raw_count",
        "mapping",
        "crop_size",
        "line_height",
        "shape",
        "confidence",
        "geometry",
    ],
)
def test_overlapping_symbol_jamo_single_requires_profile(
    case,
) -> None:
    words, raw = symbol_jamo_words()
    crop = Image.new("RGB", (1108, 31))
    line_box = _SYMBOL_JAMO_LINE
    if case == "word_count":
        words = words[:-1]
    elif case == "raw_count":
        raw = raw[:-1]
    elif case == "mapping":
        words[2] = raw[2]
    elif case == "crop_size":
        crop = Image.new("RGB", (1107, 31))
    elif case == "line_height":
        line_box = BoundingBox(
            55.68,
            201.72,
            1162.32,
            201.72,
        )
    elif case == "shape":
        raw[3] = (
            chr(0x3151) + "!",
            raw[3][1],
            raw[3][2],
        )
        words[2] = raw[3]
    elif case == "confidence":
        raw[2] = (
            raw[2][0],
            raw[2][1],
            0.4912,
        )
    else:
        raw[2] = (
            raw[2][0],
            BoundingBox(331.68, 201.72, 348.68, 231.65),
            raw[2][2],
        )
    recognizer = SymbolJamoRecognizer()

    assert (
        _recover_confirmed_overlapping_symbol_jamo_single(
            words,
            raw,
            crop,
            line_box,
            recognizer,
        )
        == words
    )
    assert recognizer.calls == 0


@pytest.mark.parametrize(
    "recognizer",
    [
        SymbolJamoRecognizer(
            full_direct=RecognizedText(
                chr(0xC790) + "?",
                1.0,
            )
        ),
        SymbolJamoRecognizer(
            full_direct=RecognizedText(
                _SYMBOL_JAMO_RECOVERED + "?",
                0.9992,
            )
        ),
        SymbolJamoRecognizer(
            full_enhanced=RecognizedText(
                chr(0xC790) + "?",
                1.0,
            )
        ),
        SymbolJamoRecognizer(
            full_enhanced=RecognizedText(
                _SYMBOL_JAMO_RECOVERED + "?",
                0.9989,
            )
        ),
        SymbolJamoRecognizer(
            core_direct=RecognizedText(
                chr(0xC790),
                1.0,
            )
        ),
        SymbolJamoRecognizer(
            core_direct=RecognizedText(
                _SYMBOL_JAMO_RECOVERED,
                0.9975,
            )
        ),
        SymbolJamoRecognizer(
            core_enhanced=RecognizedText(
                chr(0xC790),
                1.0,
            )
        ),
        SymbolJamoRecognizer(
            core_enhanced=RecognizedText(
                _SYMBOL_JAMO_RECOVERED,
                0.9975,
            )
        ),
    ],
)
def test_overlapping_symbol_jamo_single_requires_consensus(
    recognizer,
) -> None:
    words, raw = symbol_jamo_words()

    assert (
        _recover_confirmed_overlapping_symbol_jamo_single(
            words,
            raw,
            Image.new("RGB", (1108, 31)),
            _SYMBOL_JAMO_LINE,
            recognizer,
        )
        == words
    )
