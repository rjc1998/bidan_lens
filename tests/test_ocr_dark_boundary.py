import pytest
from PIL import Image, ImageDraw

from bidan_lens.models import BoundingBox, OcrDocument
from bidan_lens.ocr.base import RecognizedText
from bidan_lens.ocr.paddle import (
    _line_from_words,
    _recover_confirmed_dark_three_plus_five_split,
    _recover_word_boundaries,
)
from bidan_lens.pipeline.hit_test import hit_test

TEXT = '가나다라마바사아'
BOX = BoundingBox(10, 20, 148, 38)
SEGMENTS = ((0, 48), (54, 138))
READINGS = (
    RecognizedText(TEXT[:3], 0.999),
    RecognizedText(TEXT[3:], 0.998),
    RecognizedText(TEXT[:3], 0.9994),
    RecognizedText(TEXT[3:], 0.9993),
    RecognizedText(TEXT[:3] + ' ' + TEXT[3:], 0.90),
    RecognizedText(TEXT[:3] + ' ' + TEXT[3:], 0.88),
)


def dark_crop(background: int = 24, foreground: int = 240) -> Image.Image:
    image = Image.new('RGB', (138, 18), (background,) * 3)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 43, 13), fill=(foreground,) * 3)
    draw.rectangle((58, 4, 133, 13), fill=(foreground,) * 3)
    return image


class Recognizer:
    supports_binarized_small_text_retry = True

    def __init__(self, readings=READINGS, segments=SEGMENTS):
        self.readings = iter(readings)
        self.segments = segments
        self.images: list[Image.Image] = []
        self.thresholds: list[float] = []

    def word_boxes(self, image, space_threshold=0.07):
        self.thresholds.append(space_threshold)
        return self.segments

    def recognize(self, image):
        self.images.append(image.copy())
        return next(self.readings)


def test_dark_boundary_restores_context_and_excludes_the_new_space_from_hover() -> None:
    recognizer = Recognizer()
    image = dark_crop()
    before = image.tobytes()

    words = _recover_word_boundaries([(TEXT, BOX, 0.9968)], image, BOX, recognizer)

    assert words == [
        (TEXT[:3], BoundingBox(10, 20, 58, 38), 0.9968),
        (TEXT[3:], BoundingBox(64, 20, 148, 38), 0.9968),
    ]
    assert recognizer.thresholds == [0.02, 0.07]
    assert [image.size for image in recognizer.images] == [
        (48, 18), (84, 18), (48, 18), (84, 18), (276, 36), (276, 36),
    ]
    assert recognizer.images[4].getpixel((0, 0)) == (0, 0, 0)
    assert recognizer.images[5].getpixel((0, 0)) == (255, 255, 255)
    assert image.tobytes() == before
    line = _line_from_words(words)
    assert line.text == TEXT[:3] + ' ' + TEXT[3:]
    assert [(word.sentence_start, word.sentence_end) for word in line.eojeols] == [
        (0, 3), (4, 9),
    ]
    document = OcrDocument((line,), 1.0)
    target = hit_test(document, 34, 29)
    assert target is not None
    assert target.surface == TEXT[:3]
    assert target.sentence == line.text
    assert hit_test(document, 61, 29) is None


@pytest.mark.parametrize(('index', 'bad'), [
    (0, RecognizedText('가나다', 0.9969)),
    (1, RecognizedText('라마바사아', 0.9969)),
    (2, RecognizedText('가나다', 0.9979)),
    (3, RecognizedText('라마바사아', 0.9979)),
    (0, RecognizedText('가나마', 0.9999)),
    (1, RecognizedText('라마바사어', 0.9999)),
    (2, RecognizedText('가나다라', 0.9999)),
    (3, RecognizedText('라마바A아', 0.9999)),
    (4, RecognizedText(TEXT, 0.9999)),
    (5, RecognizedText(TEXT, 0.9999)),
    (4, RecognizedText('가나다 라마바사아', 0.8499)),
    (5, RecognizedText('가나다 라마바사아', 0.8499)),
    (4, RecognizedText('가나다 라마바사어', 0.9999)),
    (5, RecognizedText('가나다  라마바사아', 0.9999)),
])
def test_dark_boundary_requires_all_character_and_space_confirmations(
    index: int, bad: RecognizedText,
) -> None:
    recognizer = Recognizer(READINGS[:index] + (bad,))
    words = [(TEXT, BOX, 0.9968)]

    assert _recover_confirmed_dark_three_plus_five_split(
        words, dark_crop(), BOX, recognizer
    ) == words
    assert len(recognizer.images) == index + 1


@pytest.mark.parametrize('segments', [
    ((0, 138),),
    ((0, 48), (48, 54), (54, 138)),
    ((2, 48), (54, 138)),
    ((0, 48), (54, 136)),
    ((0, 49), (54, 138)),
    ((0, 45), (54, 138)),
    ((0, 41), (47, 138)),
])
def test_dark_boundary_requires_complete_segments_gap_and_compatible_pitch(segments) -> None:
    recognizer = Recognizer((), segments)
    words = [(TEXT, BOX, 0.9968)]

    assert _recover_confirmed_dark_three_plus_five_split(
        words, dark_crop(), BOX, recognizer
    ) == words
    assert recognizer.thresholds == [0.07]
    assert not recognizer.images


@pytest.mark.parametrize(('background', 'foreground'), [
    (240, 24), (65, 240), (24, 151), (24, 24),
])
def test_dark_boundary_skips_bright_or_low_contrast_crops(
    background: int, foreground: int,
) -> None:
    recognizer = Recognizer(())
    words = [(TEXT, BOX, 0.9968)]

    assert _recover_confirmed_dark_three_plus_five_split(
        words, dark_crop(background, foreground), BOX, recognizer
    ) == words
    assert not recognizer.thresholds
    assert not recognizer.images


@pytest.mark.parametrize(('text', 'confidence', 'height'), [
    (TEXT, 0.9959, 18),
    (TEXT[:-1], 0.999, 18),
    (TEXT + '가', 0.999, 18),
    (TEXT[:-1] + 'A', 0.999, 18),
    (TEXT, 0.999, 20.01),
    (TEXT, 0.999, 0),
])
def test_dark_boundary_requires_supported_word_shape_and_size(
    text: str, confidence: float, height: float,
) -> None:
    recognizer = Recognizer(())
    words = [(text, BOX, confidence)]

    assert _recover_confirmed_dark_three_plus_five_split(
        words, dark_crop(), BoundingBox(10, 20, 148, 20 + height), recognizer
    ) == words
    assert not recognizer.thresholds
    assert not recognizer.images


def test_dark_boundary_requires_recognizer_capability() -> None:
    words = [(TEXT, BOX, 0.999)]

    assert _recover_confirmed_dark_three_plus_five_split(
        words, dark_crop(), BOX, object()
    ) is words
