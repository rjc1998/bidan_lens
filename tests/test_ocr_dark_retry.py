import pytest
from PIL import Image, ImageDraw

from bidan_lens.models import BoundingBox
from bidan_lens.ocr.base import RecognizedText
from bidan_lens.ocr.paddle import PaddleOcrEngine, _retry_dark_background_hangul_pair


def dark_crop(background: int = 24, foreground: int = 240) -> Image.Image:
    image = Image.new('RGB', (34, 18), (background,) * 3)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 11, 13), fill=(foreground,) * 3)
    draw.rectangle((21, 4, 29, 13), fill=(foreground,) * 3)
    return image


class RetryRecognizer:
    supports_binarized_small_text_retry = True

    def __init__(self, readings: tuple[RecognizedText, ...]) -> None:
        self.readings = iter(readings)
        self.images: list[Image.Image] = []

    def recognize(self, image: Image.Image) -> RecognizedText:
        self.images.append(image.copy())
        return next(self.readings)


def test_dark_pair_retry_normalizes_polarity_and_retains_weakest_consensus() -> None:
    image = dark_crop()
    before = image.tobytes()
    recognizer = RetryRecognizer((
        RecognizedText('결 정', 0.98),
        RecognizedText('결정', 0.97),
        RecognizedText(' 결정 ', 0.99),
    ))

    result = _retry_dark_background_hangul_pair(
        image, 18, RecognizedText('길정', 0.8), recognizer
    )

    assert result == RecognizedText('결정', 0.97)
    assert image.tobytes() == before
    assert len(recognizer.images) == 3
    for retry_image in recognizer.images:
        assert retry_image.size == (68, 36)
        assert retry_image.mode == 'RGB'
        assert retry_image.getpixel((0, 0)) == (255, 255, 255)
        assert retry_image.getpixel((15, 15)) == (0, 0, 0)


@pytest.mark.parametrize('failure_index', [0, 1, 2])
@pytest.mark.parametrize('failed_reading', [
    RecognizedText('길정', 0.99),
    RecognizedText('결', 0.99),
    RecognizedText('결정은', 0.99),
    RecognizedText('결A', 0.99),
    RecognizedText('', 0.99),
    RecognizedText('결정', 0.969),
])
def test_dark_pair_retry_stops_on_invalid_evidence(
    failure_index: int, failed_reading: RecognizedText,
) -> None:
    original = RecognizedText('길정', 0.8)
    recognizer = RetryRecognizer(
        (RecognizedText('결정', 0.99),) * failure_index + (failed_reading,)
    )

    assert _retry_dark_background_hangul_pair(
        dark_crop(), 18, original, recognizer
    ) is original
    assert len(recognizer.images) == failure_index + 1


def test_dark_pair_retry_rejects_disagreement() -> None:
    original = RecognizedText('길정', 0.8)
    recognizer = RetryRecognizer((
        RecognizedText('결정', 0.99),
        RecognizedText('가정', 0.99),
    ))

    assert _retry_dark_background_hangul_pair(
        dark_crop(), 18, original, recognizer
    ) is original
    assert len(recognizer.images) == 2


@pytest.mark.parametrize(('height', 'text', 'confidence'), [
    (0, '길정', 0.8),
    (-1, '길정', 0.8),
    (20.01, '길정', 0.8),
    (18, '길', 0.8),
    (18, '길정은', 0.8),
    (18, '길A', 0.8),
    (18, '길정', 0.97),
])
def test_dark_pair_retry_skips_unsupported_readings(
    height: float, text: str, confidence: float,
) -> None:
    original = RecognizedText(text, confidence)
    recognizer = RetryRecognizer(())

    assert _retry_dark_background_hangul_pair(
        dark_crop(), height, original, recognizer
    ) is original
    assert not recognizer.images


@pytest.mark.parametrize(('background', 'foreground'), [
    (240, 24),
    (65, 240),
    (24, 151),
    (24, 24),
])
def test_dark_pair_retry_requires_dark_background_and_contrast(
    background: int, foreground: int,
) -> None:
    original = RecognizedText('길정', 0.8)
    recognizer = RetryRecognizer(())

    assert _retry_dark_background_hangul_pair(
        dark_crop(background, foreground), 18, original, recognizer
    ) is original
    assert not recognizer.images


def test_dark_pair_retry_requires_recognizer_support() -> None:
    original = RecognizedText('길정', 0.8)

    assert _retry_dark_background_hangul_pair(
        dark_crop(), 18, original, object()
    ) is original


@pytest.mark.parametrize('agree', [True, False])
def test_dark_pair_retry_preserves_sentence_span_and_word_geometry(agree: bool) -> None:
    class Recognizer:
        supports_binarized_small_text_retry = True

        def word_boxes(self, image, space_threshold=0.07):
            return ((0, 34), (44, 68)) if image.size == (68, 18) else ()

        def recognize(self, image):
            if image.size == (34, 18):
                return RecognizedText('길정', 0.8)
            if image.size == (68, 36):
                return RecognizedText('결정' if agree else '길정', 0.99)
            if image.size == (24, 18):
                return RecognizedText('하다', 0.9999)
            return RecognizedText('', 0.0)

    image = Image.new('RGB', (68, 18), (24, 24, 24))
    image.paste(dark_crop(), (0, 0))
    engine = PaddleOcrEngine(object(), Recognizer())

    line = engine._segmented_line(image, BoundingBox(10, 20, 78, 38))

    assert line is not None
    expected = '결정' if agree else '길정'
    assert line.text == expected + ' 하다'
    assert line.eojeols[0].text == expected
    assert line.eojeols[0].box == BoundingBox(10, 20, 44, 38)
    assert (line.eojeols[0].sentence_start, line.eojeols[0].sentence_end) == (0, 2)
    assert line.eojeols[1].box == BoundingBox(54, 20, 78, 38)
