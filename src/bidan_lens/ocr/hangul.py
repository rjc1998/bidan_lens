from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from bidan_lens.models import BoundingBox, OcrEojeol, OcrGlyph, OcrLine

_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7FF),
)
_TOKEN_RE = re.compile(r"\S+")
_EDGE_PUNCTUATION = " \t\r\n.,!?;:…·‘’“”\"'()[]{}<>《》〈〉「」『』【】"


def _trim_edges(value: str) -> tuple[str, int, int]:
    start = 0
    end = len(value)
    while start < end and unicodedata.category(value[start])[0] in {'P', 'Z'}:
        start += 1
    while end > start and unicodedata.category(value[end - 1])[0] in {'P', 'Z'}:
        end -= 1
    return value[start:end], start, end


def is_hangul(character: str) -> bool:
    if not character:
        return False
    codepoint = ord(character[0])
    return any(start <= codepoint <= end for start, end in _HANGUL_RANGES)


def contains_hangul(text: str) -> bool:
    return any(is_hangul(character) for character in text)


def normalize_korean(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _proportional_glyphs(text: str, box: BoundingBox, confidence: float) -> tuple[OcrGlyph, ...]:
    if not text:
        return ()
    width = box.width / len(text)
    return tuple(
        OcrGlyph(
            character,
            BoundingBox(box.left + i * width, box.top, box.left + (i + 1) * width, box.bottom),
            confidence,
            i,
            i + 1,
        )
        for i, character in enumerate(text)
    )


def make_line(
    text: str,
    box: BoundingBox,
    confidence: float,
    character_boxes: Sequence[BoundingBox] | None = None,
    character_confidences: Sequence[float] | None = None,
) -> OcrLine:
    """Create a line and whole-eojeol boxes from recognition output.

    Paddle recognition does not always expose CTC character alignment. In that case,
    proportional character boxes are intentionally marked only by the line confidence.
    The geometry can later be replaced by an aligned recognizer without changing the
    hit-testing contract.
    """

    text = normalize_korean(text)
    if character_boxes and len(character_boxes) == len(text):
        confidences = character_confidences or [confidence] * len(text)
        glyphs = tuple(
            OcrGlyph(char, char_box, float(confidences[index]), index, index + 1)
            for index, (char, char_box) in enumerate(zip(text, character_boxes, strict=True))
        )
    else:
        glyphs = _proportional_glyphs(text, box, confidence)

    eojeols: list[OcrEojeol] = []
    for match in _TOKEN_RE.finditer(text):
        raw_start, raw_end = match.span()
        raw = match.group()
        stripped, left_trim, right_trim = _trim_edges(raw)
        if not stripped or not contains_hangul(stripped):
            continue
        start = raw_start + left_trim
        end = raw_start + right_trim
        token_glyphs = glyphs[start:end]
        token_box = BoundingBox.union([glyph.box for glyph in token_glyphs])
        token_confidence = min((glyph.confidence for glyph in token_glyphs), default=confidence)
        eojeols.append(OcrEojeol(stripped, token_box, token_confidence, start, end, token_glyphs))
    return OcrLine(text, box, confidence, tuple(eojeols))
