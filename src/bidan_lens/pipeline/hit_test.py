from __future__ import annotations

from bidan_lens.models import HoverTarget, OcrDocument, OcrEojeol
from bidan_lens.ocr.hangul import is_hangul

_HIT_INSET_RATIO = 0.2


def _contains_hover_target(
    eojeol: OcrEojeol, x: float, y: float, padding: float
) -> bool:
    inset_x = eojeol.box.width * _HIT_INSET_RATIO
    inset_y = eojeol.box.height * _HIT_INSET_RATIO
    if not (
        eojeol.box.left + inset_x - padding
        <= x
        <= eojeol.box.right - inset_x + padding
        and eojeol.box.top + inset_y - padding
        <= y
        <= eojeol.box.bottom - inset_y + padding
    ):
        return False
    if not eojeol.glyphs:
        return True
    return any(
        is_hangul(glyph.text) and glyph.box.contains(x, y, padding)
        for glyph in eojeol.glyphs
    )


def hit_test(document: OcrDocument, x: float, y: float, padding: float = 0.0) -> HoverTarget | None:
    local_x = x - document.origin_x
    local_y = y - document.origin_y
    matches = []
    for line in document.lines:
        for eojeol in line.eojeols:
            if _contains_hover_target(eojeol, local_x, local_y, padding):
                matches.append((eojeol.box.area, line, eojeol))
    if not matches:
        return None
    _, line, eojeol = min(matches, key=lambda value: value[0])
    return HoverTarget(
        surface=eojeol.text,
        sentence=line.text,
        sentence_start=eojeol.sentence_start,
        sentence_end=eojeol.sentence_end,
        box=eojeol.box.translated(document.origin_x, document.origin_y),
        confidence=eojeol.confidence,
    )
