from bidan_lens.models import BoundingBox, OcrDocument
from bidan_lens.ocr.hangul import contains_hangul, make_line
from bidan_lens.pipeline.hit_test import hit_test


def test_hangul_ranges_include_syllables_and_jamo() -> None:
    assert contains_hangul("한글")
    assert contains_hangul("ㅎㅏㄴ")
    assert not contains_hangul("English 123")


def test_make_line_excludes_all_unicode_edge_punctuation() -> None:
    line = make_line(
        "/\uc5b4\ub514\uc5d0\uc11c/ \u2014\uba39\uc5b4\uc694\u2014",
        BoundingBox(0, 0, 140, 20),
        0.91,
    )

    assert [word.text for word in line.eojeols] == [
        "\uc5b4\ub514\uc5d0\uc11c",
        "\uba39\uc5b4\uc694",
    ]


def test_make_line_keeps_eojeol_together_and_excludes_punctuation() -> None:
    line = make_line("어디에서, 먹고 싶어요?", BoundingBox(0, 0, 210, 20), 0.91)
    assert [word.text for word in line.eojeols] == ["어디에서", "먹고", "싶어요"]
    assert [(word.sentence_start, word.sentence_end) for word in line.eojeols] == [
        (0, 4),
        (6, 8),
        (9, 12),
    ]


def test_hit_test_returns_whole_word_with_sentence_context() -> None:
    line = make_line("어디에서 먹어요", BoundingBox(0, 0, 140, 20), 0.95)
    document = OcrDocument((line,), 1.0, origin_x=100, origin_y=200)
    first = line.eojeols[0].box.center
    target = hit_test(document, first[0] + 100, first[1] + 200)
    assert target is not None
    assert target.surface == "어디에서"
    assert target.sentence == "어디에서 먹어요"
    assert target.box.left >= 100


def test_hit_test_does_not_select_spaces_or_punctuation() -> None:
    line = make_line("한국어, 공부", BoundingBox(0, 0, 100, 20), 0.9)
    document = OcrDocument((line,), 1.0)
    assert hit_test(document, 48, 10) is None


def test_hit_test_rejects_eojeol_edges() -> None:
    line = make_line("\ud55c\uad6d\uc5b4", BoundingBox(0, 0, 60, 20), 0.9)
    document = OcrDocument((line,), 1.0)

    assert hit_test(document, 2, 10) is None
    assert hit_test(document, 30, 10) is not None


def test_hit_test_requires_pointer_over_hangul_glyph() -> None:
    line = make_line("\ud55cA", BoundingBox(0, 0, 20, 20), 0.9)
    document = OcrDocument((line,), 1.0)

    assert hit_test(document, 5, 10) is not None
    assert hit_test(document, 15, 10) is None
