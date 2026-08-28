from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmarks.context_review import (
    FULL_CONTEXT_REVIEW_KIND,
    ContextReviewDecision,
    audit_context_review,
    carry_forward_context_decisions,
    inspection_context_cases,
    load_context_review,
    record_context_decision,
    review_context_cases,
    structural_context_view,
    structural_segmentation_view,
    structural_target_geometry_view,
    write_context_review,
)
from benchmarks.locked_corpus import CorpusError
from bidan_lens.models import BoundingBox, OcrDocument
from bidan_lens.ocr.base import DetectedRegion, RecognizedText
from bidan_lens.ocr.hangul import make_line


def test_context_review_round_trip_contains_only_categorical_data(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    decisions = (
        ContextReviewDecision(
            'dev-plain-0017',
            'incorrect_line_sentence_reconstruction',
        ),
    )

    write_context_review(path, 'corpus-v4', decisions)

    assert load_context_review(path, 'corpus-v4') == decisions
    value = json.loads(path.read_text(encoding='utf-8'))
    assert set(value) == {
        'schema_version',
        'corpus_id',
        'review_kind',
        'decisions',
        'summary',
    }
    assert set(value['decisions'][0]) == {'sample_id', 'decision'}
    assert value['decisions'] == [
        {
            'sample_id': 'dev-plain-0017',
            'decision': 'incorrect_line_sentence_reconstruction',
        }
    ]


def test_full_context_review_is_separately_scoped(tmp_path: Path) -> None:
    path = tmp_path / 'full-review.json'
    decisions = (
        ContextReviewDecision(
            'dev-plain-0001',
            'incorrect_line_sentence_reconstruction',
        ),
    )

    write_context_review(
        path,
        'corpus-v4',
        decisions,
        FULL_CONTEXT_REVIEW_KIND,
    )

    assert load_context_review(
        path,
        'corpus-v4',
        FULL_CONTEXT_REVIEW_KIND,
    ) == decisions
    assert load_context_review(
        path,
        None,
        FULL_CONTEXT_REVIEW_KIND,
    ) == decisions
    with pytest.raises(CorpusError, match='do not match'):
        load_context_review(path, 'corpus-v4')
    value = json.loads(path.read_text(encoding='utf-8'))
    assert value['review_kind'] == FULL_CONTEXT_REVIEW_KIND
    assert set(value['decisions'][0]) == {'sample_id', 'decision'}


def test_context_review_records_one_active_case() -> None:
    cases = (
        SimpleNamespace(sample=SimpleNamespace(sample_id='active')),
    )
    existing = (
        ContextReviewDecision(
            'prior',
            'incorrect_line_sentence_reconstruction',
        ),
    )

    decisions = record_context_decision(  # type: ignore[arg-type]
        cases,
        existing,
        'active',
        'missed_or_merged_ocr_word_boundary',
    )

    assert decisions == (
        ContextReviewDecision(
            'active',
            'missed_or_merged_ocr_word_boundary',
        ),
        existing[0],
    )
    with pytest.raises(CorpusError, match='not an active failure'):
        record_context_decision(  # type: ignore[arg-type]
            cases,
            existing,
            'resolved',
            'ambiguous_layout',
        )


def test_context_review_accepts_non_target_transcription_category() -> None:
    cases = (
        SimpleNamespace(
            sample=SimpleNamespace(
                sample_id='transcription',
                render=SimpleNamespace(
                    renderer='browser',
                    layout='single-line',
                    punctuation='natural',
                    font='test-font',
                    size_px=12,
                ),
                target=SimpleNamespace(
                    sentence='',
                    text='',
                    sentence_span=(0, 0),
                ),
                lines=(),
            ),
            target=SimpleNamespace(
                sentence='',
                surface='',
                sentence_start=0,
                sentence_end=0,
            ),
            document=SimpleNamespace(lines=()),
        ),
    )

    decisions = review_context_cases(  # type: ignore[arg-type]
        cases,
        (),
        prompt=lambda _message: '2',
        output=lambda _message: None,
    )

    assert decisions == (
        ContextReviewDecision(
            'transcription',
            'non_target_ocr_transcription_error',
        ),
    )


def test_context_review_carry_forward_requires_every_current_id() -> None:
    cases = (
        SimpleNamespace(sample=SimpleNamespace(sample_id='active')),
    )
    prior = (
        ContextReviewDecision(
            'active',
            'incorrect_line_sentence_reconstruction',
        ),
        ContextReviewDecision(
            'resolved',
            'missed_or_merged_ocr_word_boundary',
        ),
    )

    assert carry_forward_context_decisions(  # type: ignore[arg-type]
        cases, prior
    ) == (prior[0],)

    with pytest.raises(CorpusError, match='current cases changed'):
        carry_forward_context_decisions(cases, ())  # type: ignore[arg-type]


def test_context_review_rejects_unknown_decision(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match='unknown decision'):
        write_context_review(
            tmp_path / 'review.json',
            'corpus-v4',
            (ContextReviewDecision('dev-plain-0017', 'free-form text'),),
        )


def test_context_review_rejects_extra_persisted_fields(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'corpus_id': 'corpus-v4',
                'review_kind': 'functional_context',
                'decisions': [],
                'summary': {},
                'sentence': 'must not be stored',
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(CorpusError, match='do not match'):
        load_context_review(path, 'corpus-v4')


def test_context_review_rejects_non_categorical_summary(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    write_context_review(
        path,
        'corpus-v4',
        (
            ContextReviewDecision(
                'dev-plain-0017',
                'incorrect_line_sentence_reconstruction',
            ),
        ),
    )
    value = json.loads(path.read_text(encoding='utf-8'))
    value['summary'] = {'sentence': 'must not be stored'}
    path.write_text(json.dumps(value), encoding='utf-8')

    with pytest.raises(CorpusError, match='invalid summary'):
        load_context_review(path, 'corpus-v4')


def test_context_review_audit_preserves_resolved_decisions() -> None:
    audit = audit_context_review(
        (),
        (
            ContextReviewDecision(
                'dev-plain-0017',
                'missed_or_merged_ocr_word_boundary',
            ),
        ),
    )

    assert audit['complete'] is True
    assert audit['missing_sample_ids'] == []
    assert audit['resolved_sample_ids'] == ['dev-plain-0017']


def test_context_review_inspection_filters_current_reviewed_category() -> None:
    class Sample:
        def __init__(self, sample_id: str) -> None:
            self.sample_id = sample_id

    class Case:
        def __init__(self, sample_id: str) -> None:
            self.sample = Sample(sample_id)

    cases = (Case('active-line'), Case('active-punctuation'))
    decisions = (
        ContextReviewDecision(
            'active-line',
            'incorrect_line_sentence_reconstruction',
        ),
        ContextReviewDecision(
            'active-punctuation',
            'punctuation_or_structured_ascii_handling',
        ),
        ContextReviewDecision(
            'resolved-line',
            'incorrect_line_sentence_reconstruction',
        ),
    )

    selected = inspection_context_cases(
        cases,  # type: ignore[arg-type]
        decisions,
        'incorrect_line_sentence_reconstruction',
    )

    assert [case.sample.sample_id for case in selected] == ['active-line']

    selected = inspection_context_cases(
        cases,  # type: ignore[arg-type]
        decisions,
        'incorrect_line_sentence_reconstruction',
        ('active-line',),
    )

    assert [case.sample.sample_id for case in selected] == ['active-line']

    selected = inspection_context_cases(
        cases,  # type: ignore[arg-type]
        decisions,
        None,
        ('active-punctuation', 'active-line'),
    )

    assert [case.sample.sample_id for case in selected] == [
        'active-line',
        'active-punctuation',
    ]


def test_context_review_structural_view_omits_sentence_and_target_text() -> None:
    expected_sentence = 'private expected sentence'
    actual_sentence = 'private actual sentence'
    expected_line = SimpleNamespace(
        text=expected_sentence,
        box=BoundingBox(10, 20, 210, 45),
        eojeols=(
            SimpleNamespace(text='one', box=BoundingBox(10, 20, 50, 45)),
            SimpleNamespace(text='two', box=BoundingBox(60, 20, 100, 45)),
            SimpleNamespace(text='three', box=BoundingBox(110, 20, 160, 45)),
        ),
    )
    actual_line = SimpleNamespace(
        text=actual_sentence,
        box=BoundingBox(12, 20, 190, 45),
        eojeols=(
            SimpleNamespace(
                text='one',
                box=BoundingBox(12, 20, 48, 45),
                confidence=0.98765,
                sentence_start=0,
                sentence_end=3,
            ),
            SimpleNamespace(
                text='two',
                box=BoundingBox(55, 20, 91, 45),
                confidence=0.9,
                sentence_start=4,
                sentence_end=7,
            ),
            SimpleNamespace(
                text='three',
                box=BoundingBox(98, 20, 150, 45),
                confidence=0.8,
                sentence_start=8,
                sentence_end=13,
            ),
        ),
    )
    case = SimpleNamespace(
        sample=SimpleNamespace(
            sample_id='dev-plain-0017',
            lines=(expected_line,),
            target=SimpleNamespace(
                sentence=expected_sentence,
                sentence_span=(8, 16),
            ),
            render=SimpleNamespace(
                renderer='pillow',
                layout='single-line',
                punctuation='plain',
                font='test-font',
                size_px=24,
            ),
        ),
        document=SimpleNamespace(lines=(actual_line,)),
        target=SimpleNamespace(
            sentence=actual_sentence,
            sentence_start=8,
            sentence_end=14,
        ),
    )

    value = structural_context_view(case)  # type: ignore[arg-type]
    serialized = json.dumps(value, ensure_ascii=True)

    assert expected_sentence not in serialized
    assert actual_sentence not in serialized
    assert value['sample_id'] == 'dev-plain-0017'
    assert value['expected']['sentence_length'] == len(expected_sentence)  # type: ignore[index]
    assert value['actual']['sentence_length'] == len(actual_sentence)  # type: ignore[index]
    assert value['expected']['lines'][0] == {  # type: ignore[index]
        'index': 0,
        'length': len(expected_sentence),
        'eojeol_count': 3,
        'box': [10.0, 20.0, 210.0, 45.0],
        'eojeols': [
            {'length': 3, 'box': [10.0, 20.0, 50.0, 45.0]},
            {'length': 3, 'box': [60.0, 20.0, 100.0, 45.0]},
            {'length': 5, 'box': [110.0, 20.0, 160.0, 45.0]},
        ],
        'adjacency': [
            {
                'horizontal_gap': 10.0,
                'vertical_overlap_ratio': 1.0,
                'text_overlap_length': 0,
            },
            {
                'horizontal_gap': 10.0,
                'vertical_overlap_ratio': 1.0,
                'text_overlap_length': 0,
            },
        ],
    }
    assert value['actual']['lines'][0]['eojeols'][0] == {  # type: ignore[index]
        'length': 3,
        'box': [12.0, 20.0, 48.0, 45.0],
        'confidence': 0.9877,
        'span': [0, 3],
    }


def test_target_geometry_view_contains_hits_but_no_text() -> None:
    sentence = '\ud55c\uad6d\uc5b4 \uacf5\ubd80'
    line = make_line(sentence, BoundingBox(0, 0, 100, 20), 0.91)
    target_box = line.eojeols[0].box
    target_pointer = target_box.center
    sample = SimpleNamespace(
        sample_id='dev-plain-0092',
        lines=(line,),
        target=SimpleNamespace(
            text='\ud55c\uad6d\uc5b4',
            sentence=sentence,
            box=target_box,
            pointer=target_pointer,
        ),
        render=SimpleNamespace(
            renderer='browser',
            layout='single-line',
            punctuation='plain',
            font='test-font',
            size_px=24,
        ),
        negative_probes=(
            SimpleNamespace(kind='whitespace', pointer=target_pointer),
            SimpleNamespace(kind='blank', pointer=(-10.0, -10.0)),
        ),
    )
    document = OcrDocument((line,), 1.0)

    value = structural_target_geometry_view(sample, document)  # type: ignore[arg-type]
    serialized = json.dumps(value, ensure_ascii=False)

    assert sentence not in serialized
    assert '\ud55c\uad6d\uc5b4' not in serialized
    assert '\uacf5\ubd80' not in serialized
    assert value['sample_id'] == 'dev-plain-0092'
    assert value['target_match'] == {'surface': True, 'geometry': True}
    assert value['target_pointer_hit'] is not None
    assert value['negative_probes'][0]['hit'] is not None  # type: ignore[index]
    assert value['negative_probes'][1]['hit'] is None  # type: ignore[index]
    glyphs = value['document']['lines'][0]['eojeols'][0]['glyphs']  # type: ignore[index]
    assert glyphs
    assert all(set(glyph) == {'length', 'box', 'confidence', 'hangul_count'} for glyph in glyphs)


def test_segmentation_view_contains_geometry_but_no_text(tmp_path: Path) -> None:
    image = tmp_path / 'synthetic.png'
    Image.new('RGB', (100, 30)).save(image)

    class Detector:
        def detect(self, _image):
            return (DetectedRegion(BoundingBox(10, 5, 90, 25), 0.98765),)

    class Recognizer:
        def word_boxes(self, _image):
            return ((0, 30), (35, 80))

        def recognize(self, image):
            return RecognizedText("private", image.width / 100)

    engine = SimpleNamespace(detector=Detector(), recognizer=Recognizer())
    sample = SimpleNamespace(sample_id="dev-plain-0012", image=image)
    sample.target = SimpleNamespace(text="private")

    value = structural_segmentation_view(engine, sample)  # type: ignore[arg-type]

    assert value == {
        "sample_id": "dev-plain-0012",
        "image_size": [100, 30],
        "region_count": 1,
        "regions": [
            {
                "index": 0,
                "box": [10.0, 5.0, 90.0, 25.0],
                "confidence": 0.9877,
                "segment_count": 2,
                "segments": [
                    [10.0, 5.0, 40.0, 25.0],
                    [45.0, 5.0, 90.0, 25.0],
                ],
                "recognition": [
                    {
                        "length": 7,
                        "confidence": 0.3,
                        "space_count": 0,
                        "tokens": [
                            {
                                "length": 7,
                                "hangul_count": 0,
                                "ascii_alnum_count": 7,
                                "ascii_punctuation_count": 0,
                                "unicode_letter_count": 7,
                                "unicode_number_count": 0,
                                "unicode_punctuation_count": 0,
                                "unicode_symbol_count": 0,
                                "hangul_jamo_count": 0,
                                "cjk_ideograph_count": 0,
                                "latin_letter_count": 7,
                            }
                        ],
                        "target_evidence": {
                            "equals_expected_target": True,
                            "contains_expected_target": True,
                            "expected_target_occurrences": 1,
                            "expected_target_span": [0, 7],
                            "expected_target_prefix": [],
                            "expected_target_suffix": [],
                        },
                    },
                    {
                        "length": 7,
                        "confidence": 0.45,
                        "space_count": 0,
                        "tokens": [
                            {
                                "length": 7,
                                "hangul_count": 0,
                                "ascii_alnum_count": 7,
                                "ascii_punctuation_count": 0,
                                "unicode_letter_count": 7,
                                "unicode_number_count": 0,
                                "unicode_punctuation_count": 0,
                                "unicode_symbol_count": 0,
                                "hangul_jamo_count": 0,
                                "cjk_ideograph_count": 0,
                                "latin_letter_count": 7,
                            }
                        ],
                        "target_evidence": {
                            "equals_expected_target": True,
                            "contains_expected_target": True,
                            "expected_target_occurrences": 1,
                            "expected_target_span": [0, 7],
                            "expected_target_prefix": [],
                            "expected_target_suffix": [],
                        },
                    },
                ],
                "overlap_triplets": [],
                "overlap_pairs": [],
                "close_pairs": [],
                "punctuation_retries": [],
                "segment_gaps": [5.0],
            }
        ],
    }
    assert "private" not in json.dumps(value, ensure_ascii=False)
