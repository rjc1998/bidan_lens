from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.language_review import (
    LanguageReviewDecision,
    audit_language_review,
    load_language_review,
    write_language_review,
)
from benchmarks.locked_corpus import CorpusError


def test_language_review_round_trip_contains_only_categorical_data(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    decisions = (
        LanguageReviewDecision(
            'language-17', 'primary_lemma', 'annotation_convention_difference'
        ),
    )

    write_language_review(path, 'corpus-v4', 'multi-lexical', decisions)

    assert load_language_review(path, 'corpus-v4', 'multi-lexical') == decisions
    value = json.loads(path.read_text(encoding='utf-8'))
    assert set(value) == {
        'schema_version',
        'corpus_id',
        'language_class',
        'decisions',
        'summary',
    }
    assert set(value['decisions'][0]) == {
        'sample_id',
        'failure_stage',
        'decision',
    }


def test_language_review_rejects_unknown_decision(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match='unknown decision'):
        write_language_review(
            tmp_path / 'review.json',
            'corpus-v4',
            'multi-lexical',
            (LanguageReviewDecision('language-17', 'primary_lemma', 'free-form'),),
        )


def test_language_review_rejects_unknown_failure_stage(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match='unknown failure stage'):
        write_language_review(
            tmp_path / 'review.json',
            'corpus-v4',
            'multi-lexical',
            (
                LanguageReviewDecision(
                    'language-17', 'sentence text must not be stored', 'kiwi_analysis_error'
                ),
            ),
        )


def test_language_review_rejects_non_categorical_summary(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    write_language_review(
        path,
        'corpus-v4',
        'multi-lexical',
        (
            LanguageReviewDecision(
                'language-17',
                'component_role',
                'annotation_convention_difference',
            ),
        ),
    )
    value = json.loads(path.read_text(encoding='utf-8'))
    value['summary'] = {'sentence': 'must not be stored'}
    path.write_text(json.dumps(value), encoding='utf-8')

    with pytest.raises(CorpusError, match='invalid summary'):
        load_language_review(path, 'corpus-v4', 'multi-lexical')


def test_language_review_audit_detects_stale_decision() -> None:
    class Sample:
        sample_id = 'language-17'

    class Case:
        sample = Sample()
        failure_stage = 'primary_lemma'

    audit = audit_language_review(
        (Case(),),  # type: ignore[arg-type]
        (
            LanguageReviewDecision(
                'language-17', 'component_role', 'kiwi_analysis_error'
            ),
        ),
    )

    assert audit['complete'] is False
    assert audit['missing_sample_ids'] == []
    assert audit['resolved_sample_ids'] == []
    assert audit['stale_sample_ids'] == ['language-17']


def test_language_review_audit_preserves_resolved_decisions() -> None:
    audit = audit_language_review(
        (),
        (
            LanguageReviewDecision(
                'language-17', 'component_surface', 'kiwi_analysis_error'
            ),
        ),
    )

    assert audit['complete'] is True
    assert audit['resolved_sample_ids'] == ['language-17']
    assert audit['stale_sample_ids'] == []
