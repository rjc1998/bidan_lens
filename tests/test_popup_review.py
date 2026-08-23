from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.locked_corpus import CorpusError
from benchmarks.popup_review import (
    PopupReviewDecision,
    audit_popup_review,
    inspection_popup_cases,
    load_popup_review,
    record_popup_decision,
    structural_popup_view,
    write_popup_review,
)
from bidan_lens.models import AnalysisCandidate, LexicalComponent, MorphemeExplanation


def test_popup_review_round_trip_contains_only_categorical_data(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    decisions = (
        PopupReviewDecision(
            'dev-plain-0017',
            'component_role',
            'annotation_convention_difference',
        ),
    )

    write_popup_review(path, 'corpus-v5', decisions)

    assert load_popup_review(path, 'corpus-v5') == decisions
    value = json.loads(path.read_text(encoding='utf-8'))
    assert set(value) == {
        'schema_version',
        'corpus_id',
        'review_kind',
        'decisions',
        'summary',
    }
    assert set(value['decisions'][0]) == {
        'sample_id',
        'failure_stage',
        'decision',
    }
    assert 'sentence' not in path.read_text(encoding='utf-8')


def test_record_popup_decision_uses_current_failure_stage() -> None:
    cases = (
        SimpleNamespace(
            sample=SimpleNamespace(sample_id='dev-plain-0075'),
            failure_stage='grammar_roles',
        ),
    )
    existing = (
        PopupReviewDecision(
            'dev-plain-0003',
            'component_role',
            'kiwi_analysis_error',
        ),
    )

    assert record_popup_decision(  # type: ignore[arg-type]
        cases,
        existing,
        'dev-plain-0075',
        'corpus_oracle_defect',
    ) == (
        existing[0],
        PopupReviewDecision(
            'dev-plain-0075',
            'grammar_roles',
            'corpus_oracle_defect',
        ),
    )
def test_popup_review_rejects_unknown_decision(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match='unknown decision'):
        write_popup_review(
            tmp_path / 'review.json',
            'corpus-v5',
            (PopupReviewDecision('dev-plain-0017', 'component_role', 'free text'),),
        )


def test_popup_review_rejects_extra_persisted_fields(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'corpus_id': 'corpus-v5',
                'review_kind': 'first_popup_analysis',
                'decisions': [],
                'summary': {},
                'sentence': 'must not be stored',
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(CorpusError, match='do not match'):
        load_popup_review(path, 'corpus-v5')


def test_popup_review_rejects_non_categorical_summary(tmp_path: Path) -> None:
    path = tmp_path / 'review.json'
    write_popup_review(
        path,
        'corpus-v5',
        (
            PopupReviewDecision(
                'dev-plain-0017',
                'component_role',
                'annotation_convention_difference',
            ),
        ),
    )
    value = json.loads(path.read_text(encoding='utf-8'))
    value['summary'] = {'sentence': 'must not be stored'}
    path.write_text(json.dumps(value), encoding='utf-8')

    with pytest.raises(CorpusError, match='invalid summary'):
        load_popup_review(path, 'corpus-v5')


def test_popup_review_audit_flags_stale_failure_stage() -> None:
    class Sample:
        sample_id = 'dev-plain-0017'

    class Case:
        sample = Sample()
        failure_stage = 'component_count'

    audit = audit_popup_review(
        (Case(),),  # type: ignore[arg-type]
        (
            PopupReviewDecision(
                'dev-plain-0017',
                'component_role',
                'annotation_convention_difference',
            ),
        ),
    )

    assert audit['complete'] is False
    assert audit['stale_sample_ids'] == ['dev-plain-0017']


def test_popup_review_inspection_filters_current_stage_and_decision() -> None:
    def case(sample_id: str, failure_stage: str):
        return SimpleNamespace(
            sample=SimpleNamespace(sample_id=sample_id),
            failure_stage=failure_stage,
        )

    cases = (case('lemma', 'primary_lemma'), case('role', 'component_role'))
    decisions = (
        PopupReviewDecision('lemma', 'primary_lemma', 'kiwi_analysis_error'),
        PopupReviewDecision('role', 'component_role', 'ambiguous_korean'),
    )

    selected = inspection_popup_cases(
        cases,  # type: ignore[arg-type]
        decisions,
        'primary_lemma',
        'kiwi_analysis_error',
    )

    assert [item.sample.sample_id for item in selected] == ['lemma']


def test_popup_review_structural_view_omits_analysis_text() -> None:
    expected_lemma = 'private expected lemma'
    actual_lemma = 'private actual lemma'
    expected_component = SimpleNamespace(
        surface='private expected surface',
        lemma=expected_lemma,
        learner_role='noun',
        entries=(),
    )
    candidate = AnalysisCandidate(
        surface='private actual surface',
        lemma=actual_lemma,
        score=1.25,
        morphemes=(
            MorphemeExplanation('private morpheme', actual_lemma, 'noun'),
        ),
        lexical_components=(
            LexicalComponent('private actual surface', actual_lemma, 'noun'),
        ),
    )
    case = SimpleNamespace(
        sample=SimpleNamespace(
            sample_id='dev-plain-0017',
            target=SimpleNamespace(
                text='private target',
                expected_lemma=expected_lemma,
                expected_labels=frozenset({'noun'}),
                expected_components=(expected_component,),
            ),
        ),
        failure_stage='primary_lemma',
        candidates=(candidate,),
    )

    value = structural_popup_view(  # type: ignore[arg-type]
        case,
        'kiwi_analysis_error',
    )
    serialized = json.dumps(value, ensure_ascii=True)

    assert 'private' not in serialized
    assert value['sample_id'] == 'dev-plain-0017'
    assert value['candidates'][0]['failure_stage'] == 'primary_lemma'  # type: ignore[index]
    assert value['candidates'][0]['components'][0]['learner_role'] == 'noun'  # type: ignore[index]
