'''Local review of rendered first-popup analysis disagreements.'''

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from benchmarks.locked_corpus import CorpusError, _lock_files, load_sources
from benchmarks.plain_evaluator import (
    PlainEngineLike,
    PlainSample,
    _asset,
    _functional_context,
    _language_failure_stage,
    _match,
    _normal,
    _quick_annotations,
    load_plain_samples,
    validate_plain_corpus,
)
from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.models import AnalysisCandidate, HoverTarget
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer
from bidan_lens.pipeline.hit_test import hit_test

REVIEW_SCHEMA_VERSION = 1
POPUP_REVIEW_KIND = 'first_popup_analysis'
POPUP_REVIEW_DECISIONS = (
    'kiwi_analysis_error',
    'annotation_convention_difference',
    'equivalent_learner_interpretation',
    'corpus_oracle_defect',
    'ambiguous_korean',
)
POPUP_FAILURE_STAGES = (
    'missing_candidate',
    'primary_lemma',
    'grammar_roles',
    'component_count',
    'component_surface',
    'component_lemma',
    'component_role',
    'contextual_dictionary_group',
    'primary_dictionary_group',
    'spacing',
)


@dataclass(frozen=True, slots=True)
class PopupReviewCase:
    sample: PlainSample
    target: HoverTarget
    failure_stage: str
    candidates: tuple[AnalysisCandidate, ...]


@dataclass(frozen=True, slots=True)
class PopupReviewDecision:
    sample_id: str
    failure_stage: str
    decision: str


def collect_popup_review_cases(
    engine: PlainEngineLike,
    analyzer: KoreanAnalyzer,
    samples: tuple[PlainSample, ...],
) -> tuple[PopupReviewCase, ...]:
    cases: list[PopupReviewCase] = []
    for sample in samples:
        with Image.open(sample.image) as source:
            document = engine.recognize(source.convert('RGB'))
        target = hit_test(document, *sample.target.pointer)
        target_hit = bool(
            target
            and _normal(target.surface) == sample.target.text
            and _match(sample.target.box, [(target.surface, target.box)]) is not None
        )
        if target is None or not target_hit or not _functional_context(
            target.sentence,
            (target.sentence_start, target.sentence_end),
            sample.target.sentence,
            sample.target.sentence_span,
        ):
            continue
        candidates = analyzer.analyze(
            target.sentence,
            (target.sentence_start, target.sentence_end),
        )
        failure_stage = _language_failure_stage(
            candidates[0] if candidates else None,
            sample.target,
        )
        if failure_stage is not None:
            cases.append(PopupReviewCase(sample, target, failure_stage, candidates))
    return tuple(cases)


def write_popup_review(
    path: Path,
    corpus_id: str,
    decisions: tuple[PopupReviewDecision, ...],
) -> None:
    ordered = tuple(sorted(decisions, key=lambda item: item.sample_id))
    if len({item.sample_id for item in ordered}) != len(ordered):
        raise CorpusError('popup review contains duplicate sample ids')
    if any(item.decision not in POPUP_REVIEW_DECISIONS for item in ordered):
        raise CorpusError('popup review contains an unknown decision')
    if any(item.failure_stage not in POPUP_FAILURE_STAGES for item in ordered):
        raise CorpusError('popup review contains an unknown failure stage')
    value = {
        'schema_version': REVIEW_SCHEMA_VERSION,
        'corpus_id': corpus_id,
        'review_kind': POPUP_REVIEW_KIND,
        'decisions': [
            {
                'sample_id': item.sample_id,
                'failure_stage': item.failure_stage,
                'decision': item.decision,
            }
            for item in ordered
        ],
        'summary': dict(sorted(Counter(item.decision for item in ordered).items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    temporary.replace(path)


def load_popup_review(
    path: Path,
    corpus_id: str,
) -> tuple[PopupReviewDecision, ...]:
    if not path.is_file():
        return ()
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError('cannot read popup review decisions') from error
    if (
        not isinstance(value, dict)
        or set(value)
        != {'schema_version', 'corpus_id', 'review_kind', 'decisions', 'summary'}
        or value.get('schema_version') != REVIEW_SCHEMA_VERSION
        or value.get('corpus_id') != corpus_id
        or value.get('review_kind') != POPUP_REVIEW_KIND
        or not isinstance(value.get('decisions'), list)
    ):
        raise CorpusError('popup review decisions do not match this corpus')
    decisions: list[PopupReviewDecision] = []
    for item in value['decisions']:
        if not isinstance(item, dict) or set(item) != {
            'sample_id',
            'failure_stage',
            'decision',
        }:
            raise CorpusError('popup review contains an invalid decision')
        sample_id = item['sample_id']
        failure_stage = item['failure_stage']
        decision = item['decision']
        if (
            not isinstance(sample_id, str)
            or failure_stage not in POPUP_FAILURE_STAGES
            or decision not in POPUP_REVIEW_DECISIONS
        ):
            raise CorpusError('popup review contains an invalid decision')
        decisions.append(PopupReviewDecision(sample_id, failure_stage, decision))
    if len({item.sample_id for item in decisions}) != len(decisions):
        raise CorpusError('popup review contains duplicate sample ids')
    expected_summary = dict(
        sorted(Counter(item.decision for item in decisions).items())
    )
    if value.get('summary') != expected_summary:
        raise CorpusError('popup review contains an invalid summary')
    return tuple(decisions)


def audit_popup_review(
    cases: tuple[PopupReviewCase, ...],
    decisions: tuple[PopupReviewDecision, ...],
) -> dict[str, object]:
    expected = {item.sample.sample_id: item.failure_stage for item in cases}
    actual = {item.sample_id: item for item in decisions}
    resolved = sorted(set(actual) - set(expected))
    stale = sorted(
        sample_id
        for sample_id, item in actual.items()
        if sample_id in expected and expected[sample_id] != item.failure_stage
    )
    missing = sorted(set(expected) - set(actual))
    return {
        'cases': len(expected),
        'decisions': len(actual),
        'complete': not missing and not stale,
        'missing_sample_ids': missing,
        'resolved_sample_ids': resolved,
        'stale_sample_ids': stale,
        'failure_stages': dict(sorted(Counter(expected.values()).items())),
        'review_decisions': dict(
            sorted(Counter(item.decision for item in actual.values()).items())
        ),
    }


def inspection_popup_cases(
    cases: tuple[PopupReviewCase, ...],
    decisions: tuple[PopupReviewDecision, ...],
    failure_stage: str | None = None,
    decision: str | None = None,
) -> tuple[PopupReviewCase, ...]:
    reviewed = {item.sample_id: item for item in decisions}
    if failure_stage is None and decision is None:
        return tuple(
            case
            for case in cases
            if (prior := reviewed.get(case.sample.sample_id)) is None
            or prior.failure_stage != case.failure_stage
        )
    return tuple(
        case
        for case in cases
        if (failure_stage is None or case.failure_stage == failure_stage)
        and (
            decision is None
            or (
                (prior := reviewed.get(case.sample.sample_id)) is not None
                and prior.failure_stage == case.failure_stage
                and prior.decision == decision
            )
        )
    )


def record_popup_decision(
    cases: tuple[PopupReviewCase, ...],
    decisions: tuple[PopupReviewDecision, ...],
    sample_id: str,
    decision: str,
) -> tuple[PopupReviewDecision, ...]:
    current = next(
        (case for case in cases if case.sample.sample_id == sample_id),
        None,
    )
    if current is None:
        raise CorpusError(f'popup review sample is not an active failure: {sample_id}')
    updated = {item.sample_id: item for item in decisions}
    updated[sample_id] = PopupReviewDecision(
        sample_id,
        current.failure_stage,
        decision,
    )
    return tuple(sorted(updated.values(), key=lambda item: item.sample_id))


def _component_view(component: Any) -> dict[str, str]:
    return {
        'surface': component.surface,
        'lemma': component.lemma,
        'learner_role': component.learner_role,
    }


def _component_structure(
    component: Any,
    expected: Any | None = None,
) -> dict[str, object]:
    entries = getattr(component, 'dictionary_entries', None)
    if entries is None:
        entries = getattr(component, 'entries', ())
    return {
        'surface_length': len(component.surface),
        'lemma_length': len(component.lemma),
        'learner_role': component.learner_role,
        'dictionary_entry_count': len(entries),
        'expected_surface_match': bool(
            expected is not None and component.surface == expected.surface
        ),
        'expected_lemma_match': bool(
            expected is not None and component.lemma == expected.lemma
        ),
        'expected_role_match': bool(
            expected is not None
            and component.learner_role == expected.learner_role
        ),
    }


def _candidate_view(candidate: AnalysisCandidate, rank: int) -> dict[str, object]:
    return {
        'rank': rank,
        'lemma': candidate.lemma,
        'score': candidate.score,
        'labels': sorted(
            {item.label for item in candidate.features}
            | {item.learner_label for item in candidate.morphemes}
        ),
        'components': [_component_view(item) for item in candidate.lexical_components],
    }


def _expected_structure(expected: Any) -> dict[str, object]:
    return {
        'target_length': len(expected.text),
        'lemma_length': len(expected.expected_lemma),
        'labels': sorted(expected.expected_labels),
        'components': [
            _component_structure(component)
            for component in expected.expected_components
        ],
    }


def _candidate_structure(
    candidate: AnalysisCandidate,
    rank: int,
    expected: Any,
) -> dict[str, object]:
    expected_components = expected.expected_components
    return {
        'rank': rank,
        'failure_stage': _language_failure_stage(candidate, expected),
        'score': round(candidate.score, 4),
        'surface_length': len(candidate.surface),
        'lemma_length': len(candidate.lemma),
        'expected_lemma_match': candidate.lemma == expected.expected_lemma,
        'labels': sorted(
            {item.label for item in candidate.features}
            | {item.learner_label for item in candidate.morphemes}
        ),
        'components': [
            _component_structure(
                component,
                expected_components[index]
                if index < len(expected_components)
                else None,
            )
            for index, component in enumerate(candidate.lexical_components)
        ],
        'morphemes': [
            {
                'surface_length': len(item.surface),
                'lemma_length': len(item.lemma),
                'learner_label': item.learner_label,
            }
            for item in candidate.morphemes
        ],
        'dictionary_entry_count': len(candidate.dictionary_entries),
        'uncertain': candidate.uncertain,
        'has_interpreted_surface': candidate.interpreted_surface is not None,
    }


def structural_popup_view(
    case: PopupReviewCase,
    decision: str | None = None,
) -> dict[str, object]:
    expected = case.sample.target
    return {
        'sample_id': case.sample.sample_id,
        'failure_stage': case.failure_stage,
        'review_decision': decision,
        'expected': _expected_structure(expected),
        'candidates': [
            _candidate_structure(candidate, rank, expected)
            for rank, candidate in enumerate(case.candidates, start=1)
        ],
    }


def _display_case(case: PopupReviewCase, output: Callable[[str], None]) -> None:
    expected = case.sample.target
    value = {
        'sample_id': case.sample.sample_id,
        'failure_stage': case.failure_stage,
        'expected_sentence': expected.sentence,
        'actual_sentence': case.target.sentence,
        'expected_span': list(expected.sentence_span),
        'actual_span': [case.target.sentence_start, case.target.sentence_end],
        'expected': {
            'lemma': expected.expected_lemma,
            'labels': sorted(expected.expected_labels),
            'components': [_component_view(item) for item in expected.expected_components],
        },
        'candidates': [
            _candidate_view(candidate, rank)
            for rank, candidate in enumerate(case.candidates, start=1)
        ],
    }
    output(json.dumps(value, ensure_ascii=False, indent=2))


def _display_compact_case(
    case: PopupReviewCase,
    output: Callable[[str], None],
) -> None:
    expected = case.sample.target
    first = case.candidates[0] if case.candidates else None
    value = {
        'sample_id': case.sample.sample_id,
        'failure_stage': case.failure_stage,
        'sentence': case.target.sentence,
        'span': [case.target.sentence_start, case.target.sentence_end],
        'expected': {
            'lemma': expected.expected_lemma,
            'labels': sorted(expected.expected_labels),
            'components': [_component_view(item) for item in expected.expected_components],
        },
        'actual': None if first is None else _candidate_view(first, 1),
        'alternatives': [
            _candidate_view(candidate, rank)
            for rank, candidate in enumerate(case.candidates[1:], start=2)
        ],
    }
    output(json.dumps(value, ensure_ascii=False))


def review_popup_cases(
    cases: tuple[PopupReviewCase, ...],
    existing: tuple[PopupReviewDecision, ...],
    *,
    prompt: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> tuple[PopupReviewDecision, ...]:
    decisions = {item.sample_id: item for item in existing}
    choices = '\n'.join(
        f'  {index}: {decision}'
        for index, decision in enumerate(POPUP_REVIEW_DECISIONS, start=1)
    )
    valid_answers = {
        str(index) for index in range(1, len(POPUP_REVIEW_DECISIONS) + 1)
    }
    for case in cases:
        prior = decisions.get(case.sample.sample_id)
        if prior is not None and prior.failure_stage == case.failure_stage:
            continue
        _display_case(case, output)
        output(choices)
        while True:
            answer = prompt('decision [1-5]: ').strip()
            if answer in valid_answers:
                decision = POPUP_REVIEW_DECISIONS[int(answer) - 1]
                break
            output('Choose a category from 1 through 5.')
        decisions[case.sample.sample_id] = PopupReviewDecision(
            case.sample.sample_id,
            case.failure_stage,
            decision,
        )
    return tuple(decisions.values())


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='Review first-popup analysis failures without persisting corpus text'
    )
    parser.add_argument('assets', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('decisions', type=Path)
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='display unresolved cases locally without writing a report',
    )
    parser.add_argument(
        '--audit',
        action='store_true',
        help='validate an existing decision report without prompting',
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='show a compact local-only case view with --inspect',
    )
    parser.add_argument(
        '--failure-stage',
        choices=POPUP_FAILURE_STAGES,
        help='with --inspect, display only one categorical failure stage',
    )
    parser.add_argument(
        '--decision',
        choices=POPUP_REVIEW_DECISIONS,
        help='with --inspect, display current cases in one reviewed category',
    )
    parser.add_argument(
        '--structure-only',
        action='store_true',
        help='with --inspect, omit all corpus and analysis text',
    )
    parser.add_argument(
        '--sample-id',
        help='stable sample ID for --record-decision',
    )
    parser.add_argument(
        '--record-decision',
        choices=POPUP_REVIEW_DECISIONS,
        help='record one categorical decision without displaying corpus text',
    )
    arguments = parser.parse_args()
    if (
        arguments.compact
        or arguments.failure_stage
        or arguments.decision
        or arguments.structure_only
    ) and not arguments.inspect:
        parser.error('inspection filters and views require --inspect')
    if arguments.compact and arguments.structure_only:
        parser.error('--compact and --structure-only cannot be combined')
    if bool(arguments.sample_id) != bool(arguments.record_decision):
        parser.error('--sample-id and --record-decision must be used together')
    if arguments.record_decision and arguments.inspect:
        parser.error('--record-decision cannot be combined with --inspect')

    validate_plain_corpus(arguments.corpus)
    corpus_id, locked = _lock_files(arguments.corpus)
    sources = load_sources(arguments.corpus, locked)
    selected = _quick_annotations(arguments.corpus, locked)
    samples = load_plain_samples(arguments.corpus, 'plain', locked, sources, selected)
    dictionary = SqliteDictionaryStore(_asset(arguments.assets, 'dictionary.sqlite3'))
    analyzer = KoreanAnalyzer(dictionary)
    engine = PaddleOcrEngine(
        PaddleDetector(_asset(arguments.assets, 'korean_detection.onnx')),
        PaddleRecognizer(
            _asset(arguments.assets, 'korean_recognition.onnx'),
            _asset(arguments.assets, 'korean_characters.txt'),
        ),
    )
    cases = collect_popup_review_cases(engine, analyzer, samples)
    existing = load_popup_review(arguments.decisions, corpus_id)
    audit = audit_popup_review(cases, existing)
    if arguments.record_decision:
        decisions = record_popup_decision(
            cases,
            existing,
            arguments.sample_id,
            arguments.record_decision,
        )
        write_popup_review(arguments.decisions, corpus_id, decisions)
        print(json.dumps(audit_popup_review(cases, decisions), ensure_ascii=True, indent=2))
        return
    if arguments.audit:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        if not audit['complete']:
            raise SystemExit(1)
        return
    if arguments.inspect:
        reviewed = {item.sample_id: item for item in existing}
        selected_cases = inspection_popup_cases(
            cases,
            existing,
            arguments.failure_stage,
            arguments.decision,
        )
        for case in selected_cases:
            if arguments.structure_only:
                prior = reviewed.get(case.sample.sample_id)
                decision = None if prior is None else prior.decision
                print(
                    json.dumps(
                        structural_popup_view(case, decision),
                        ensure_ascii=True,
                    )
                )
            elif arguments.compact:
                _display_compact_case(case, print)
            else:
                _display_case(case, print)
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return
    decisions = review_popup_cases(cases, existing)
    write_popup_review(arguments.decisions, corpus_id, decisions)
    print(json.dumps(audit_popup_review(cases, decisions), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
