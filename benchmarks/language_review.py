'''Local review of focused language disagreements.'''

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.locked_corpus import CorpusError, _lock_files, load_sources
from benchmarks.plain_evaluator import (
    LanguageSample,
    _asset,
    _language_failure_stage,
    load_language_samples,
    validate_plain_corpus,
)
from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.models import AnalysisCandidate

REVIEW_SCHEMA_VERSION = 1
REVIEW_DECISIONS = (
    'kiwi_analysis_error',
    'annotation_convention_difference',
    'equivalent_learner_interpretation',
    'corpus_oracle_defect',
    'ambiguous_korean',
)
REVIEW_FAILURE_STAGES = (
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
class LanguageReviewCase:
    sample: LanguageSample
    failure_stage: str
    candidates: tuple[AnalysisCandidate, ...]


@dataclass(frozen=True, slots=True)
class LanguageReviewDecision:
    sample_id: str
    failure_stage: str
    decision: str


def collect_language_review_cases(
    analyzer: KoreanAnalyzer,
    samples: tuple[LanguageSample, ...],
    language_class: str,
) -> tuple[LanguageReviewCase, ...]:
    cases: list[LanguageReviewCase] = []
    for sample in samples:
        if sample.target.language_class != language_class:
            continue
        candidates = analyzer.analyze(sample.sentence, sample.sentence_span)
        first = candidates[0] if candidates else None
        failure_stage = _language_failure_stage(first, sample.target)
        if failure_stage is not None:
            cases.append(LanguageReviewCase(sample, failure_stage, candidates))
    return tuple(cases)


def write_language_review(
    path: Path,
    corpus_id: str,
    language_class: str,
    decisions: tuple[LanguageReviewDecision, ...],
) -> None:
    ordered = tuple(sorted(decisions, key=lambda item: item.sample_id))
    if len({item.sample_id for item in ordered}) != len(ordered):
        raise CorpusError('language review contains duplicate sample ids')
    if any(item.decision not in REVIEW_DECISIONS for item in ordered):
        raise CorpusError('language review contains an unknown decision')
    if any(item.failure_stage not in REVIEW_FAILURE_STAGES for item in ordered):
        raise CorpusError('language review contains an unknown failure stage')
    value = {
        'schema_version': REVIEW_SCHEMA_VERSION,
        'corpus_id': corpus_id,
        'language_class': language_class,
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


def load_language_review(
    path: Path,
    corpus_id: str,
    language_class: str,
) -> tuple[LanguageReviewDecision, ...]:
    if not path.is_file():
        return ()
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError('cannot read language review decisions') from error
    if (
        not isinstance(value, dict)
        or set(value)
        != {'schema_version', 'corpus_id', 'language_class', 'decisions', 'summary'}
        or value.get('schema_version') != REVIEW_SCHEMA_VERSION
        or value.get('corpus_id') != corpus_id
        or value.get('language_class') != language_class
        or not isinstance(value.get('decisions'), list)
    ):
        raise CorpusError('language review decisions do not match this corpus')
    decisions: list[LanguageReviewDecision] = []
    for item in value['decisions']:
        if not isinstance(item, dict) or set(item) != {
            'sample_id',
            'failure_stage',
            'decision',
        }:
            raise CorpusError('language review contains an invalid decision')
        sample_id = item['sample_id']
        failure_stage = item['failure_stage']
        decision = item['decision']
        if (
            not isinstance(sample_id, str)
            or not isinstance(failure_stage, str)
            or failure_stage not in REVIEW_FAILURE_STAGES
            or decision not in REVIEW_DECISIONS
        ):
            raise CorpusError('language review contains an invalid decision')
        decisions.append(LanguageReviewDecision(sample_id, failure_stage, decision))
    if len({item.sample_id for item in decisions}) != len(decisions):
        raise CorpusError('language review contains duplicate sample ids')
    expected_summary = dict(
        sorted(Counter(item.decision for item in decisions).items())
    )
    if value.get('summary') != expected_summary:
        raise CorpusError('language review contains an invalid summary')
    return tuple(decisions)


def audit_language_review(
    cases: tuple[LanguageReviewCase, ...],
    decisions: tuple[LanguageReviewDecision, ...],
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


def _component_view(component: Any) -> dict[str, str]:
    return {
        'surface': component.surface,
        'lemma': component.lemma,
        'learner_role': component.learner_role,
    }


def _display_case(case: LanguageReviewCase, output: Callable[[str], None]) -> None:
    target = case.sample.target
    first = case.candidates[0] if case.candidates else None
    value = {
        'sample_id': case.sample.sample_id,
        'failure_stage': case.failure_stage,
        'sentence': case.sample.sentence,
        'target_span': list(case.sample.sentence_span),
        'expected': {
            'lemma': target.expected_lemma,
            'labels': sorted(target.expected_labels),
            'components': [_component_view(item) for item in target.expected_components],
        },
        'actual': None
        if first is None
        else {
            'lemma': first.lemma,
            'labels': sorted(
                {item.label for item in first.features}
                | {item.learner_label for item in first.morphemes}
            ),
            'components': [_component_view(item) for item in first.lexical_components],
        },
        'alternatives': [
            {
                'rank': rank,
                'lemma': candidate.lemma,
                'score': candidate.score,
                'components': [
                    _component_view(item) for item in candidate.lexical_components
                ],
            }
            for rank, candidate in enumerate(case.candidates[1:], start=2)
        ],
    }
    output(json.dumps(value, ensure_ascii=False, indent=2))


def review_language_cases(
    cases: tuple[LanguageReviewCase, ...],
    existing: tuple[LanguageReviewDecision, ...],
    *,
    prompt: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> tuple[LanguageReviewDecision, ...]:
    decisions = {item.sample_id: item for item in existing}
    choices = '\n'.join(
        f'  {index}: {decision}'
        for index, decision in enumerate(REVIEW_DECISIONS, start=1)
    )
    valid_answers = {str(index) for index in range(1, len(REVIEW_DECISIONS) + 1)}
    for case in cases:
        prior = decisions.get(case.sample.sample_id)
        if prior is not None and prior.failure_stage == case.failure_stage:
            continue
        _display_case(case, output)
        output(choices)
        while True:
            answer = prompt('decision [1-5]: ').strip()
            if answer in valid_answers:
                decision = REVIEW_DECISIONS[int(answer) - 1]
                break
            output('Choose a category from 1 through 5.')
        decisions[case.sample.sample_id] = LanguageReviewDecision(
            case.sample.sample_id,
            case.failure_stage,
            decision,
        )
    return tuple(decisions.values())


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='Review language disagreements without persisting corpus text'
    )
    parser.add_argument('assets', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('decisions', type=Path)
    parser.add_argument(
        '--language-class',
        choices=('multi-lexical', 'auxiliary'),
        default='multi-lexical',
    )
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
        '--failure-stage',
        choices=REVIEW_FAILURE_STAGES,
        help='with --inspect, display only one categorical failure stage',
    )
    arguments = parser.parse_args()
    if arguments.failure_stage and not arguments.inspect:
        parser.error('--failure-stage requires --inspect')

    validate_plain_corpus(arguments.corpus)
    corpus_id, locked = _lock_files(arguments.corpus)
    sources = load_sources(arguments.corpus, locked)
    samples = load_language_samples(arguments.corpus, locked, sources)
    dictionary = SqliteDictionaryStore(_asset(arguments.assets, 'dictionary.sqlite3'))
    analyzer = KoreanAnalyzer(dictionary)
    cases = collect_language_review_cases(analyzer, samples, arguments.language_class)
    existing = load_language_review(
        arguments.decisions, corpus_id, arguments.language_class
    )
    audit = audit_language_review(cases, existing)
    if arguments.audit:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        if not audit['complete']:
            raise SystemExit(1)
        return
    if arguments.inspect:
        reviewed = {item.sample_id: item.failure_stage for item in existing}
        for case in cases:
            if (
                reviewed.get(case.sample.sample_id) != case.failure_stage
                and (
                    arguments.failure_stage is None
                    or case.failure_stage == arguments.failure_stage
                )
            ):
                _display_case(case, print)
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return
    decisions = review_language_cases(cases, existing)
    write_language_review(
        arguments.decisions,
        corpus_id,
        arguments.language_class,
        decisions,
    )
    print(json.dumps(audit_language_review(cases, decisions), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
