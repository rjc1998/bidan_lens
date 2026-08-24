'''Local review of rendered functional-context disagreements.'''

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
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
    _match,
    _normal,
    _quick_annotations,
    load_plain_samples,
    validate_plain_corpus,
)
from bidan_lens.models import HoverTarget, OcrDocument
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer
from bidan_lens.pipeline.hit_test import hit_test

REVIEW_SCHEMA_VERSION = 1
CONTEXT_REVIEW_KIND = 'functional_context'
CONTEXT_REVIEW_DECISIONS = (
    'missed_or_merged_ocr_word_boundary',
    'incorrect_line_sentence_reconstruction',
    'punctuation_or_structured_ascii_handling',
    'incorrect_target_span',
    'ambiguous_layout',
)


@dataclass(frozen=True, slots=True)
class ContextReviewCase:
    sample: PlainSample
    document: OcrDocument
    target: HoverTarget


@dataclass(frozen=True, slots=True)
class ContextReviewDecision:
    sample_id: str
    decision: str


def collect_context_review_cases(
    engine: PlainEngineLike,
    samples: tuple[PlainSample, ...],
) -> tuple[ContextReviewCase, ...]:
    cases: list[ContextReviewCase] = []
    for sample in samples:
        with Image.open(sample.image) as source:
            document = engine.recognize(source.convert('RGB'))
        target = hit_test(document, *sample.target.pointer)
        target_hit = bool(
            target
            and _normal(target.surface) == sample.target.text
            and _match(sample.target.box, [(target.surface, target.box)]) is not None
        )
        if (
            target_hit
            and target is not None
            and not _functional_context(
                target.sentence,
                (target.sentence_start, target.sentence_end),
                sample.target.sentence,
                sample.target.sentence_span,
            )
        ):
            cases.append(ContextReviewCase(sample, document, target))
    return tuple(cases)


def write_context_review(
    path: Path,
    corpus_id: str,
    decisions: tuple[ContextReviewDecision, ...],
) -> None:
    ordered = tuple(sorted(decisions, key=lambda item: item.sample_id))
    if len({item.sample_id for item in ordered}) != len(ordered):
        raise CorpusError('context review contains duplicate sample ids')
    if any(item.decision not in CONTEXT_REVIEW_DECISIONS for item in ordered):
        raise CorpusError('context review contains an unknown decision')
    value = {
        'schema_version': REVIEW_SCHEMA_VERSION,
        'corpus_id': corpus_id,
        'review_kind': CONTEXT_REVIEW_KIND,
        'decisions': [
            {'sample_id': item.sample_id, 'decision': item.decision}
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


def load_context_review(
    path: Path,
    corpus_id: str,
) -> tuple[ContextReviewDecision, ...]:
    if not path.is_file():
        return ()
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError('cannot read context review decisions') from error
    if (
        not isinstance(value, dict)
        or set(value)
        != {'schema_version', 'corpus_id', 'review_kind', 'decisions', 'summary'}
        or value.get('schema_version') != REVIEW_SCHEMA_VERSION
        or value.get('corpus_id') != corpus_id
        or value.get('review_kind') != CONTEXT_REVIEW_KIND
        or not isinstance(value.get('decisions'), list)
    ):
        raise CorpusError('context review decisions do not match this corpus')
    decisions: list[ContextReviewDecision] = []
    for item in value['decisions']:
        if not isinstance(item, dict) or set(item) != {'sample_id', 'decision'}:
            raise CorpusError('context review contains an invalid decision')
        sample_id = item['sample_id']
        decision = item['decision']
        if not isinstance(sample_id, str) or decision not in CONTEXT_REVIEW_DECISIONS:
            raise CorpusError('context review contains an invalid decision')
        decisions.append(ContextReviewDecision(sample_id, decision))
    if len({item.sample_id for item in decisions}) != len(decisions):
        raise CorpusError('context review contains duplicate sample ids')
    expected_summary = dict(
        sorted(Counter(item.decision for item in decisions).items())
    )
    if value.get('summary') != expected_summary:
        raise CorpusError('context review contains an invalid summary')
    return tuple(decisions)


def audit_context_review(
    cases: tuple[ContextReviewCase, ...],
    decisions: tuple[ContextReviewDecision, ...],
) -> dict[str, object]:
    current = {item.sample.sample_id for item in cases}
    reviewed = {item.sample_id: item for item in decisions}
    missing = sorted(current - set(reviewed))
    resolved = sorted(set(reviewed) - current)
    return {
        'cases': len(current),
        'decisions': len(reviewed),
        'complete': not missing,
        'missing_sample_ids': missing,
        'resolved_sample_ids': resolved,
        'review_decisions': dict(
            sorted(Counter(item.decision for item in reviewed.values()).items())
        ),
    }


def inspection_context_cases(
    cases: tuple[ContextReviewCase, ...],
    decisions: tuple[ContextReviewDecision, ...],
    decision: str | None = None,
    sample_id: str | None = None,
) -> tuple[ContextReviewCase, ...]:
    reviewed = {item.sample_id: item.decision for item in decisions}
    if decision is None and sample_id is None:
        selected = tuple(
            case for case in cases if case.sample.sample_id not in reviewed
        )
    else:
        selected = tuple(
            case
            for case in cases
            if decision is None
            or reviewed.get(case.sample.sample_id) == decision
        )
    if sample_id is not None:
        selected = tuple(
            case for case in selected if case.sample.sample_id == sample_id
        )
    return selected


def _box_view(box: Any) -> list[float]:
    return [
        round(float(box.left), 2),
        round(float(box.top), 2),
        round(float(box.right), 2),
        round(float(box.bottom), 2),
    ]


def _common_prefix_length(left: str, right: str) -> int:
    return next(
        (
            index
            for index, pair in enumerate(zip(left, right, strict=False))
            if pair[0] != pair[1]
        ),
        min(len(left), len(right)),
    )


def _common_suffix_length(left: str, right: str) -> int:
    return _common_prefix_length(left[::-1], right[::-1])


def _line_structure(lines: Any, sentence: str) -> tuple[list[dict[str, object]], int | None]:
    values: list[dict[str, object]] = []
    selected_index: int | None = None
    for index, line in enumerate(lines):
        if selected_index is None and line.text == sentence:
            selected_index = index
        eojeols = list(line.eojeols)
        adjacency = []
        for left, right in zip(eojeols, eojeols[1:], strict=False):
            overlap_length = max(
                (
                    length
                    for length in range(
                        min(len(left.text), len(right.text)),
                        0,
                        -1,
                    )
                    if left.text[-length:] == right.text[:length]
                ),
                default=0,
            )
            vertical_overlap = max(
                0.0,
                min(left.box.bottom, right.box.bottom)
                - max(left.box.top, right.box.top),
            )
            adjacency.append(
                {
                    'horizontal_gap': round(float(right.box.left - left.box.right), 2),
                    'vertical_overlap_ratio': round(
                        vertical_overlap
                        / max(1.0, min(left.box.height, right.box.height)),
                        4,
                    ),
                    'text_overlap_length': overlap_length,
                }
            )
        values.append(
            {
                'index': index,
                'length': len(line.text),
                'eojeol_count': len(line.eojeols),
                'box': _box_view(line.box),
                'eojeols': [
                    {
                        'length': len(item.text),
                        'box': _box_view(item.box),
                        **(
                            {'confidence': round(float(item.confidence), 4)}
                            if hasattr(item, 'confidence')
                            else {}
                        ),
                        **(
                            {
                                'span': [
                                    int(item.sentence_start),
                                    int(item.sentence_end),
                                ]
                            }
                            if hasattr(item, 'sentence_start')
                            and hasattr(item, 'sentence_end')
                            else {}
                        ),
                    }
                    for item in eojeols
                ],
                'adjacency': adjacency,
            }
        )
    return values, selected_index


def _token_structure(sentence: str) -> list[dict[str, int]]:
    values = []
    for token in sentence.split():
        values.append(
            {
                'length': len(token),
                'hangul_count': sum('\uac00' <= item <= '\ud7a3' for item in token),
                'ascii_alnum_count': sum(
                    item.isascii() and item.isalnum() for item in token
                ),
                'ascii_punctuation_count': sum(
                    item.isascii() and not item.isalnum() for item in token
                ),
                'unicode_letter_count': sum(
                    unicodedata.category(item).startswith('L') for item in token
                ),
                'unicode_number_count': sum(
                    unicodedata.category(item).startswith('N') for item in token
                ),
                'unicode_punctuation_count': sum(
                    unicodedata.category(item).startswith('P') for item in token
                ),
                'unicode_symbol_count': sum(
                    unicodedata.category(item).startswith('S') for item in token
                ),
                'hangul_jamo_count': sum(
                    0x1100 <= ord(item) <= 0x11FF
                    or 0x3130 <= ord(item) <= 0x318F
                    or 0xA960 <= ord(item) <= 0xA97F
                    or 0xD7B0 <= ord(item) <= 0xD7FF
                    for item in token
                ),
                'cjk_ideograph_count': sum(
                    0x3400 <= ord(item) <= 0x4DBF
                    or 0x4E00 <= ord(item) <= 0x9FFF
                    or 0xF900 <= ord(item) <= 0xFAFF
                    for item in token
                ),
                'latin_letter_count': sum(
                    'LATIN' in unicodedata.name(item, '')
                    and unicodedata.category(item).startswith('L')
                    for item in token
                ),
            }
        )
    return values


def _sentence_structure(
    sentence: str,
    span: tuple[int, int],
    lines: Any,
) -> dict[str, object]:
    line_values, line_index = _line_structure(lines, sentence)
    return {
        'sentence_length': len(sentence),
        'eojeol_count': len(sentence.split()),
        'tokens': _token_structure(sentence),
        'target_span': list(span),
        'target_prefix_length': span[0],
        'target_suffix_length': len(sentence) - span[1],
        'target_line_index': line_index,
        'lines': line_values,
    }


def structural_context_view(case: ContextReviewCase) -> dict[str, object]:
    '''Return local inspection data without recognized or oracle text.'''
    sample = case.sample
    expected = sample.target.sentence
    actual = case.target.sentence
    expected_span = sample.target.sentence_span
    actual_span = (case.target.sentence_start, case.target.sentence_end)
    return {
        'sample_id': sample.sample_id,
        'render': {
            'renderer': sample.render.renderer,
            'layout': sample.render.layout,
            'punctuation': sample.render.punctuation,
            'font': sample.render.font,
            'size_px': sample.render.size_px,
        },
        'expected': _sentence_structure(expected, expected_span, sample.lines),
        'actual': _sentence_structure(actual, actual_span, case.document.lines),
        'comparison': {
            'sentence_length_delta': len(actual) - len(expected),
            'eojeol_count_delta': len(actual.split()) - len(expected.split()),
            'common_prefix_length': _common_prefix_length(expected, actual),
            'common_suffix_length': _common_suffix_length(expected, actual),
            'expected_contains_actual': actual in expected,
            'actual_contains_expected': expected in actual,
            'target_start_delta': actual_span[0] - expected_span[0],
            'target_end_delta': actual_span[1] - expected_span[1],
        },
    }


def _hover_target_structure(target: HoverTarget | None) -> dict[str, object] | None:
    if target is None:
        return None
    return {
        'surface_length': len(target.surface),
        'sentence_length': len(target.sentence),
        'span': [target.sentence_start, target.sentence_end],
        'box': _box_view(target.box),
        'confidence': round(float(target.confidence), 4),
    }


def structural_target_geometry_view(
    sample: PlainSample,
    document: OcrDocument,
) -> dict[str, object]:
    '''Return target and negative-probe geometry without OCR or oracle text.'''
    target = hit_test(document, *sample.target.pointer)
    expected_lines, _ = _line_structure(sample.lines, sample.target.sentence)
    actual_lines = []
    for index, line in enumerate(document.lines):
        actual_lines.append(
            {
                'index': index,
                'length': len(line.text),
                'box': _box_view(line.box),
                'confidence': round(float(line.confidence), 4),
                'eojeols': [
                    {
                        'length': len(eojeol.text),
                        'box': _box_view(eojeol.box),
                        'confidence': round(float(eojeol.confidence), 4),
                        'span': [eojeol.sentence_start, eojeol.sentence_end],
                        'glyphs': [
                            {
                                'length': len(glyph.text),
                                'box': _box_view(glyph.box),
                                'confidence': round(float(glyph.confidence), 4),
                                'hangul_count': sum(
                                    '\uac00' <= character <= '\ud7a3'
                                    for character in glyph.text
                                ),
                            }
                            for glyph in eojeol.glyphs
                        ],
                    }
                    for eojeol in line.eojeols
                ],
            }
        )
    return {
        'sample_id': sample.sample_id,
        'render': {
            'renderer': sample.render.renderer,
            'layout': sample.render.layout,
            'punctuation': sample.render.punctuation,
            'font': sample.render.font,
            'size_px': sample.render.size_px,
        },
        'expected': {
            'target_length': len(sample.target.text),
            'target_box': _box_view(sample.target.box),
            'target_pointer': [round(value, 2) for value in sample.target.pointer],
            'lines': expected_lines,
        },
        'target_pointer_hit': _hover_target_structure(target),
        'target_match': {
            'surface': bool(
                target and _normal(target.surface) == sample.target.text
            ),
            'geometry': bool(
                target
                and _match(sample.target.box, [(target.surface, target.box)])
                is not None
            ),
        },
        'negative_probes': [
            {
                'kind': probe.kind,
                'pointer': [round(value, 2) for value in probe.pointer],
                'hit': _hover_target_structure(
                    hit_test(document, *probe.pointer)
                ),
            }
            for probe in sample.negative_probes
        ],
        'document': {
            'origin': [document.origin_x, document.origin_y],
            'lines': actual_lines,
        },
    }


def structural_segmentation_view(
    engine: PaddleOcrEngine,
    sample: PlainSample,
) -> dict[str, object]:
    '''Return detector and segment geometry without OCR or oracle text.'''
    with Image.open(sample.image) as source:
        image = source.convert('RGB')
    regions = engine.detector.detect(image)
    values = []
    segmenter = getattr(engine.recognizer, 'word_boxes', None)
    for index, region in enumerate(regions):
        crop = image.crop(
            (
                math.floor(region.box.left),
                math.floor(region.box.top),
                math.ceil(region.box.right),
                math.ceil(region.box.bottom),
            )
        )
        segments = tuple(segmenter(crop)) if callable(segmenter) else ()
        boxes = [
            [
                round(float(region.box.left + left), 2),
                round(float(region.box.top), 2),
                round(float(region.box.left + right), 2),
                round(float(region.box.bottom), 2),
            ]
            for left, right in segments
        ]
        recognized_segments = []
        recognized_texts = []
        for left, right in segments:
            recognized = engine.recognizer.recognize(
                crop.crop((left, 0, right, crop.height))
            )
            recognized_texts.append(recognized.text.replace(' ', ''))
            recognized_segments.append(
                {
                    'length': len(recognized.text),
                    'confidence': round(float(recognized.confidence), 6),
                    'space_count': sum(item.isspace() for item in recognized.text),
                    'tokens': _token_structure(recognized.text),
                }
            )
        overlap_triplets = []
        for segment_index in range(len(segments) - 2):
            first, middle, last = segments[segment_index : segment_index + 3]
            if middle[0] - first[1] >= 0 or last[0] - middle[1] >= 0:
                continue
            combined = engine.recognizer.recognize(
                crop.crop((first[0], 0, last[1], crop.height))
            )
            combined_text = combined.text.replace(' ', '')
            first_text, middle_text, last_text = recognized_texts[
                segment_index : segment_index + 3
            ]
            overlap_triplets.append(
                {
                    'segment_indices': [
                        segment_index,
                        segment_index + 1,
                        segment_index + 2,
                    ],
                    'combined_length': len(combined_text),
                    'combined_confidence': round(float(combined.confidence), 6),
                    'equals_first_middle': combined_text == first_text + middle_text,
                    'equals_middle_last': combined_text == middle_text + last_text,
                    'equals_all': combined_text
                    == first_text + middle_text + last_text,
                    'tokens': _token_structure(combined_text),
                }
            )
        overlap_pairs = []
        for segment_index, (first, last) in enumerate(
            zip(segments, segments[1:], strict=False)
        ):
            if last[0] - first[1] >= 0:
                continue
            combined = engine.recognizer.recognize(
                crop.crop((first[0], 0, last[1], crop.height))
            )
            combined_text = combined.text.replace(' ', '')
            overlap_pairs.append(
                {
                    'segment_indices': [segment_index, segment_index + 1],
                    'overlap_ratio': round(
                        float((first[1] - last[0]) / region.box.height), 4
                    ),
                    'combined_length': len(combined_text),
                    'combined_confidence': round(float(combined.confidence), 6),
                    'equals_concatenation': combined_text
                    == recognized_texts[segment_index]
                    + recognized_texts[segment_index + 1],
                    'tokens': _token_structure(combined_text),
                }
            )
        close_pairs = []
        for segment_index, (first, last) in enumerate(
            zip(segments, segments[1:], strict=False)
        ):
            gap = last[0] - first[1]
            if not 0 <= gap <= region.box.height * 0.22:
                continue
            combined = engine.recognizer.recognize(
                crop.crop((first[0], 0, last[1], crop.height))
            )
            combined_text = combined.text.replace(' ', '')
            close_pairs.append(
                {
                    'segment_indices': [segment_index, segment_index + 1],
                    'gap_ratio': round(float(gap / region.box.height), 4),
                    'combined_length': len(combined_text),
                    'combined_confidence': round(float(combined.confidence), 6),
                    'equals_concatenation': combined_text
                    == recognized_texts[segment_index]
                    + recognized_texts[segment_index + 1],
                    'starts_with_first': combined_text.startswith(
                        recognized_texts[segment_index]
                    ),
                    'ends_with_last': combined_text.endswith(
                        recognized_texts[segment_index + 1]
                    ),
                    'added_suffix_punctuation': bool(
                        recognized_texts[segment_index]
                        and combined_text.startswith(recognized_texts[segment_index])
                        and len(combined_text)
                        == len(recognized_texts[segment_index]) + 1
                        and unicodedata.category(combined_text[-1]).startswith('P')
                    ),
                    'tokens': _token_structure(combined_text),
                }
            )
        punctuation_retries = []
        padding = max(2, round(region.box.height * 0.2))
        for segment_index, ((left, right), text) in enumerate(
            zip(segments, recognized_texts, strict=True)
        ):
            if not text or not all(
                unicodedata.category(character).startswith('P')
                for character in text
            ):
                continue
            padded = engine.recognizer.recognize(
                crop.crop(
                    (
                        max(0, left - padding),
                        0,
                        min(crop.width, right + padding),
                        crop.height,
                    )
                )
            )
            punctuation_retries.append(
                {
                    'segment_indices': [segment_index],
                    'padding': padding,
                    'length': len(padded.text),
                    'confidence': round(float(padded.confidence), 6),
                    'tokens': _token_structure(padded.text),
                }
            )
        values.append(
            {
                'index': index,
                'box': _box_view(region.box),
                'confidence': round(float(region.confidence), 4),
                'segment_count': len(segments),
                'segments': boxes,
                'recognition': recognized_segments,
                'overlap_triplets': overlap_triplets,
                'overlap_pairs': overlap_pairs,
                'close_pairs': close_pairs,
                'punctuation_retries': punctuation_retries,
                'segment_gaps': [
                    round(float(right[0] - left[2]), 2)
                    for left, right in zip(boxes, boxes[1:], strict=False)
                ],
            }
        )
    return {
        'sample_id': sample.sample_id,
        'image_size': list(image.size),
        'region_count': len(regions),
        'regions': values,
    }


def _display_case(case: ContextReviewCase, output: Callable[[str], None]) -> None:
    sample = case.sample
    value = {
        'sample_id': sample.sample_id,
        'render': {
            'renderer': sample.render.renderer,
            'layout': sample.render.layout,
            'punctuation': sample.render.punctuation,
            'font': sample.render.font,
            'size_px': sample.render.size_px,
        },
        'expected': {
            'sentence': sample.target.sentence,
            'target': sample.target.text,
            'target_span': list(sample.target.sentence_span),
            'lines': [
                {'text': line.text, 'box': _box_view(line.box)} for line in sample.lines
            ],
        },
        'actual': {
            'sentence': case.target.sentence,
            'target': case.target.surface,
            'target_span': [case.target.sentence_start, case.target.sentence_end],
            'lines': [
                {'text': line.text, 'box': _box_view(line.box)}
                for line in case.document.lines
            ],
        },
    }
    output(json.dumps(value, ensure_ascii=False, indent=2))


def _display_compact_case(
    case: ContextReviewCase,
    output: Callable[[str], None],
) -> None:
    sample = case.sample
    value = {
        'sample_id': sample.sample_id,
        'render': [
            sample.render.renderer,
            sample.render.layout,
            sample.render.punctuation,
            sample.render.font,
            sample.render.size_px,
        ],
        'expected_sentence': sample.target.sentence,
        'actual_sentence': case.target.sentence,
        'expected_span': list(sample.target.sentence_span),
        'actual_span': [case.target.sentence_start, case.target.sentence_end],
        'actual_lines': [line.text for line in case.document.lines],
    }
    output(json.dumps(value, ensure_ascii=False))


def review_context_cases(
    cases: tuple[ContextReviewCase, ...],
    existing: tuple[ContextReviewDecision, ...],
    *,
    prompt: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> tuple[ContextReviewDecision, ...]:
    decisions = {item.sample_id: item for item in existing}
    choices = '\n'.join(
        f'  {index}: {decision}'
        for index, decision in enumerate(CONTEXT_REVIEW_DECISIONS, start=1)
    )
    valid_answers = {
        str(index) for index in range(1, len(CONTEXT_REVIEW_DECISIONS) + 1)
    }
    for case in cases:
        if case.sample.sample_id in decisions:
            continue
        _display_case(case, output)
        output(choices)
        while True:
            answer = prompt('decision [1-5]: ').strip()
            if answer in valid_answers:
                decision = CONTEXT_REVIEW_DECISIONS[int(answer) - 1]
                break
            output('Choose a category from 1 through 5.')
        decisions[case.sample.sample_id] = ContextReviewDecision(
            case.sample.sample_id,
            decision,
        )
    return tuple(decisions.values())


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='Review context disagreements without persisting corpus text'
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
        '--decision',
        choices=CONTEXT_REVIEW_DECISIONS,
        help='with --inspect, display current cases in one reviewed category',
    )
    parser.add_argument(
        '--geometry-only',
        action='store_true',
        help='with --inspect, omit all corpus and recognized text',
    )
    parser.add_argument(
        '--sample-id',
        help='with --inspect, limit output to one stable sample ID',
    )
    parser.add_argument(
        '--segmentation-only',
        action='store_true',
        help='with --inspect and --sample-id, show only detector/segment geometry',
    )
    parser.add_argument(
        '--target-geometry',
        action='append',
        metavar='SAMPLE_ID',
        help=(
            'inspect target and negative-probe geometry for one stable ID; '
            'repeat to inspect multiple IDs'
        ),
    )
    parser.add_argument(
        '--target-segmentation',
        action='store_true',
        help='with --target-geometry, include detector and segment geometry',
    )
    arguments = parser.parse_args()
    if (
        arguments.compact
        or arguments.decision
        or arguments.geometry_only
        or arguments.sample_id
        or arguments.segmentation_only
    ) and not arguments.inspect:
        parser.error(
            '--compact, --decision, --geometry-only, --sample-id, and '
            '--segmentation-only require --inspect'
        )
    if arguments.compact and arguments.geometry_only:
        parser.error('--compact and --geometry-only cannot be combined')
    if arguments.segmentation_only and not arguments.sample_id:
        parser.error('--segmentation-only requires --sample-id')
    if arguments.target_geometry and (
        arguments.inspect
        or arguments.audit
        or arguments.compact
        or arguments.decision
        or arguments.geometry_only
        or arguments.sample_id
        or arguments.segmentation_only
    ):
        parser.error('--target-geometry cannot be combined with another review mode')
    if arguments.target_segmentation and not arguments.target_geometry:
        parser.error('--target-segmentation requires --target-geometry')

    validate_plain_corpus(arguments.corpus)
    corpus_id, locked = _lock_files(arguments.corpus)
    sources = load_sources(arguments.corpus, locked)
    selected = _quick_annotations(arguments.corpus, locked)
    samples = load_plain_samples(arguments.corpus, 'plain', locked, sources, selected)
    engine = PaddleOcrEngine(
        PaddleDetector(_asset(arguments.assets, 'korean_detection.onnx')),
        PaddleRecognizer(
            _asset(arguments.assets, 'korean_recognition.onnx'),
            _asset(arguments.assets, 'korean_characters.txt'),
        ),
    )
    if arguments.target_geometry:
        samples_by_id = {sample.sample_id: sample for sample in samples}
        missing = sorted(set(arguments.target_geometry) - set(samples_by_id))
        if missing:
            parser.error(f'unknown quick-tier sample ID: {missing[0]}')
        for sample_id in arguments.target_geometry:
            sample = samples_by_id[sample_id]
            with Image.open(sample.image) as source:
                document = engine.recognize(source.convert('RGB'))
            value = structural_target_geometry_view(sample, document)
            if arguments.target_segmentation:
                value['segmentation'] = structural_segmentation_view(engine, sample)
            print(
                json.dumps(
                    value,
                    ensure_ascii=True,
                )
            )
        return
    cases = collect_context_review_cases(engine, samples)
    existing = load_context_review(arguments.decisions, corpus_id)
    audit = audit_context_review(cases, existing)
    if arguments.audit:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        if not audit['complete']:
            raise SystemExit(1)
        return
    if arguments.inspect:
        for case in inspection_context_cases(
            cases,
            existing,
            arguments.decision,
            arguments.sample_id,
        ):
            if arguments.segmentation_only:
                print(
                    json.dumps(
                        structural_segmentation_view(engine, case.sample),
                        ensure_ascii=True,
                    )
                )
            elif arguments.geometry_only:
                print(json.dumps(structural_context_view(case), ensure_ascii=True))
            elif arguments.compact:
                _display_compact_case(case, print)
            else:
                _display_case(case, print)
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return
    decisions = review_context_cases(cases, existing)
    write_context_review(arguments.decisions, corpus_id, decisions)
    print(json.dumps(audit_context_review(cases, decisions), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
