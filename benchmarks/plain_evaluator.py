"""End-to-end, aggregate-only evaluation for the v4 plain-v1 corpus."""

from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from benchmarks.locked_corpus import (
    LOCK_NAME,
    PLAIN_SCHEMA_VERSION,
    SOURCE_MANIFEST_NAME,
    AnalyzerLike,
    CorpusError,
    ExpectedEojeol,
    ExpectedLine,
    Provenance,
    SourceEvidence,
    _box,
    _lock_files,
    _match,
    _p95,
    _provenance,
    _read_object,
    _sha256,
    _wilson_interval,
    load_sources,
)
from benchmarks.plain_corpus import (
    FONT_FAMILIES,
    LANGUAGE_COUNT,
    LANGUAGE_PER_CLASS,
    PLAIN_COUNT,
    PLAIN_ORACLE,
    PLAIN_SOURCE_LOCK,
    PUNCTUATION_CLASSES,
    QUICK_COUNT,
    REQUIRED_SIZES,
    STRESS_COUNT,
    STRESS_SIZE,
)
from bidan_lens.analysis.grammar import known_particle_suffixes
from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore, SqliteDictionaryStore
from bidan_lens.models import (
    AnalysisCandidate,
    BoundingBox,
    DictionaryEntry,
    OcrDocument,
    PopupResult,
)
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer
from bidan_lens.pipeline.hit_test import hit_test

OCR_PRIMARY = 97.0
OCR_FLOOR = 95.0
POPUP_PRIMARY = 92.0
POPUP_FLOOR = 88.0
FALSE_PROMOTION_MAX = 0.5
FALSE_ACTIVATION_MAX = 0.5
NEGATIVE_PROBE_KINDS = frozenset(
    {'whitespace', 'punctuation', 'english', 'blank', 'near-miss'}
)


@dataclass(frozen=True, slots=True)
class ExpectedDictionaryEntry:
    entry_id: str
    headword: str
    senses: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class ExpectedComponent:
    surface: str
    lemma: str
    learner_role: str
    entries: tuple[ExpectedDictionaryEntry, ...]


@dataclass(frozen=True, slots=True)
class NegativeProbe:
    kind: str
    pointer: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PlainTarget:
    text: str
    box: BoundingBox
    pointer: tuple[float, float]
    sentence: str
    sentence_span: tuple[int, int]
    expected_lemma: str
    expected_labels: frozenset[str]
    expected_dictionary_entries: tuple[ExpectedDictionaryEntry, ...]
    target_class: str
    expected_components: tuple[ExpectedComponent, ...]
    expected_spacing: str | None
    language_class: str | None


@dataclass(frozen=True, slots=True)
class RenderMetadata:
    renderer: str
    renderer_version: str
    font: str
    font_sha256: str
    size_px: int
    weight: int
    scale_percent: int
    theme: str
    layout: str
    punctuation: str
    stress: bool


@dataclass(frozen=True, slots=True)
class PlainSample:
    sample_id: str
    image: Path
    lines: tuple[ExpectedLine, ...]
    target: PlainTarget
    render: RenderMetadata
    provenance: Provenance
    negative_probes: tuple[NegativeProbe, ...]


@dataclass(frozen=True, slots=True)
class LanguageSample:
    sample_id: str
    sentence: str
    sentence_span: tuple[int, int]
    target: PlainTarget
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    sample_id: str
    render: RenderMetadata
    exact_eojeols: int
    total_eojeols: int
    missing_eojeols: int
    target_hit: bool
    sentence_span: bool
    sentence_exact: bool
    component_analysis: bool
    dictionary_fidelity: bool
    full_popup: bool
    alternative_recovery: bool
    false_promotion: bool
    negative_probes: tuple[str, ...]
    negative_activations: tuple[str, ...]
    latency_ms: float
    failed_stage: str | None


class PlainEngineLike(Protocol):
    def recognize(self, image: Image.Image) -> OcrDocument: ...


def _normal(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").strip()


def _expected_entries(
    value: Any, *, allow_empty: bool = False
) -> tuple[ExpectedDictionaryEntry, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CorpusError("plain target has no expected dictionary entries")
    entries: list[ExpectedDictionaryEntry] = []
    for raw_entry in value:
        if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("senses"), list):
            raise CorpusError("plain target has an invalid dictionary entry")
        entry_id = raw_entry.get("entry_id")
        headword = raw_entry.get("headword")
        senses: list[tuple[int, str]] = []
        for raw_sense in raw_entry["senses"]:
            if (
                not isinstance(raw_sense, dict)
                or not isinstance(raw_sense.get("order"), int)
                or not isinstance(raw_sense.get("definition"), str)
            ):
                raise CorpusError("plain target has an invalid dictionary sense")
            senses.append((raw_sense["order"], _normal(raw_sense["definition"])))
        if not isinstance(entry_id, str) or not isinstance(headword, str) or not senses:
            raise CorpusError("plain target has an invalid dictionary entry")
        entries.append(ExpectedDictionaryEntry(entry_id, _normal(headword), tuple(senses)))
    return tuple(entries)


def _expected_lines(value: Any) -> tuple[ExpectedLine, ...]:
    if not isinstance(value, list) or not value:
        raise CorpusError("plain sample has no expected lines")
    result: list[ExpectedLine] = []
    for raw_line in value:
        if not isinstance(raw_line, dict) or not isinstance(raw_line.get("eojeols"), list):
            raise CorpusError("plain sample contains an invalid line")
        text = raw_line.get("text")
        if not isinstance(text, str) or not text:
            raise CorpusError("plain sample contains an invalid line")
        eojeols: list[ExpectedEojeol] = []
        for raw_eojeol in raw_line["eojeols"]:
            if not isinstance(raw_eojeol, dict) or not isinstance(raw_eojeol.get("text"), str):
                raise CorpusError("plain sample contains an invalid eojeol")
            eojeols.append(ExpectedEojeol(_normal(raw_eojeol["text"]), _box(raw_eojeol.get("box"))))
        if not eojeols:
            raise CorpusError("plain sample line has no Korean eojeols")
        result.append(ExpectedLine(_normal(text), _box(raw_line.get("box")), tuple(eojeols)))
    return tuple(result)


def _expected_components(value: Any) -> tuple[ExpectedComponent, ...]:
    if not isinstance(value, list) or not value:
        raise CorpusError('plain target has no expected lexical components')
    components: list[ExpectedComponent] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise CorpusError('plain target has an invalid lexical component')
        surface = raw.get('surface')
        lemma = raw.get('lemma')
        role = raw.get('learner_role')
        if not all(isinstance(item, str) and item for item in (surface, lemma, role)):
            raise CorpusError('plain target has an invalid lexical component')
        components.append(
            ExpectedComponent(
                _normal(surface),
                _normal(lemma),
                role,
                _expected_entries(
                    raw.get('expected_dictionary_entries'), allow_empty=True
                ),
            )
        )
    return tuple(components)


def _negative_probes(value: Any) -> tuple[NegativeProbe, ...]:
    if not isinstance(value, list) or not value:
        raise CorpusError('plain sample has no negative probes')
    probes: list[NegativeProbe] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise CorpusError('plain sample has an invalid negative probe')
        kind = raw.get('kind')
        pointer = raw.get('pointer')
        if (
            kind not in {'blank', 'english', 'whitespace', 'punctuation', 'near-miss'}
            or not isinstance(pointer, list)
            or len(pointer) != 2
            or not all(isinstance(item, (int, float)) for item in pointer)
        ):
            raise CorpusError('plain sample has an invalid negative probe')
        probes.append(NegativeProbe(kind, (float(pointer[0]), float(pointer[1]))))
    return tuple(probes)


def _target(value: Any) -> PlainTarget:
    if not isinstance(value, dict):
        raise CorpusError("plain sample has no target")
    text = value.get("text")
    pointer = value.get("pointer")
    sentence = value.get("sentence")
    span = value.get("sentence_span")
    lemma = value.get("expected_lemma")
    labels = value.get("expected_labels")
    target_class = value.get("target_class")
    spacing = value.get('expected_spacing')
    language_class = value.get('language_class')
    valid = (
        isinstance(text, str)
        and bool(text)
        and isinstance(pointer, list)
        and len(pointer) == 2
        and all(isinstance(item, (int, float)) for item in pointer)
        and isinstance(sentence, str)
        and isinstance(span, list)
        and len(span) == 2
        and all(isinstance(item, int) for item in span)
        and isinstance(lemma, str)
        and isinstance(labels, list)
        and all(isinstance(item, str) for item in labels)
        and target_class in {"particle", "conjugated", "plain"}
        and (spacing is None or isinstance(spacing, str))
        and language_class in {None, 'multi-lexical', 'auxiliary'}
    )
    if not valid:
        raise CorpusError("plain sample target is invalid")
    start, end = span
    if start < 0 or end <= start or end > len(sentence) or sentence[start:end] != text:
        raise CorpusError("plain target span does not select its complete eojeol")
    box = _box(value.get("box"))
    if not box.contains(float(pointer[0]), float(pointer[1])):
        raise CorpusError("plain target pointer is outside its eojeol")
    return PlainTarget(
        _normal(text),
        box,
        (float(pointer[0]), float(pointer[1])),
        _normal(sentence),
        (start, end),
        _normal(lemma),
        frozenset(labels),
        _expected_entries(value.get("expected_dictionary_entries")),
        str(target_class),
        _expected_components(value.get('expected_components')),
        _normal(spacing) if isinstance(spacing, str) else None,
        language_class,
    )


def _render(value: Any, stress: bool) -> RenderMetadata:
    if not isinstance(value, dict):
        raise CorpusError("plain sample has no render metadata")
    renderer = value.get("renderer")
    renderer_version = value.get("renderer_version")
    font = value.get("font")
    font_sha256 = value.get("font_sha256")
    size = value.get("size_px")
    weight = value.get("weight")
    scale = value.get("scale_percent")
    theme = value.get("theme")
    layout = value.get("layout")
    punctuation = value.get("punctuation")
    valid = (
        renderer in {"browser", "desktop"}
        and isinstance(renderer_version, str)
        and bool(renderer_version)
        and font in FONT_FAMILIES
        and isinstance(font_sha256, str)
        and len(font_sha256) == 64
        and all(character in "0123456789abcdef" for character in font_sha256)
        and isinstance(size, int)
        and weight in {400, 700}
        and scale in {100, 125, 150, 200}
        and theme in {"light", "dark"}
        and layout in {"single-line", "multi-line"}
        and punctuation in PUNCTUATION_CLASSES
        and value.get("stress") is stress
    )
    expected_sizes = {STRESS_SIZE} if stress else set(REQUIRED_SIZES)
    if not valid or size not in expected_sizes:
        raise CorpusError("plain sample has invalid render metadata")
    return RenderMetadata(
        renderer,
        renderer_version,
        font,
        font_sha256,
        size,
        weight,
        scale,
        theme,
        layout,
        punctuation,
        stress,
    )


def _validate_build_metadata(root: Path, locked: dict[str, str]) -> None:
    required = {"acquisition.json", PLAIN_SOURCE_LOCK.name, "renderer.json", "quick.json"}
    if not required <= set(locked):
        raise CorpusError("plain-v1 corpus is missing hash-locked build metadata")
    source_lock = root / PLAIN_SOURCE_LOCK.name
    acquisition = _read_object(root / "acquisition.json")
    if (
        acquisition.get("schema_version") != 1
        or acquisition.get("source_lock") != PLAIN_SOURCE_LOCK.name
        or acquisition.get("source_lock_sha256") != _sha256(source_lock)
    ):
        raise CorpusError("plain-v1 acquisition metadata does not match its source lock")
    renderer = _read_object(root / "renderer.json")
    versions = renderer.get("renderers")
    if (
        renderer.get("schema_version") != 1
        or renderer.get("required_sizes_px") != list(REQUIRED_SIZES)
        or renderer.get("stress_size_px") != STRESS_SIZE
        or renderer.get("scales_percent") != [100, 125, 150, 200]
        or renderer.get("punctuation_classes") != list(PUNCTUATION_CLASSES)
        or not isinstance(versions, dict)
        or set(versions) != {"browser", "desktop"}
        or not all(isinstance(item, str) and item for item in versions.values())
    ):
        raise CorpusError("plain-v1 renderer metadata is invalid")


def load_plain_samples(
    root: Path,
    category: str,
    locked: dict[str, str],
    sources: dict[str, SourceEvidence],
    selected_annotations: frozenset[str] | None = None,
) -> tuple[PlainSample, ...]:
    if category not in {"plain", "plain_stress"}:
        raise CorpusError("invalid plain-v1 category")
    stress = category == "plain_stress"
    category_root = root / category
    annotations = sorted(category_root.glob("*.json")) if category_root.is_dir() else []
    samples: list[PlainSample] = []
    identities: set[tuple[str, str]] = set()
    for annotation in annotations:
        annotation_key = annotation.relative_to(root).as_posix()
        if selected_annotations is not None and annotation_key not in selected_annotations:
            continue
        if annotation_key not in locked:
            raise CorpusError("plain annotation is not hash-locked")
        value = _read_object(annotation)
        if value.get("schema_version") != PLAIN_SCHEMA_VERSION:
            raise CorpusError("plain annotation has an unsupported schema")
        image_name = value.get("image")
        sample_id = value.get("sample_id")
        if not isinstance(image_name, str) or not isinstance(sample_id, str) or not sample_id:
            raise CorpusError("plain sample identity is invalid")
        image = annotation.parent / image_name
        try:
            image_key = image.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise CorpusError("plain sample image escapes the corpus") from error
        if image_key not in locked or not image.is_file():
            raise CorpusError("plain sample image is not hash-locked")
        provenance = _provenance(value.get("provenance"), sources)
        if provenance is None or provenance.oracle != PLAIN_ORACLE:
            raise CorpusError("plain sample does not use the independent render oracle")
        identity = (provenance.source_id, provenance.source_sample_id)
        if identity in identities:
            raise CorpusError("plain category contains a duplicate source sample")
        identities.add(identity)
        samples.append(
            PlainSample(
                sample_id,
                image,
                _expected_lines(value.get("lines")),
                _target(value.get("target")),
                _render(value.get("render"), stress),
                provenance,
                _negative_probes(value.get('negative_probes')),
            )
        )
    return tuple(samples)


def _quick_annotations(root: Path, locked: dict[str, str]) -> frozenset[str]:
    path = root / "quick.json"
    if "quick.json" not in locked or not path.is_file():
        raise CorpusError("plain quick manifest is not hash-locked")
    value = _read_object(path)
    samples = value.get("samples")
    if value.get("schema_version") != 1 or not isinstance(samples, list):
        raise CorpusError("plain quick manifest is invalid")
    selected = frozenset(item for item in samples if isinstance(item, str))
    if len(selected) != len(samples) or any(item not in locked for item in selected):
        raise CorpusError("plain quick manifest references an invalid sample")
    return selected


def load_language_samples(
    root: Path,
    locked: dict[str, str],
    sources: dict[str, SourceEvidence],
) -> tuple[LanguageSample, ...]:
    language_root = root / 'language'
    annotations = sorted(language_root.glob('*.json')) if language_root.is_dir() else []
    samples: list[LanguageSample] = []
    identities: set[tuple[str, str]] = set()
    for annotation in annotations:
        key = annotation.relative_to(root).as_posix()
        if key not in locked:
            raise CorpusError('focused language annotation is not hash-locked')
        value = _read_object(annotation)
        if value.get('schema_version') != PLAIN_SCHEMA_VERSION:
            raise CorpusError('focused language annotation has an unsupported schema')
        sample_id = value.get('sample_id')
        provenance = _provenance(value.get('provenance'), sources)
        if not isinstance(sample_id, str) or provenance is None:
            raise CorpusError('focused language sample is invalid')
        identity = (provenance.source_id, provenance.source_sample_id)
        if identity in identities:
            raise CorpusError('focused language tier contains a duplicate source sample')
        identities.add(identity)
        target = _target(value.get('target'))
        samples.append(
            LanguageSample(
                sample_id,
                target.sentence,
                target.sentence_span,
                target,
                provenance,
            )
        )
    return tuple(samples)


def _evidence_files(root: Path) -> list[str]:
    manifest = _read_object(root / SOURCE_MANIFEST_NAME)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CorpusError("plain source manifest has no sources")
    result: list[str] = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("license_evidence"), str):
            raise CorpusError("plain source has no license evidence")
        result.append(source["license_evidence"])
    return sorted(set(result))


def lock_plain_corpus(
    root: Path, corpus_id: str, *, allow_incomplete: bool = False
) -> dict[str, object]:
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {LOCK_NAME, f"{LOCK_NAME}.tmp"}
    }
    value = {
        "schema_version": PLAIN_SCHEMA_VERSION,
        "profile": "plain-v1",
        "corpus_id": corpus_id,
        "source_manifest": SOURCE_MANIFEST_NAME,
        "license_evidence": _evidence_files(root),
        "files": files,
    }
    temporary = root / f"{LOCK_NAME}.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(root / LOCK_NAME)
    return validate_plain_corpus(root, allow_incomplete=allow_incomplete)


def validate_plain_corpus(root: Path, *, allow_incomplete: bool = False) -> dict[str, object]:
    lock_value = _read_object(root / LOCK_NAME)
    if lock_value.get("schema_version") != PLAIN_SCHEMA_VERSION:
        raise CorpusError("plain-v1 requires a v4 corpus lock")
    if lock_value.get("profile") != "plain-v1":
        raise CorpusError("v4 corpus lock does not name the plain-v1 profile")
    corpus_id, locked = _lock_files(root)
    _validate_build_metadata(root, locked)
    sources = load_sources(root, locked)
    plain = load_plain_samples(root, "plain", locked, sources)
    stress = load_plain_samples(root, "plain_stress", locked, sources)
    language = load_language_samples(root, locked, sources)
    quick = _quick_annotations(root, locked)
    all_identities = [
        (sample.provenance.source_id, sample.provenance.source_sample_id)
        for sample in (*plain, *stress, *language)
    ]
    if len(set(all_identities)) != len(all_identities):
        raise CorpusError("plain and stress corpora share a source sample")
    renderer_versions = _read_object(root / "renderer.json")["renderers"]
    if any(
        sample.render.renderer_version != renderer_versions[sample.render.renderer]
        for sample in (*plain, *stress)
    ):
        raise CorpusError("plain sample renderer version differs from renderer metadata")
    complete = (
        len(plain) == PLAIN_COUNT
        and len(stress) == STRESS_COUNT
        and len(language) == LANGUAGE_COUNT
        and len(quick) == QUICK_COUNT
    )
    if complete:
        size_counts = Counter(sample.render.size_px for sample in plain)
        punctuation_counts = Counter(sample.render.punctuation for sample in plain)
        renderer_counts = Counter(sample.render.renderer for sample in plain)
        font_counts = Counter(sample.render.font for sample in plain)
        weight_counts = Counter(sample.render.weight for sample in plain)
        scale_counts = Counter(sample.render.scale_percent for sample in plain)
        theme_counts = Counter(sample.render.theme for sample in plain)
        layout_counts = Counter(sample.render.layout for sample in plain)
        target_counts = Counter(sample.target.target_class for sample in plain)
        main_language_counts = Counter(
            sample.target.language_class
            for sample in plain
            if sample.target.language_class is not None
        )
        language_counts = Counter(sample.target.language_class for sample in language)
        negative_probe_kinds = {
            probe.kind for sample in plain for probe in sample.negative_probes
        }
        balanced = (
            size_counts
            == Counter(
                {size: PLAIN_COUNT // len(REQUIRED_SIZES) for size in REQUIRED_SIZES}
            )
            and punctuation_counts
            == Counter(
                {
                    punctuation: PLAIN_COUNT // len(PUNCTUATION_CLASSES)
                    for punctuation in PUNCTUATION_CLASSES
                }
            )
            and renderer_counts == Counter({"browser": 1_000, "desktop": 1_000})
            and font_counts == Counter(
                {
                    "noto-sans-kr": 400,
                    "noto-serif-kr": 400,
                    "nanum-gothic": 400,
                    "nanum-myeongjo": 400,
                    "malgun-gothic": 400,
                }
            )
            and weight_counts == Counter({400: 1_000, 700: 1_000})
            and scale_counts == Counter({100: 500, 125: 500, 150: 500, 200: 500})
            and theme_counts == Counter({"light": 1_000, "dark": 1_000})
            and layout_counts
            == Counter({"single-line": 1_000, "multi-line": 1_000})
            and target_counts
            == Counter({"particle": 800, "conjugated": 800, "plain": 400})
            and main_language_counts['multi-lexical'] >= 100
            and main_language_counts['auxiliary'] >= 100
            and language_counts
            == Counter(
                {
                    'multi-lexical': LANGUAGE_PER_CLASS,
                    'auxiliary': LANGUAGE_PER_CLASS,
                }
            )
            and negative_probe_kinds == NEGATIVE_PROBE_KINDS
            and all(sample.render.size_px == STRESS_SIZE for sample in stress)
        )
        if not balanced:
            raise CorpusError("plain-v1 corpus does not meet its required balanced strata")
    if not complete and not allow_incomplete:
        raise CorpusError("plain-v1 corpus does not have exact release sample counts")
    split_counts = Counter(
        sample.provenance.source_split for sample in (*plain, *stress, *language)
    )
    return {
        "corpus_id": corpus_id,
        "profile": "plain-v1",
        "sources": len(sources),
        "counts": {
            "plain": len(plain),
            "plain_stress": len(stress),
            'language': len(language),
            "quick": len(quick),
        },
        "splits": dict(sorted(split_counts.items())),
        "release_sample_counts": complete,
    }


def _entry_signature(entry: DictionaryEntry) -> tuple[str, str, tuple[tuple[int, str], ...]]:
    return (
        entry.entry_id,
        _normal(entry.headword),
        tuple((sense.order, _normal(sense.definition)) for sense in entry.senses),
    )


def _expected_signature(
    entry: ExpectedDictionaryEntry,
) -> tuple[str, str, tuple[tuple[int, str], ...]]:
    return (entry.entry_id, entry.headword, entry.senses)


_IGNORABLE_EDGE_PUNCTUATION = frozenset(
    map(
        chr,
        (
            33, 34, 39, 40, 41, 44, 45, 46, 47, 58, 59, 63, 91, 93, 123, 125,
            0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026,
            0x3008, 0x3009, 0x300A, 0x300B,
        ),
    )
)


def _context_tokens(sentence: str) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    matches = tuple(re.finditer(r'\S+', _normal(sentence)))
    tokens = tuple(
        match.group().strip(''.join(_IGNORABLE_EDGE_PUNCTUATION)) for match in matches
    )
    return tokens, tuple(match.span() for match in matches)


def _target_token_index(spans: tuple[tuple[int, int], ...], target_span: tuple[int, int]) -> int:
    start, end = target_span
    for index, (token_start, token_end) in enumerate(spans):
        if token_start <= start and end <= token_end:
            return index
    return -1


def _functional_context(
    actual_sentence: str,
    actual_span: tuple[int, int],
    expected_sentence: str,
    expected_span: tuple[int, int],
) -> bool:
    actual_tokens, actual_spans = _context_tokens(actual_sentence)
    expected_tokens, expected_spans = _context_tokens(expected_sentence)
    return (
        actual_tokens == expected_tokens
        and _target_token_index(actual_spans, actual_span)
        == _target_token_index(expected_spans, expected_span)
        and 0 <= actual_span[0] < actual_span[1] <= len(actual_sentence)
    )


def _component_matches(candidate: AnalysisCandidate, target: PlainTarget) -> bool:
    actual = tuple(
        (
            _normal(component.surface),
            _normal(component.lemma),
            component.learner_role,
        )
        for component in candidate.lexical_components
    )
    expected = tuple(
        (
            component.surface,
            component.lemma,
            component.learner_role,
        )
        for component in target.expected_components
    )
    return actual == expected


def _spacing_matches(candidate: AnalysisCandidate, target: PlainTarget) -> bool:
    actual = _normal(candidate.interpreted_surface) if candidate.interpreted_surface else None
    return actual == target.expected_spacing


def _analysis_matches(candidate: AnalysisCandidate | None, target: PlainTarget) -> bool:
    if candidate is None or _normal(candidate.lemma) != target.expected_lemma:
        return False
    labels = {feature.label for feature in candidate.features}
    labels.update(morpheme.learner_label for morpheme in candidate.morphemes)
    return target.expected_labels <= labels and _component_matches(candidate, target)


def _dictionary_matches(candidate: AnalysisCandidate | None, target: PlainTarget) -> bool:
    if candidate is None:
        return False
    actual = tuple(_entry_signature(entry) for entry in candidate.dictionary_entries)
    expected = tuple(_expected_signature(entry) for entry in target.expected_dictionary_entries)
    actual_components = tuple(
        tuple(_entry_signature(entry) for entry in component.dictionary_entries)
        for component in candidate.lexical_components
    )
    expected_components = tuple(
        tuple(_expected_signature(entry) for entry in component.entries)
        for component in target.expected_components
    )
    return actual == expected and actual_components == expected_components


def _role_dictionary_positions(learner_role: str) -> tuple[str, ...]:
    if learner_role == 'helping verb':
        return ('보조 동사', '보조 형용사')
    if learner_role == 'descriptive verb':
        return ('adjective',)
    if learner_role == 'action verb':
        return ('verb',)
    if learner_role in {
        'noun',
        'name or proper noun',
        'pronoun',
        'number',
        'dependent noun',
    }:
        return ('noun',)
    return ()


def _direct_component_entries(
    dictionary: DictionaryStore, component: ExpectedComponent
) -> tuple[DictionaryEntry, ...]:
    ordered: list[DictionaryEntry] = []
    seen: set[str] = set()
    for part_of_speech in _role_dictionary_positions(component.learner_role):
        for entry in dictionary.lookup(component.lemma, part_of_speech, 10):
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                ordered.append(entry)
    for entry in dictionary.lookup(component.lemma, None, 10):
        if entry.entry_id not in seen:
            seen.add(entry.entry_id)
            ordered.append(entry)
    return tuple(ordered[:10])


def _direct_dictionary_conformance(
    dictionary: DictionaryStore, samples: tuple[LanguageSample, ...]
) -> dict[str, object]:
    expected_groups: dict[
        tuple[str, str, tuple[tuple[str, str, tuple[tuple[int, str], ...]], ...]],
        ExpectedComponent,
    ] = {}
    for sample in samples:
        for component in sample.target.expected_components:
            signature = tuple(_expected_signature(entry) for entry in component.entries)
            expected_groups.setdefault(
                (component.lemma, component.learner_role, signature), component
            )
    matches = 0
    mismatch_reasons: Counter[str] = Counter()
    for component in expected_groups.values():
        actual = tuple(
            _entry_signature(entry)
            for entry in _direct_component_entries(dictionary, component)
        )
        expected = tuple(_expected_signature(entry) for entry in component.entries)
        if actual == expected:
            matches += 1
        else:
            actual_ids = tuple(entry[0] for entry in actual)
            expected_ids = tuple(entry[0] for entry in expected)
            if set(actual_ids) != set(expected_ids):
                mismatch_reasons['entry_set'] += 1
            elif actual_ids != expected_ids:
                mismatch_reasons['entry_order'] += 1
            elif tuple(entry[1] for entry in actual) != tuple(
                entry[1] for entry in expected
            ):
                mismatch_reasons['headword'] += 1
            else:
                mismatch_reasons['senses'] += 1
    total = len(expected_groups)
    if total == 0:
        raise CorpusError('plain evaluation contains no dictionary conformance groups')
    return {
        'groups': total,
        'matching_groups': matches,
        'mismatching_groups': total - matches,
        'mismatch_reasons': dict(sorted(mismatch_reasons.items())),
        'pct': _percent(matches, total),
    }


def _language_failure_stage(
    candidate: AnalysisCandidate | None, target: PlainTarget
) -> str | None:
    if candidate is None:
        return 'missing_candidate'
    if _normal(candidate.lemma) != target.expected_lemma:
        return 'primary_lemma'
    labels = {feature.label for feature in candidate.features}
    labels.update(morpheme.learner_label for morpheme in candidate.morphemes)
    if not target.expected_labels <= labels:
        return 'grammar_roles'
    if len(candidate.lexical_components) != len(target.expected_components):
        return 'component_count'
    for actual, expected in zip(
        candidate.lexical_components, target.expected_components, strict=True
    ):
        if _normal(actual.surface) != expected.surface:
            return 'component_surface'
        if _normal(actual.lemma) != expected.lemma:
            return 'component_lemma'
        if actual.learner_role != expected.learner_role:
            return 'component_role'
        if tuple(_entry_signature(entry) for entry in actual.dictionary_entries) != tuple(
            _expected_signature(entry) for entry in expected.entries
        ):
            return 'contextual_dictionary_group'
    if tuple(_entry_signature(entry) for entry in candidate.dictionary_entries) != tuple(
        _expected_signature(entry) for entry in target.expected_dictionary_entries
    ):
        return 'primary_dictionary_group'
    if not _spacing_matches(candidate, target):
        return 'spacing'
    return None


def _evaluate_sample(
    engine: PlainEngineLike, analyzer: AnalyzerLike, sample: PlainSample
) -> SampleOutcome:
    with Image.open(sample.image) as source:
        image = source.convert("RGB")
    component_candidates = analyzer.analyze(sample.target.sentence, sample.target.sentence_span)
    component_first = component_candidates[0] if component_candidates else None
    component_analysis = bool(
        component_first
        and _analysis_matches(component_first, sample.target)
        and _spacing_matches(component_first, sample.target)
    )
    dictionary_fidelity = _dictionary_matches(component_first, sample.target)

    started = time.perf_counter()
    document = engine.recognize(image)
    predicted_eojeols = [(item.text, item.box) for line in document.lines for item in line.eojeols]
    exact_eojeols = missing_eojeols = total_eojeols = 0
    for expected_line in sample.lines:
        for expected_eojeol in expected_line.eojeols:
            total_eojeols += 1
            match = _match(expected_eojeol.box, predicted_eojeols)
            if match is None:
                missing_eojeols += 1
                continue
            text, index = match
            exact_eojeols += _normal(text) == expected_eojeol.text
            predicted_eojeols.pop(index)

    target = hit_test(document, *sample.target.pointer)
    target_hit = bool(
        target
        and _normal(target.surface) == sample.target.text
        and _match(sample.target.box, [(target.surface, target.box)]) is not None
    )
    end_candidates: tuple[AnalysisCandidate, ...] = ()
    functional_context = False
    sentence_exact = False
    if target_hit and target is not None:
        sentence_exact = (
            _normal(target.sentence) == sample.target.sentence
            and (target.sentence_start, target.sentence_end) == sample.target.sentence_span
        )
        functional_context = _functional_context(
            target.sentence,
            (target.sentence_start, target.sentence_end),
            sample.target.sentence,
            sample.target.sentence_span,
        )
        end_candidates = analyzer.analyze(
            target.sentence, (target.sentence_start, target.sentence_end)
        )
    end_first = end_candidates[0] if end_candidates else None
    first_correct = _analysis_matches(end_first, sample.target) and _dictionary_matches(
        end_first, sample.target
    ) and bool(end_first and _spacing_matches(end_first, sample.target))
    alternative_recovery = any(
        _analysis_matches(candidate, sample.target)
        and _dictionary_matches(candidate, sample.target)
        and _spacing_matches(candidate, sample.target)
        for candidate in end_candidates
    )
    full_popup = bool(target_hit and functional_context and first_correct)
    false_promotion = bool(
        end_first
        and end_first.interpreted_surface is not None
        and not _spacing_matches(end_first, sample.target)
    )
    negative_kinds = tuple(probe.kind for probe in sample.negative_probes)
    negative_activations = tuple(
        probe.kind
        for probe in sample.negative_probes
        if hit_test(document, *probe.pointer) is not None
    )
    if target_hit and target is not None:
        PopupResult(target, end_candidates, requested_at=started)
    latency_ms = (time.perf_counter() - started) * 1000
    if not target_hit:
        failed_stage = "target"
    elif not functional_context:
        failed_stage = "context"
    elif not _analysis_matches(end_first, sample.target):
        failed_stage = "analysis"
    elif not _dictionary_matches(end_first, sample.target):
        failed_stage = "dictionary"
    elif end_first is not None and not _spacing_matches(end_first, sample.target):
        failed_stage = 'spacing'
    else:
        failed_stage = None
    return SampleOutcome(
        sample.sample_id,
        sample.render,
        exact_eojeols,
        total_eojeols,
        missing_eojeols,
        target_hit,
        functional_context,
        sentence_exact,
        component_analysis,
        dictionary_fidelity,
        full_popup,
        alternative_recovery,
        false_promotion,
        negative_kinds,
        negative_activations,
        latency_ms,
        failed_stage,
    )


def _percent(successes: int, total: int) -> float:
    return round(successes / total * 100, 2) if total else 0.0


def _outcome_summary(values: Iterable[SampleOutcome]) -> dict[str, object]:
    outcomes = tuple(values)
    samples = len(outcomes)
    exact = sum(item.exact_eojeols for item in outcomes)
    eojeols = sum(item.total_eojeols for item in outcomes)
    missing = sum(item.missing_eojeols for item in outcomes)
    popup = sum(item.full_popup for item in outcomes)
    target = sum(item.target_hit for item in outcomes)
    sentences = sum(item.sentence_span for item in outcomes)
    exact_sentences = sum(item.sentence_exact for item in outcomes)
    component = sum(item.component_analysis for item in outcomes)
    definitions = sum(item.dictionary_fidelity for item in outcomes)
    alternatives = sum(item.alternative_recovery for item in outcomes)
    false_promotions = sum(item.false_promotion for item in outcomes)
    negative_probe_counts = Counter(
        kind for item in outcomes for kind in item.negative_probes
    )
    negative_activation_counts = Counter(
        kind for item in outcomes for kind in item.negative_activations
    )
    if not samples or not eojeols:
        raise CorpusError("plain evaluation contains no measurable samples")
    ocr_low, ocr_high = _wilson_interval(exact, eojeols)
    popup_low, popup_high = _wilson_interval(popup, samples)
    durations = [item.latency_ms for item in outcomes]
    return {
        "samples": samples,
        "eojeols": eojeols,
        "whole_eojeol_exact_pct": _percent(exact, eojeols),
        "whole_eojeol_exact_ci95_low_pct": round(ocr_low * 100, 2),
        "whole_eojeol_exact_ci95_high_pct": round(ocr_high * 100, 2),
        "missing_eojeol_pct": _percent(missing, eojeols),
        "target_selection_pct": _percent(target, samples),
        'functional_context_accuracy_pct': _percent(sentences, samples),
        'exact_sentence_transcription_pct': _percent(exact_sentences, samples),
        "component_lemma_breakdown_first_pct": _percent(component, samples),
        "exact_krdict_fidelity_first_pct": _percent(definitions, samples),
        "fully_correct_first_popup_pct": _percent(popup, samples),
        "fully_correct_first_popup_ci95_low_pct": round(popup_low * 100, 2),
        "fully_correct_first_popup_ci95_high_pct": round(popup_high * 100, 2),
        "alternative_candidate_recovery_pct": _percent(alternatives, samples),
        "false_promotions": false_promotions,
        "false_promotion_rate_pct": round(false_promotions / samples * 100, 3),
        'negative_activation_rate_pct': _percent(
            sum(negative_activation_counts.values()), sum(negative_probe_counts.values())
        ),
        'negative_activation_by_kind': {
            kind: {
                'probes': total,
                'activations': negative_activation_counts[kind],
                'rate_pct': _percent(negative_activation_counts[kind], total),
            }
            for kind, total in sorted(negative_probe_counts.items())
        },
        "latency_median_ms": round(statistics.median(durations), 2),
        "latency_p95_ms": round(_p95(durations), 2),
    }


def _strata(outcomes: tuple[SampleOutcome, ...]) -> dict[str, object]:
    dimensions = {
        "size_px": lambda item: str(item.render.size_px),
        "font": lambda item: item.render.font,
        "renderer": lambda item: item.render.renderer,
        "scale_percent": lambda item: str(item.render.scale_percent),
        "punctuation": lambda item: item.render.punctuation,
        "weight": lambda item: str(item.render.weight),
        "theme": lambda item: item.render.theme,
        "layout": lambda item: item.render.layout,
    }
    result: dict[str, object] = {}
    for dimension, key in dimensions.items():
        grouped: dict[str, list[SampleOutcome]] = defaultdict(list)
        for outcome in outcomes:
            grouped[key(outcome)].append(outcome)
        result[dimension] = {
            value: _outcome_summary(items) for value, items in sorted(grouped.items())
        }
    return result


def _strata_meet_floors(strata: dict[str, object]) -> bool:
    for dimension in ("size_px", "punctuation"):
        values = strata[dimension]
        assert isinstance(values, dict)
        for result in values.values():
            assert isinstance(result, dict)
            if (
                result["whole_eojeol_exact_pct"] < OCR_FLOOR
                or result["fully_correct_first_popup_pct"] < POPUP_FLOOR
            ):
                return False
    return True


def _asset(assets: Path, name: str) -> Path:
    candidates = (assets / name, assets / "models" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise CorpusError(f"plain evaluator is missing production asset: {name}")


def _write_diagnostics(path: Path, outcomes: tuple[SampleOutcome, ...]) -> None:
    failures = [
        {
            "sample_id": item.sample_id,
            "failed_stage": item.failed_stage,
            "size_px": item.render.size_px,
            "font": item.render.font,
            "renderer": item.render.renderer,
            "punctuation": item.render.punctuation,
        }
        for item in outcomes
        if item.failed_stage is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def evaluate_plain(
    engine: PlainEngineLike,
    analyzer: AnalyzerLike,
    samples: tuple[PlainSample, ...],
) -> tuple[dict[str, object], tuple[SampleOutcome, ...]]:
    outcomes = tuple(_evaluate_sample(engine, analyzer, sample) for sample in samples)
    summary = _outcome_summary(outcomes)
    summary["strata"] = _strata(outcomes)
    return summary, outcomes


def evaluate_language(
    analyzer: AnalyzerLike,
    dictionary: DictionaryStore,
    samples: tuple[LanguageSample, ...],
) -> dict[str, object]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    grouped_failures: dict[str, Counter[str]] = defaultdict(Counter)
    failures: Counter[str] = Counter()
    role_confusions: Counter[str] = Counter()
    missing_grammar_roles: Counter[str] = Counter()
    primary_lemma_alternative_ranks: Counter[str] = Counter()
    primary_lemma_alternative_signals: Counter[str] = Counter()
    primary_lemma_structure: Counter[str] = Counter()
    particle_recovery_signals: Counter[str] = Counter()
    multi_component_rerank_audit: Counter[str] = Counter()
    all_results: list[bool] = []
    for sample in samples:
        candidates = analyzer.analyze(sample.sentence, sample.sentence_span)
        first = candidates[0] if candidates else None
        failure_stage = _language_failure_stage(first, sample.target)
        correct = failure_stage is None
        language_class = sample.target.language_class
        if language_class is None:
            raise CorpusError('focused language sample has no language class')
        if first is not None:
            richer = next(
                (
                    candidate
                    for candidate in candidates[1:]
                    if len(candidate.lexical_components) > len(first.lexical_components)
                    and len(candidate.lexical_components) >= 2
                    and all(
                        component.dictionary_entries
                        for component in candidate.lexical_components
                    )
                ),
                None,
            )
            if richer is not None:
                gap = first.score - richer.score
                if gap <= 0.25:
                    gap_bucket = 'gap_le_0_25'
                elif gap <= 0.5:
                    gap_bucket = 'gap_le_0_5'
                elif gap <= 1.0:
                    gap_bucket = 'gap_le_1_0'
                else:
                    gap_bucket = 'gap_gt_1_0'
                if _language_failure_stage(richer, sample.target) is None:
                    outcome = 'recover'
                elif failure_stage is None:
                    outcome = 'regress'
                else:
                    outcome = 'still_incorrect'
                multi_component_rerank_audit[f'{gap_bucket}_{outcome}'] += 1
        grouped[language_class].append(correct)
        if failure_stage is not None:
            failures[failure_stage] += 1
            grouped_failures[language_class][failure_stage] += 1
            if failure_stage == 'component_role' and first is not None:
                for actual, expected in zip(
                    first.lexical_components,
                    sample.target.expected_components,
                    strict=True,
                ):
                    if actual.learner_role != expected.learner_role:
                        role_confusions[
                            f'{expected.learner_role} -> {actual.learner_role}'
                        ] += 1
                        break
            elif failure_stage == 'primary_lemma':
                expected_components = sample.target.expected_components
                actual_components = first.lexical_components if first is not None else ()
                primary_lemma_structure[
                    f'component_count_{len(expected_components)}_to_{len(actual_components)}'
                ] += 1
                if len(actual_components) == len(expected_components):
                    primary_lemma_structure['component_count_matches'] += 1
                    surfaces_match = all(
                        _normal(actual.surface) == expected.surface
                        for actual, expected in zip(
                            actual_components, expected_components, strict=True
                        )
                    )
                    roles_match = all(
                        actual.learner_role == expected.learner_role
                        for actual, expected in zip(
                            actual_components, expected_components, strict=True
                        )
                    )
                    lemmas_match = sum(
                        _normal(actual.lemma) == expected.lemma
                        for actual, expected in zip(
                            actual_components, expected_components, strict=True
                        )
                    )
                    if surfaces_match:
                        primary_lemma_structure['component_surfaces_match'] += 1
                    if roles_match:
                        primary_lemma_structure['component_roles_match'] += 1
                    if lemmas_match == len(expected_components):
                        primary_lemma_structure['all_component_lemmas_match'] += 1
                    elif lemmas_match == len(expected_components) - 1:
                        primary_lemma_structure['one_component_lemma_differs'] += 1
                matching = next(
                    (
                        (rank, candidate)
                        for rank, candidate in enumerate(candidates[1:], start=2)
                        if _normal(candidate.lemma) == sample.target.expected_lemma
                    ),
                    None,
                )
                primary_lemma_alternative_ranks[
                    f'rank_{matching[0]}' if matching is not None else 'no_match'
                ] += 1
                if matching is not None and first is not None:
                    alternative = matching[1]
                    if sample.target.expected_spacing is not None:
                        primary_lemma_alternative_signals['verified_spacing'] += 1
                    if _component_matches(alternative, sample.target):
                        primary_lemma_alternative_signals['complete_components'] += 1
                    if _dictionary_matches(alternative, sample.target):
                        primary_lemma_alternative_signals['complete_dictionary'] += 1
                    alternative_labels = {
                        feature.label for feature in alternative.features
                    }
                    alternative_labels.update(
                        morpheme.learner_label for morpheme in alternative.morphemes
                    )
                    if sample.target.expected_labels <= alternative_labels:
                        primary_lemma_alternative_signals['complete_grammar'] += 1
                    first_component_count = len(first.lexical_components)
                    alternative_component_count = len(alternative.lexical_components)
                    if alternative_component_count > first_component_count:
                        primary_lemma_alternative_signals['more_components'] += 1
                    elif alternative_component_count == first_component_count:
                        primary_lemma_alternative_signals['equal_components'] += 1
                    else:
                        primary_lemma_alternative_signals['fewer_components'] += 1
                    first_defined = sum(
                        bool(component.dictionary_entries)
                        for component in first.lexical_components
                    )
                    alternative_defined = sum(
                        bool(component.dictionary_entries)
                        for component in alternative.lexical_components
                    )
                    if alternative_defined > first_defined:
                        primary_lemma_alternative_signals[
                            'more_defined_components'
                        ] += 1
                    elif alternative_defined == first_defined:
                        primary_lemma_alternative_signals[
                            'equal_defined_components'
                        ] += 1
                    else:
                        primary_lemma_alternative_signals[
                            'fewer_defined_components'
                        ] += 1
            elif failure_stage == 'grammar_roles' and first is not None:
                actual_labels = {feature.label for feature in first.features}
                actual_labels.update(
                    morpheme.learner_label for morpheme in first.morphemes
                )
                missing = sample.target.expected_labels - actual_labels
                missing_grammar_roles.update(missing)
                if 'particle' in missing:
                    particle_recovery_signals['cases'] += 1
                    particle_recovery_signals[
                        f'component_count_{len(first.lexical_components)}'
                    ] += 1
                    particle_recovery_signals[
                        f'target_class_{sample.target.target_class}'
                    ] += 1
                    if sample.target.expected_components:
                        suffix_length = len(sample.target.text) - len(
                            ''.join(
                                component.surface
                                for component in sample.target.expected_components
                            )
                        )
                        particle_recovery_signals[
                            f'oracle_nonlexical_length_{suffix_length}'
                        ] += 1
                    suffix = next(
                        (
                            value
                            for value in known_particle_suffixes()
                            if sample.target.text.endswith(value)
                            and len(sample.target.text) > len(value)
                        ),
                        None,
                    )
                    if suffix is not None:
                        particle_recovery_signals['known_suffix'] += 1
                        base = sample.target.text[: -len(suffix)]
                        if dictionary.lookup(base, 'noun', 1) or dictionary.lookup(
                            base, None, 1
                        ):
                            particle_recovery_signals['dictionary_backed_base'] += 1
                        if _normal(first.lemma) == base:
                            particle_recovery_signals['candidate_lemma_matches_base'] += 1
                        if len(first.lexical_components) == 1:
                            component = first.lexical_components[0]
                            if component.learner_role in {
                                'noun',
                                'name or proper noun',
                                'pronoun',
                                'number',
                                'dependent noun',
                            }:
                                particle_recovery_signals['single_noun_component'] += 1
                            if _normal(component.surface) == base:
                                particle_recovery_signals[
                                    'component_surface_matches_base'
                                ] += 1
        all_results.append(correct)
    if not all_results:
        raise CorpusError('plain evaluation contains no focused language samples')
    successes = sum(all_results)
    low, high = _wilson_interval(successes, len(all_results))
    conformance = _direct_dictionary_conformance(dictionary, samples)
    return {
        'samples': len(all_results),
        'fully_correct_first_popup_pct': _percent(successes, len(all_results)),
        'fully_correct_first_popup_ci95_low_pct': round(low * 100, 2),
        'fully_correct_first_popup_ci95_high_pct': round(high * 100, 2),
        'direct_krdict_conformance_pct': conformance['pct'],
        'direct_krdict_conformance': conformance,
        'failure_stages': dict(sorted(failures.items())),
        'component_role_confusions': dict(sorted(role_confusions.items())),
        'missing_grammar_roles': dict(sorted(missing_grammar_roles.items())),
        'primary_lemma_alternative_ranks': dict(
            sorted(primary_lemma_alternative_ranks.items())
        ),
        'primary_lemma_alternative_signals': dict(
            sorted(primary_lemma_alternative_signals.items())
        ),
        'primary_lemma_structure': dict(sorted(primary_lemma_structure.items())),
        'particle_recovery_signals': dict(sorted(particle_recovery_signals.items())),
        'multi_component_rerank_audit': dict(
            sorted(multi_component_rerank_audit.items())
        ),
        'by_class': {
            language_class: {
                'samples': len(values),
                'fully_correct_first_popup_pct': _percent(sum(values), len(values)),
                'failure_stages': dict(sorted(grouped_failures[language_class].items())),
            }
            for language_class, values in sorted(grouped.items())
        },
    }


def run_plain(
    assets: Path,
    corpus: Path,
    *,
    quick: bool = False,
    diagnostics: Path | None = None,
    engine: PlainEngineLike | None = None,
    analyzer: AnalyzerLike | None = None,
    dictionary: DictionaryStore | None = None,
) -> dict[str, object]:
    validation = validate_plain_corpus(corpus, allow_incomplete=quick)
    corpus_id, locked = _lock_files(corpus)
    sources = load_sources(corpus, locked)
    selected = _quick_annotations(corpus, locked) if quick else None
    plain_samples = load_plain_samples(corpus, "plain", locked, sources, selected)
    stress_samples = (
        () if quick else load_plain_samples(corpus, "plain_stress", locked, sources)
    )
    language_samples = () if quick else load_language_samples(corpus, locked, sources)
    if engine is None:
        engine = PaddleOcrEngine(
            PaddleDetector(_asset(assets, "korean_detection.onnx")),
            PaddleRecognizer(
                _asset(assets, "korean_recognition.onnx"),
                _asset(assets, "korean_characters.txt"),
            ),
        )
    if analyzer is None:
        dictionary = dictionary or SqliteDictionaryStore(_asset(assets, 'dictionary.sqlite3'))
        analyzer = KoreanAnalyzer(dictionary)
    if language_samples and dictionary is None:
        dictionary = SqliteDictionaryStore(_asset(assets, 'dictionary.sqlite3'))
    plain, outcomes = evaluate_plain(engine, analyzer, plain_samples)
    stress: dict[str, object] | None = None
    stress_outcomes: tuple[SampleOutcome, ...] = ()
    if stress_samples:
        stress, stress_outcomes = evaluate_plain(engine, analyzer, stress_samples)
        stress["release_blocking"] = False
    language = (
        evaluate_language(analyzer, dictionary, language_samples)
        if language_samples and dictionary is not None
        else None
    )
    if diagnostics is not None:
        _write_diagnostics(diagnostics, (*outcomes, *stress_outcomes))
    split_counts = Counter(sample.provenance.source_split for sample in plain_samples)
    evidence = {
        "provenance_complete": all(
            sample.provenance.oracle == PLAIN_ORACLE for sample in plain_samples
        ),
        "source_count": len(
            {
                source_id
                for sample in plain_samples
                for source_id in (
                    sample.provenance.source_id,
                    *sample.provenance.supporting_source_ids,
                )
            }
        ),
        "oracle_counts": dict(
            sorted(Counter(sample.provenance.oracle for sample in plain_samples).items())
        ),
        "split_counts": dict(sorted(split_counts.items())),
    }
    plain["evidence"] = evidence
    strata = plain["strata"]
    assert isinstance(strata, dict)
    primary = (
        plain["whole_eojeol_exact_pct"] >= OCR_PRIMARY
        and plain["fully_correct_first_popup_pct"] >= POPUP_PRIMARY
        and language is not None
        and language['fully_correct_first_popup_pct'] >= POPUP_PRIMARY
    )
    floors = (
        plain["whole_eojeol_exact_pct"] >= OCR_FLOOR
        and plain["fully_correct_first_popup_pct"] >= POPUP_FLOOR
        and _strata_meet_floors(strata)
        and language is not None
        and language['fully_correct_first_popup_pct'] >= POPUP_FLOOR
        and all(
            result['fully_correct_first_popup_pct'] >= POPUP_FLOOR
            for result in language['by_class'].values()
        )
    )
    correction = plain["false_promotion_rate_pct"] < FALSE_PROMOTION_MAX
    negative = (
        plain['negative_activation_rate_pct'] < FALSE_ACTIVATION_MAX
        and all(
            result['rate_pct'] < FALSE_ACTIVATION_MAX
            for result in plain['negative_activation_by_kind'].values()
        )
    )
    dictionary_conformance = bool(
        language and language['direct_krdict_conformance_pct'] == 100.0
    )
    latency = plain["latency_median_ms"] <= 500 and plain["latency_p95_ms"] <= 1_000
    release_evidence = (
        evidence["provenance_complete"]
        and set(evidence["oracle_counts"]) == {PLAIN_ORACLE}
        and set(split_counts) == {"test"}
    )
    complete = bool(validation["release_sample_counts"]) and not quick
    result: dict[str, object] = {
        "corpus_id": corpus_id,
        "profile": "plain-v1",
        "mode": "quick" if quick else "full",
        "plain": plain,
        "plain_stress": stress,
        'plain_language': language,
        "passes_primary_targets": primary,
        "passes_exceptional_floors": floors,
        "passes_correction_gate": correction,
        'passes_negative_activation_gate': negative,
        'passes_dictionary_conformance': dictionary_conformance,
        "passes_automated_pipeline_latency": latency,
        "release_eligible": bool(
            complete
            and primary
            and floors
            and correction
            and negative
            and dictionary_conformance
            and latency
            and release_evidence
        ),
    }
    return result
