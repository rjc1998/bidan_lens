"""Evaluate a licensed, hash-locked release corpus without exposing sample text."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from PIL import Image

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.models import AnalysisCandidate, BoundingBox, OcrDocument
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer

LOCK_NAME = "corpus.lock.json"
SOURCE_MANIFEST_NAME = "sources.json"
SCHEMA_VERSION = 2
PLAIN_SCHEMA_VERSION = 3
SOURCE_SCHEMA_VERSION = 1
EXPECTED_COUNTS = {"clean": 500, "subtitles": 300, "complex": 200, "morphology": 300}
PRIMARY_TARGETS = {"clean": 95.0, "subtitles": 90.0, "complex": 80.0}
RELEASE_ORACLES = {
    "clean": frozenset({"known-render", "published-annotation", "manual-reviewed"}),
    "subtitles": frozenset({"known-render", "published-annotation", "manual-reviewed"}),
    "complex": frozenset({"published-annotation", "manual-reviewed"}),
    "morphology": frozenset(
        {"published-annotation-independent-map", "published-manual", "manual-reviewed"}
    ),
}


class CorpusError(RuntimeError):
    """The corpus is incomplete, unsafe, or differs from its lock file."""


class OcrEngineLike(Protocol):
    def recognize(self, image: Image.Image) -> OcrDocument: ...


class AnalyzerLike(Protocol):
    def analyze(
        self, sentence: str, target_span: tuple[int, int], max_candidates: int = 5
    ) -> tuple[AnalysisCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    source_id: str
    license_id: str
    license_evidence: str
    annotation_basis: str
    allowed_oracles: frozenset[str]


@dataclass(frozen=True, slots=True)
class Provenance:
    source_id: str
    source_sample_id: str
    source_split: str
    oracle: str
    supporting_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpectedEojeol:
    text: str
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class ExpectedLine:
    text: str
    box: BoundingBox
    eojeols: tuple[ExpectedEojeol, ...]


@dataclass(frozen=True, slots=True)
class OcrSample:
    image: Path
    lines: tuple[ExpectedLine, ...]
    provenance: Provenance | None = None


@dataclass(frozen=True, slots=True)
class MorphologyCase:
    sentence: str
    target_span: tuple[int, int]
    expected_lemma: str
    expected_labels: frozenset[str]
    expected_interpreted_surface: str | None = None
    provenance: Provenance | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    unsafe_part = any(part in {"", ".", ".."} for part in relative.parts)
    if relative.is_absolute() or not relative.parts or unsafe_part:
        raise CorpusError("corpus contains an unsafe relative path")
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CorpusError("corpus path escapes its root") from error
    return candidate


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read corpus metadata file: {path.name}") from error
    if not isinstance(value, dict):
        raise CorpusError(f"corpus metadata must be an object: {path.name}")
    return value


def _lock_files(root: Path) -> tuple[str, dict[str, str]]:
    lock = _read_object(root / LOCK_NAME)
    if lock.get("schema_version") not in {SCHEMA_VERSION, PLAIN_SCHEMA_VERSION}:
        raise CorpusError("unsupported corpus lock schema")
    corpus_id = lock.get("corpus_id")
    files = lock.get("files")
    evidence = lock.get("license_evidence")
    source_manifest = lock.get("source_manifest")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise CorpusError("corpus lock has no stable corpus_id")
    if not isinstance(files, dict) or not files:
        raise CorpusError("corpus lock has no files")
    valid_evidence = (
        isinstance(evidence, list)
        and bool(evidence)
        and all(isinstance(item, str) for item in evidence)
    )
    if not valid_evidence:
        raise CorpusError("corpus lock must list license evidence")
    if source_manifest != SOURCE_MANIFEST_NAME:
        raise CorpusError("corpus lock must name sources.json as its source manifest")

    checked: dict[str, str] = {}
    for relative, expected_hash in files.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise CorpusError("corpus lock has an invalid file entry")
        path = _relative_path(root, relative)
        if not path.is_file() or _sha256(path) != expected_hash.lower():
            raise CorpusError(f"corpus file failed verification: {path.name}")
        checked[relative] = expected_hash.lower()
    for relative in evidence:
        path = _relative_path(root, relative)
        if relative not in checked or not path.is_file():
            raise CorpusError("license evidence must be present and hash-locked")
    if SOURCE_MANIFEST_NAME not in checked:
        raise CorpusError("source manifest must be present and hash-locked")
    return corpus_id, checked


def load_sources(root: Path, locked: dict[str, str]) -> dict[str, SourceEvidence]:
    path = root / SOURCE_MANIFEST_NAME
    if SOURCE_MANIFEST_NAME not in locked or not path.is_file():
        raise CorpusError("sources.json is not hash-locked")
    value = _read_object(path)
    if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise CorpusError("unsupported source manifest schema")
    source_values = value.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise CorpusError("source manifest must list at least one source")

    sources: dict[str, SourceEvidence] = {}
    for source_value in source_values:
        if not isinstance(source_value, dict):
            raise CorpusError("source manifest contains an invalid source")
        source_id = source_value.get("id")
        license_id = source_value.get("license_id")
        evidence = source_value.get("license_evidence")
        annotation_basis = source_value.get("annotation_basis")
        allowed_oracles = source_value.get("allowed_oracles")
        valid = (
            isinstance(source_id, str)
            and bool(source_id.strip())
            and isinstance(license_id, str)
            and bool(license_id.strip())
            and isinstance(evidence, str)
            and isinstance(annotation_basis, str)
            and bool(annotation_basis.strip())
            and isinstance(allowed_oracles, list)
            and bool(allowed_oracles)
            and all(isinstance(item, str) and item for item in allowed_oracles)
        )
        if not valid:
            raise CorpusError("source manifest contains an invalid source")
        if source_id in sources:
            raise CorpusError("source manifest contains duplicate source ids")
        if evidence not in locked or not _relative_path(root, evidence).is_file():
            raise CorpusError("source license evidence is not hash-locked")
        sources[source_id] = SourceEvidence(
            source_id,
            license_id,
            evidence,
            annotation_basis,
            frozenset(allowed_oracles),
        )
    return sources


def _provenance(value: Any, sources: dict[str, SourceEvidence] | None) -> Provenance | None:
    if sources is None:
        return None
    if not isinstance(value, dict):
        raise CorpusError("corpus sample has no provenance")
    source_id = value.get("source_id")
    sample_id = value.get("source_sample_id")
    source_split = value.get("source_split")
    oracle = value.get("oracle")
    supporting = value.get("supporting_source_ids", [])
    values = (source_id, sample_id, source_split, oracle)
    valid_supporting = isinstance(supporting, list) and all(
        isinstance(item, str) and item for item in supporting
    )
    if not all(isinstance(item, str) and item for item in values) or not valid_supporting:
        raise CorpusError("corpus sample has invalid provenance")
    if source_split.casefold() in {"train", "training"}:
        raise CorpusError("release evaluation cannot use a training split")
    source = sources.get(source_id)
    if source is None:
        raise CorpusError("corpus sample references an unknown source")
    if oracle not in source.allowed_oracles:
        raise CorpusError("corpus sample uses an unapproved annotation oracle")
    if any(item not in sources for item in supporting):
        raise CorpusError("corpus sample references an unknown supporting source")
    return Provenance(source_id, sample_id, source_split, oracle, tuple(sorted(set(supporting))))


def _box(value: Any) -> BoundingBox:
    if not isinstance(value, list) or len(value) != 4:
        raise CorpusError("annotation has an invalid box")
    try:
        box = BoundingBox(*(float(item) for item in value))
    except (TypeError, ValueError) as error:
        raise CorpusError("annotation has an invalid box") from error
    if box.area <= 0:
        raise CorpusError("annotation boxes must have positive area")
    return box


def load_ocr_samples(
    root: Path,
    category: str,
    locked: dict[str, str],
    sources: dict[str, SourceEvidence] | None = None,
) -> tuple[OcrSample, ...]:
    category_root = root / category
    annotations = sorted(category_root.glob("*.json")) if category_root.is_dir() else []
    samples: list[OcrSample] = []
    sample_ids: set[tuple[str, str]] = set()
    for annotation in annotations:
        annotation_key = annotation.relative_to(root).as_posix()
        if annotation_key not in locked:
            raise CorpusError("an OCR annotation is not hash-locked")
        value = _read_object(annotation)
        image_value = value.get("image")
        lines_value = value.get("lines")
        if not isinstance(image_value, str) or not isinstance(lines_value, list) or not lines_value:
            raise CorpusError("OCR annotation has an invalid sample structure")
        image = _relative_path(annotation.parent, image_value)
        try:
            image_key = image.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise CorpusError("OCR image escapes the corpus root") from error
        if image_key not in locked or not image.is_file():
            raise CorpusError("OCR image is not hash-locked")

        lines: list[ExpectedLine] = []
        for line_value in lines_value:
            if not isinstance(line_value, dict):
                raise CorpusError("OCR annotation has an invalid line")
            text = line_value.get("text")
            eojeol_values = line_value.get("eojeols")
            if not isinstance(text, str) or not text or not isinstance(eojeol_values, list):
                raise CorpusError("OCR annotation has an invalid line")
            eojeols: list[ExpectedEojeol] = []
            for eojeol_value in eojeol_values:
                valid_eojeol = isinstance(eojeol_value, dict) and isinstance(
                    eojeol_value.get("text"), str
                )
                if not valid_eojeol:
                    raise CorpusError("OCR annotation has an invalid eojeol")
                eojeols.append(ExpectedEojeol(eojeol_value["text"], _box(eojeol_value.get("box"))))
            if not eojeols:
                raise CorpusError("each annotated line needs at least one eojeol")
            lines.append(ExpectedLine(text, _box(line_value.get("box")), tuple(eojeols)))
        provenance = _provenance(value.get("provenance"), sources)
        if provenance is not None:
            identity = (provenance.source_id, provenance.source_sample_id)
            if identity in sample_ids:
                raise CorpusError("OCR category contains a duplicate source sample")
            sample_ids.add(identity)
        samples.append(OcrSample(image, tuple(lines), provenance))
    return tuple(samples)


def load_morphology_cases(
    root: Path,
    locked: dict[str, str],
    sources: dict[str, SourceEvidence] | None = None,
) -> tuple[MorphologyCase, ...]:
    path = root / "morphology.jsonl"
    if path.relative_to(root).as_posix() not in locked or not path.is_file():
        raise CorpusError("morphology.jsonl is not hash-locked")
    cases: list[MorphologyCase] = []
    sample_ids: set[tuple[str, str]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CorpusError("cannot read morphology corpus") from error
    for raw in lines:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CorpusError("morphology corpus contains invalid JSON") from error
        if not isinstance(value, dict):
            raise CorpusError("morphology case must be an object")
        sentence = value.get("sentence")
        span = value.get("target_span")
        lemma = value.get("expected_lemma")
        labels = value.get("expected_labels")
        correction = value.get("expected_interpreted_surface")
        if (
            not isinstance(sentence, str)
            or not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(item, int) for item in span)
            or not isinstance(lemma, str)
            or not isinstance(labels, list)
            or not all(isinstance(item, str) for item in labels)
            or (correction is not None and not isinstance(correction, str))
        ):
            raise CorpusError("morphology case has an invalid structure")
        start, end = span
        if start < 0 or end <= start or end > len(sentence):
            raise CorpusError("morphology case has an invalid target span")
        provenance = _provenance(value.get("provenance"), sources)
        if provenance is not None:
            identity = (provenance.source_id, provenance.source_sample_id)
            if identity in sample_ids:
                raise CorpusError("morphology corpus contains a duplicate source sample")
            sample_ids.add(identity)
        cases.append(
            MorphologyCase(
                sentence,
                (start, end),
                lemma,
                frozenset(labels),
                correction,
                provenance,
            )
        )
    return tuple(cases)


def _overlap_score(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    return intersection / min(first.area, second.area) if intersection else 0.0


def _match(
    expected: BoundingBox, predicted: list[tuple[str, BoundingBox]]
) -> tuple[str, int] | None:
    candidates = [
        (_overlap_score(expected, box), index, text) for index, (text, box) in enumerate(predicted)
    ]
    if not candidates:
        return None
    score, index, text = max(candidates)
    return (text, index) if score >= 0.25 else None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * (proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5
    margin /= denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _evidence_summary(
    items: tuple[OcrSample, ...] | tuple[MorphologyCase, ...],
) -> dict[str, object]:
    provenance = [item.provenance for item in items if item.provenance is not None]
    all_source_ids = {
        source_id
        for item in provenance
        for source_id in (item.source_id, *item.supporting_source_ids)
    }
    return {
        "provenance_complete": len(provenance) == len(items),
        "source_count": len(all_source_ids),
        "oracle_counts": dict(sorted(Counter(item.oracle for item in provenance).items())),
    }


def evaluate_ocr(engine: OcrEngineLike, samples: tuple[OcrSample, ...]) -> dict[str, float | int]:
    exact_eojeols = exact_lines = missing_eojeols = total_eojeols = total_lines = 0
    durations: list[float] = []
    for sample in samples:
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        started = time.perf_counter()
        document = engine.recognize(image)
        durations.append((time.perf_counter() - started) * 1000)
        predicted_eojeols = [
            (item.text, item.box) for line in document.lines for item in line.eojeols
        ]
        predicted_lines = [(line.text, line.box) for line in document.lines]
        for expected_line in sample.lines:
            total_lines += 1
            line_match = _match(expected_line.box, predicted_lines)
            if line_match is not None:
                text, index = line_match
                exact_lines += text == expected_line.text
                predicted_lines.pop(index)
            for expected_eojeol in expected_line.eojeols:
                total_eojeols += 1
                match = _match(expected_eojeol.box, predicted_eojeols)
                if match is None:
                    missing_eojeols += 1
                    continue
                text, index = match
                exact_eojeols += text == expected_eojeol.text
                predicted_eojeols.pop(index)
    if not samples or not total_eojeols or not total_lines:
        raise CorpusError("OCR category contains no measurable annotations")
    ci_low, ci_high = _wilson_interval(exact_eojeols, total_eojeols)
    return {
        "samples": len(samples),
        "eojeols": total_eojeols,
        "whole_eojeol_exact_pct": round(exact_eojeols / total_eojeols * 100, 2),
        "whole_eojeol_exact_ci95_low_pct": round(ci_low * 100, 2),
        "whole_eojeol_exact_ci95_high_pct": round(ci_high * 100, 2),
        "line_exact_pct": round(exact_lines / total_lines * 100, 2),
        "missing_eojeol_pct": round(missing_eojeols / total_eojeols * 100, 2),
        "latency_median_ms": round(statistics.median(durations), 2),
        "latency_p95_ms": round(_p95(durations), 2),
    }


def evaluate_morphology(
    analyzer: AnalyzerLike, cases: tuple[MorphologyCase, ...]
) -> dict[str, float | int]:
    correct = definitions = combined = corrections = promotions = false_promotions = 0
    candidate_definitions = candidate_combined = 0
    durations: list[float] = []
    for case in cases:
        started = time.perf_counter()
        candidates = analyzer.analyze(case.sentence, case.target_span)
        durations.append((time.perf_counter() - started) * 1000)
        first = candidates[0] if candidates else None
        labels = {feature.label for feature in first.features} if first else set()
        is_correct = bool(
            first and first.lemma == case.expected_lemma and case.expected_labels <= labels
        )
        has_definition = bool(first and first.dictionary_entries)
        has_candidate_definition = any(candidate.dictionary_entries for candidate in candidates)
        has_correct_candidate_definition = any(
            candidate.lemma == case.expected_lemma and candidate.dictionary_entries
            for candidate in candidates
        )
        correct += is_correct
        definitions += has_definition
        combined += is_correct and has_definition
        candidate_definitions += has_candidate_definition
        candidate_combined += has_correct_candidate_definition
        corrections += any(candidate.interpreted_surface is not None for candidate in candidates)
        if first and first.interpreted_surface is not None:
            promotions += 1
            false_promotions += first.interpreted_surface != case.expected_interpreted_surface
    if not cases:
        raise CorpusError("morphology corpus contains no cases")
    ci_low, ci_high = _wilson_interval(correct, len(cases))
    return {
        "samples": len(cases),
        "correct_lemma_and_breakdown_first_pct": round(correct / len(cases) * 100, 2),
        "correct_lemma_and_breakdown_first_ci95_low_pct": round(ci_low * 100, 2),
        "correct_lemma_and_breakdown_first_ci95_high_pct": round(ci_high * 100, 2),
        "first_candidate_definition_pct": round(definitions / len(cases) * 100, 2),
        "combined_first_result_pct": round(combined / len(cases) * 100, 2),
        "definition_available_in_candidates_pct": round(
            candidate_definitions / len(cases) * 100, 2
        ),
        "correct_lemma_with_definition_in_candidates_pct": round(
            candidate_combined / len(cases) * 100, 2
        ),
        "marked_correction_cases": corrections,
        "marked_correction_promotions": promotions,
        "false_promotions": false_promotions,
        "false_promotion_rate_pct": round(false_promotions / len(cases) * 100, 3),
        "latency_median_ms": round(statistics.median(durations), 2),
        "latency_p95_ms": round(_p95(durations), 2),
    }


def run(assets: Path, corpus: Path, category: str, allow_incomplete: bool) -> dict[str, object]:
    corpus_id, locked = _lock_files(corpus)
    sources = load_sources(corpus, locked)
    selected = tuple(EXPECTED_COUNTS) if category == "all" else (category,)
    results: dict[str, object] = {
        "corpus_id": corpus_id,
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    engine: PaddleOcrEngine | None = None
    analyzer: KoreanAnalyzer | None = None
    release_eligible = True
    for selected_category in selected:
        if selected_category == "morphology":
            cases = load_morphology_cases(corpus, locked, sources)
            if analyzer is None:
                analyzer = KoreanAnalyzer(SqliteDictionaryStore(assets / "dictionary.sqlite3"))
            result = evaluate_morphology(analyzer, cases)
            passes = (
                result["correct_lemma_and_breakdown_first_pct"] >= 90.0
                and result["false_promotion_rate_pct"] < 0.5
            )
        else:
            samples = load_ocr_samples(corpus, selected_category, locked, sources)
            if engine is None:
                engine = PaddleOcrEngine(
                    PaddleDetector(assets / "korean_detection.onnx"),
                    PaddleRecognizer(
                        assets / "korean_recognition.onnx", assets / "korean_characters.txt"
                    ),
                )
            result = evaluate_ocr(engine, samples)
            passes = result["whole_eojeol_exact_pct"] >= PRIMARY_TARGETS[selected_category]
        complete = result["samples"] == EXPECTED_COUNTS[selected_category]
        if not complete and not allow_incomplete:
            raise CorpusError(
                f"{selected_category} requires exactly {EXPECTED_COUNTS[selected_category]} samples"
            )
        evidence = _evidence_summary(
            cases if selected_category == "morphology" else samples
        )
        evidence_oracles = set(evidence["oracle_counts"])
        passes_evidence = bool(
            evidence["provenance_complete"]
            and evidence_oracles
            and evidence_oracles <= RELEASE_ORACLES[selected_category]
        )
        result["complete_release_sample_count"] = complete
        result["passes_primary_targets"] = passes
        result["passes_evidence_contract"] = passes_evidence
        result["evidence"] = evidence
        results[selected_category] = result
        release_eligible &= bool(complete and passes and passes_evidence)
    results["release_eligible"] = release_eligible
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a private, licensed, hash-locked corpus")
    parser.add_argument("assets", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--profile", choices=("legacy", "plain-v1"), default="legacy")
    parser.add_argument("--category", choices=(*EXPECTED_COUNTS, "all"), default="all")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="permit development runs while keeping release_eligible false",
    )
    parser.add_argument("--quick", action="store_true", help="run the locked 200-sample dev subset")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="write local failure sample ids and stages without text or pixels",
    )
    arguments = parser.parse_args()
    if arguments.profile == "plain-v1":
        from benchmarks.plain_evaluator import run_plain

        if arguments.allow_incomplete:
            parser.error("plain-v1 uses --quick instead of --allow-incomplete")
        results = run_plain(
            arguments.assets,
            arguments.corpus,
            quick=arguments.quick,
            diagnostics=arguments.diagnostics,
        )
    else:
        if arguments.quick or arguments.diagnostics:
            parser.error("--quick and --diagnostics require --profile plain-v1")
        results = run(
            arguments.assets, arguments.corpus, arguments.category, arguments.allow_incomplete
        )
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
