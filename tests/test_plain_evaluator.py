import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.locked_corpus import CorpusError
from benchmarks.plain_corpus import PLAIN_ORACLE
from benchmarks.plain_evaluator import (
    _functional_context,
    evaluate_plain,
    load_plain_samples,
    lock_plain_corpus,
    run_plain,
    validate_plain_corpus,
)
from bidan_lens.models import (
    AnalysisCandidate,
    BoundingBox,
    DictionaryEntry,
    DictionarySense,
    LearnerFeature,
    LexicalComponent,
    OcrDocument,
    OcrEojeol,
    OcrLine,
)

SENTENCE = "오늘 (어디에서) 만나요"
TARGET = "어디에서"
TARGET_SPAN = (4, 8)
TARGET_BOX = BoundingBox(50, 10, 110, 30)


class FakeOcr:
    def __init__(self, target: str = TARGET, sentence: str = SENTENCE) -> None:
        self.target = target
        self.sentence = sentence

    def recognize(self, _image: Image.Image) -> OcrDocument:
        eojeols = (
            OcrEojeol("오늘", BoundingBox(5, 10, 35, 30), 1.0, 0, 2),
            OcrEojeol(self.target, TARGET_BOX, 1.0, *TARGET_SPAN),
            OcrEojeol("만나요", BoundingBox(125, 10, 175, 30), 1.0, 10, 13),
        )
        return OcrDocument(
            (OcrLine(self.sentence, BoundingBox(5, 10, 175, 30), 1.0, eojeols),), 0.0
        )


@dataclass
class FakeAnalyzer:
    candidates: tuple[AnalysisCandidate, ...]

    def analyze(
        self, _sentence: str, _span: tuple[int, int], max_candidates: int = 5
    ) -> tuple[AnalysisCandidate, ...]:
        return self.candidates[:max_candidates]


def _entry(reverse: bool = False) -> DictionaryEntry:
    senses = (
        DictionarySense("where", 1),
        DictionarySense("at what place", 2),
    )
    if reverse:
        senses = tuple(reversed(senses))
    return DictionaryEntry("entry-1", "어디", None, None, None, senses)


def _candidate(*, correct: bool = True, reverse: bool = False) -> AnalysisCandidate:
    entry = _entry(reverse)
    return AnalysisCandidate(
        TARGET,
        "어디" if correct else "다르다",
        1.0,
        features=(LearnerFeature("location particle", "marks a location"),),
        dictionary_entries=(entry,),
        lexical_components=(
            LexicalComponent('어디', '어디', 'noun', (entry,)),
        ),
    )


def _annotation(sample_id: str, image: str, source_sample_id: str, stress: bool) -> dict:
    size = 10 if stress else 12
    return {
        "schema_version": 4,
        "sample_id": sample_id,
        "image": image,
        "lines": [
            {
                "text": SENTENCE,
                "box": [5, 10, 175, 30],
                "eojeols": [
                    {"text": "오늘", "box": [5, 10, 35, 30]},
                    {"text": TARGET, "box": [50, 10, 110, 30]},
                    {"text": "만나요", "box": [125, 10, 175, 30]},
                ],
            }
        ],
        "target": {
            "text": TARGET,
            "box": [50, 10, 110, 30],
            "pointer": [80, 20],
            "sentence": SENTENCE,
            "sentence_span": list(TARGET_SPAN),
            "expected_lemma": "어디",
            "expected_labels": ["location particle"],
            "expected_dictionary_entries": [
                {
                    "entry_id": "entry-1",
                    "headword": "어디",
                    "senses": [
                        {"order": 1, "definition": "where"},
                        {"order": 2, "definition": "at what place"},
                    ],
                }
            ],
            'expected_components': [
                {
                    'surface': '어디',
                    'lemma': '어디',
                    'learner_role': 'noun',
                    'expected_dictionary_entries': [
                        {
                            'entry_id': 'entry-1',
                            'headword': '어디',
                            'senses': [
                                {'order': 1, 'definition': 'where'},
                                {'order': 2, 'definition': 'at what place'},
                            ],
                        }
                    ],
                }
            ],
            'expected_spacing': None,
            'language_class': None,
            "target_class": "particle",
        },
        'negative_probes': [
            {'kind': 'english', 'pointer': [190, 5]},
            {'kind': 'blank', 'pointer': [190, 45]},
            {'kind': 'whitespace', 'pointer': [42, 20]},
            {'kind': 'punctuation', 'pointer': [45, 20]},
            {'kind': 'near-miss', 'pointer': [80, 40]},
        ],
        "render": {
            "renderer": "desktop",
            "renderer_version": "test-desktop",
            "font": "malgun-gothic",
            "font_sha256": "0" * 64,
            "size_px": size,
            "weight": 400,
            "scale_percent": 100,
            "theme": "light",
            "layout": "single-line",
            "punctuation": "brackets",
            "stress": stress,
        },
        "provenance": {
            "source_id": "source",
            "source_sample_id": source_sample_id,
            "source_split": "test",
            "oracle": PLAIN_ORACLE,
        },
    }


def _corpus(tmp_path: Path, *, duplicate: bool = False) -> Path:
    root = tmp_path / "corpus"
    (root / "plain").mkdir(parents=True)
    (root / "plain_stress").mkdir()
    (root / "LICENSE.txt").write_text("test license", encoding="utf-8")
    (root / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "source",
                        "license_id": "test",
                        "license_evidence": "LICENSE.txt",
                        "annotation_basis": "known rendering",
                        "allowed_oracles": [PLAIN_ORACLE],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_lock = root / "plain_sources.lock.json"
    source_lock.write_text(
        json.dumps({"schema_version": 1, "artifacts": []}), encoding="utf-8"
    )
    source_lock_hash = hashlib.sha256(source_lock.read_bytes()).hexdigest()
    (root / "acquisition.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lock": "plain_sources.lock.json",
                "source_lock_sha256": source_lock_hash,
            }
        ),
        encoding="utf-8",
    )
    (root / "renderer.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "renderers": {"browser": "test-browser", "desktop": "test-desktop"},
                "required_sizes_px": [12, 14, 16, 18, 20, 24, 32, 40],
                "stress_size_px": 10,
                "scales_percent": [100, 125, 150, 200],
                "punctuation_classes": [
                    "natural",
                    "terminal",
                    "comma-colon",
                    "quotes",
                    "brackets",
                    "ellipsis",
                    "dash-slash",
                    "mixed",
                ],
            }
        ),
        encoding="utf-8",
    )
    for category, stress in (("plain", False), ("plain_stress", True)):
        Image.new("RGB", (200, 50), "white").save(root / category / "0001.png")
        source_id = "source-test-1" if duplicate else f"source-test-{category}"
        value = _annotation(f"{category}-1", "0001.png", source_id, stress)
        (root / category / "0001.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
    (root / "quick.json").write_text(
        json.dumps({"schema_version": 1, "samples": ["plain/0001.json"]}),
        encoding="utf-8",
    )
    return root


def _load(root: Path):
    lock_plain_corpus(root, "plain-test", allow_incomplete=True)
    from benchmarks.locked_corpus import _lock_files, load_sources

    _, files = _lock_files(root)
    return load_plain_samples(root, "plain", files, load_sources(root, files))


def test_plain_pipeline_checks_complete_eojeol_breakdown_and_dictionary_order(
    tmp_path: Path,
) -> None:
    samples = _load(_corpus(tmp_path))

    result, outcomes = evaluate_plain(FakeOcr(), FakeAnalyzer((_candidate(),)), samples)

    assert result["whole_eojeol_exact_pct"] == 100.0
    assert result["target_selection_pct"] == 100.0
    assert result["component_lemma_breakdown_first_pct"] == 100.0
    assert result["exact_krdict_fidelity_first_pct"] == 100.0
    assert result["fully_correct_first_popup_pct"] == 100.0
    assert result['functional_context_accuracy_pct'] == 100.0
    assert result['negative_activation_rate_pct'] == 0.0
    assert outcomes[0].failed_stage is None


def test_functional_context_ignores_edges_but_preserves_internal_structure() -> None:
    expected = '어제, “학교에서” 12:30에 갔어요.'
    expected_span = (5, 9)

    assert _functional_context(
        '어제 학교에서 12:30에 갔어요',
        (3, 7),
        expected,
        expected_span,
    )
    assert not _functional_context(
        '어제 학교에서 1230에 갔어요',
        (3, 7),
        expected,
        expected_span,
    )
    assert not _functional_context(
        '어제 학교에서12:30에 갔어요',
        (3, 7),
        expected,
        expected_span,
    )


def test_plain_pipeline_reports_dictionary_order_failure(tmp_path: Path) -> None:
    samples = _load(_corpus(tmp_path))

    result, outcomes = evaluate_plain(
        FakeOcr(), FakeAnalyzer((_candidate(reverse=True),)), samples
    )

    assert result["component_lemma_breakdown_first_pct"] == 100.0
    assert result["exact_krdict_fidelity_first_pct"] == 0.0
    assert result["fully_correct_first_popup_pct"] == 0.0
    assert outcomes[0].failed_stage == "dictionary"


def test_plain_pipeline_counts_alternative_candidate_recovery(tmp_path: Path) -> None:
    samples = _load(_corpus(tmp_path))

    result, outcomes = evaluate_plain(
        FakeOcr(), FakeAnalyzer((_candidate(correct=False), _candidate())), samples
    )

    assert result["fully_correct_first_popup_pct"] == 0.0
    assert result["alternative_candidate_recovery_pct"] == 100.0
    assert outcomes[0].failed_stage == "analysis"


@pytest.mark.parametrize(
    ("engine", "failed_stage"),
    [
        (FakeOcr("어디"), "target"),
        (FakeOcr(sentence="내일 (어디에서) 만나요"), "context"),
    ],
)
def test_plain_pipeline_identifies_ocr_failure_stage(
    tmp_path: Path, engine: FakeOcr, failed_stage: str
) -> None:
    samples = _load(_corpus(tmp_path))

    _, outcomes = evaluate_plain(engine, FakeAnalyzer((_candidate(),)), samples)

    assert outcomes[0].failed_stage == failed_stage
    assert outcomes[0].full_popup is False


def test_plain_pipeline_counts_marked_correction_false_promotion(tmp_path: Path) -> None:
    samples = _load(_corpus(tmp_path))
    candidate = AnalysisCandidate(
        TARGET,
        "어디",
        1.0,
        features=(LearnerFeature("location particle", "marks a location"),),
        dictionary_entries=(_entry(),),
        interpreted_surface="어디 에서",
    )

    result, _ = evaluate_plain(FakeOcr(), FakeAnalyzer((candidate,)), samples)

    assert result["false_promotions"] == 1
    assert result["false_promotion_rate_pct"] == 100.0


def test_plain_quick_run_excludes_nonblocking_stress_tier(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    lock_plain_corpus(root, "plain-test", allow_incomplete=True)

    result = run_plain(
        tmp_path / "unused-assets",
        root,
        quick=True,
        engine=FakeOcr(),
        analyzer=FakeAnalyzer((_candidate(),)),
    )

    assert result["plain"]["samples"] == 1
    assert result["plain_stress"] is None
    assert result["release_eligible"] is False


def test_plain_lock_detects_tampering_and_cross_category_duplicates(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    lock_plain_corpus(root, "plain-test", allow_incomplete=True)
    (root / "plain/0001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="failed verification"):
        validate_plain_corpus(root, allow_incomplete=True)

    duplicate_root = _corpus(tmp_path / "duplicate", duplicate=True)
    with pytest.raises(CorpusError, match="share a source sample"):
        lock_plain_corpus(duplicate_root, "plain-test", allow_incomplete=True)


def test_plain_lock_rejects_training_provenance(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    annotation = root / "plain/0001.json"
    value = json.loads(annotation.read_text(encoding="utf-8"))
    value["provenance"]["source_split"] = "train"
    annotation.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CorpusError, match="training split"):
        lock_plain_corpus(root, "plain-test", allow_incomplete=True)
