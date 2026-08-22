import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.locked_corpus import (
    CorpusError,
    ExpectedEojeol,
    ExpectedLine,
    MorphologyCase,
    OcrSample,
    _lock_files,
    evaluate_morphology,
    evaluate_ocr,
)
from bidan_lens.models import (
    AnalysisCandidate,
    BoundingBox,
    DictionaryEntry,
    DictionarySense,
    LearnerFeature,
    OcrDocument,
    OcrEojeol,
    OcrLine,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lock_requires_hash_locked_license_evidence(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("redistribution permitted", encoding="utf-8")
    (tmp_path / "corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_id": "reviewed-v1",
                "license_evidence": ["LICENSE.txt"],
                "files": {"LICENSE.txt": _hash(license_path)},
            }
        ),
        encoding="utf-8",
    )

    corpus_id, files = _lock_files(tmp_path)

    assert corpus_id == "reviewed-v1"
    assert "LICENSE.txt" in files


def test_lock_rejects_changed_file(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("first", encoding="utf-8")
    (tmp_path / "corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_id": "reviewed-v1",
                "license_evidence": ["LICENSE.txt"],
                "files": {"LICENSE.txt": _hash(license_path)},
            }
        ),
        encoding="utf-8",
    )
    license_path.write_text("changed", encoding="utf-8")

    with pytest.raises(CorpusError, match="failed verification"):
        _lock_files(tmp_path)


class FakeOcr:
    def recognize(self, _image: Image.Image) -> OcrDocument:
        box = BoundingBox(0, 0, 90, 30)
        eojeol = OcrEojeol("어디에서", box, 0.99, 0, 4)
        return OcrDocument((OcrLine("어디에서", box, 0.99, (eojeol,)),), 0.0)


def test_ocr_evaluation_returns_aggregate_exact_metrics(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (100, 40), "white").save(image)
    box = BoundingBox(0, 0, 90, 30)
    sample = OcrSample(image, (ExpectedLine("어디에서", box, (ExpectedEojeol("어디에서", box),)),))

    result = evaluate_ocr(FakeOcr(), (sample,))

    assert result["samples"] == 1
    assert result["whole_eojeol_exact_pct"] == 100.0
    assert result["line_exact_pct"] == 100.0
    assert result["missing_eojeol_pct"] == 0.0
    assert "어디에서" not in json.dumps(result, ensure_ascii=False)


@dataclass
class FakeAnalyzer:
    candidates: tuple[AnalysisCandidate, ...]

    def analyze(self, _sentence: str, _span: tuple[int, int], max_candidates: int = 5):
        return self.candidates[:max_candidates]


def _entry() -> DictionaryEntry:
    return DictionaryEntry("1", "먹다", "verb", None, "beginner", (DictionarySense("to eat"),))


def test_morphology_evaluation_counts_false_promotions_over_entire_corpus() -> None:
    candidate = AnalysisCandidate(
        "먹고싶어요",
        "먹다",
        1.0,
        features=(LearnerFeature("polite style", "polite"),),
        dictionary_entries=(_entry(),),
        interpreted_surface="먹고 싶어요",
    )
    cases = tuple(
        MorphologyCase("먹고싶어요", (0, 5), "먹다", frozenset({"polite style"}))
        for _ in range(300)
    )

    result = evaluate_morphology(FakeAnalyzer((candidate,)), cases)

    assert result["false_promotions"] == 300
    assert result["false_promotion_rate_pct"] == 100.0
    assert result["correct_lemma_and_breakdown_first_pct"] == 100.0


def test_expected_marked_correction_is_not_false() -> None:
    candidate = AnalysisCandidate("먹고싶어요", "먹다", 1.0, interpreted_surface="먹고 싶어요")
    case = MorphologyCase(
        "먹고싶어요",
        (0, 5),
        "먹다",
        frozenset(),
        expected_interpreted_surface="먹고 싶어요",
    )

    result = evaluate_morphology(FakeAnalyzer((candidate,)), (case,))

    assert result["marked_correction_promotions"] == 1
    assert result["false_promotions"] == 0
    assert result["false_promotion_rate_pct"] == 0.0
