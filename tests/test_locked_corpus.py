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
    load_sources,
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


@pytest.mark.parametrize('profile', ['legacy', 'plain-v1'])
@pytest.mark.parametrize('actual_id', ['older-dev', 'release', None])
def test_cli_rejects_wrong_corpus_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, profile: str, actual_id,
) -> None:
    from benchmarks import locked_corpus, plain_evaluator

    (tmp_path / 'corpus.lock.json').write_text(
        json.dumps({'corpus_id': actual_id}), encoding='utf-8'
    )

    def unexpected_run(*args, **kwargs):
        pytest.fail('evaluation must not start for a mismatched corpus')

    monkeypatch.setattr(locked_corpus, 'run', unexpected_run)
    monkeypatch.setattr(plain_evaluator, 'run_plain', unexpected_run)
    monkeypatch.setattr('sys.argv', [
        'locked_corpus', 'missing-assets', str(tmp_path), '--profile', profile,
        '--expected-corpus-id', 'intended-dev',
    ])

    with pytest.raises(SystemExit) as error:
        locked_corpus.main()

    assert error.value.code == 2
    assert 'corpus identity does not match' in capsys.readouterr().err


@pytest.mark.parametrize('profile', ['legacy', 'plain-v1'])
def test_cli_matching_identity_still_runs_normal_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str,
) -> None:
    from benchmarks import locked_corpus

    (tmp_path / 'corpus.lock.json').write_text(
        json.dumps({'corpus_id': 'intended-dev'}), encoding='utf-8'
    )
    monkeypatch.setattr('sys.argv', [
        'locked_corpus', 'missing-assets', str(tmp_path), '--profile', profile,
        '--expected-corpus-id', 'intended-dev',
    ])

    with pytest.raises(CorpusError, match='corpus lock'):
        locked_corpus.main()


@pytest.mark.parametrize('profile', ['legacy', 'plain-v1'])
@pytest.mark.parametrize('fails', [False, True])
def test_cli_report_preserves_previous_result_until_evaluation_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    profile: str, fails: bool,
) -> None:
    from benchmarks import locked_corpus, plain_evaluator

    output = tmp_path / 'report.json'
    output.write_bytes(b'previous report')
    result = {'corpus_id': 'fixture', 'release_eligible': False}

    def evaluate(*args, **kwargs):
        assert output.read_bytes() == b'previous report'
        if fails:
            raise CorpusError('fixture evaluation failed')
        return result

    monkeypatch.setattr(locked_corpus, 'run', evaluate)
    monkeypatch.setattr(plain_evaluator, 'run_plain', evaluate)
    monkeypatch.setattr('sys.argv', [
        'locked_corpus', str(tmp_path / 'assets'), str(tmp_path / 'corpus'),
        '--profile', profile, '--output', str(output),
    ])
    if fails:
        with pytest.raises(CorpusError, match='fixture evaluation failed'):
            locked_corpus.main()
        assert output.read_bytes() == b'previous report'
    else:
        locked_corpus.main()
        expected = (json.dumps(result, ensure_ascii=True, indent=2) + '\n').encode('utf-8')
        assert output.read_bytes() == expected
    assert capsys.readouterr().out == ''
    assert list(tmp_path.iterdir()) == [output]


def test_report_replace_failure_preserves_previous_file_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.locked_corpus import _write_report

    output = tmp_path / 'report.json'
    output.write_bytes(b'previous report')

    def fail_replace(source: Path, target: Path):
        assert source.parent == target.parent
        assert source.read_bytes() == b'{}\n'
        raise PermissionError('fixture replacement denied')

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(PermissionError, match='fixture replacement denied'):
        _write_report(output, '{}')
    assert output.read_bytes() == b'previous report'
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('destination', ['corpus', 'assets', 'diagnostics'])
def test_cli_rejects_report_destination_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, destination: str,
) -> None:
    from benchmarks import locked_corpus, plain_evaluator

    output = tmp_path / destination / 'report.json'
    output.parent.mkdir()
    output.write_bytes(b'preserved')

    def unexpected_run(*args, **kwargs):
        pytest.fail('evaluation must not start for an output collision')

    monkeypatch.setattr(plain_evaluator, 'run_plain', unexpected_run)
    monkeypatch.setattr('sys.argv', [
        'locked_corpus', str(tmp_path / 'assets'), str(tmp_path / 'corpus'),
        '--profile', 'plain-v1', '--output', str(output),
        '--diagnostics', str(tmp_path / 'diagnostics' / 'report.json'),
    ])
    with pytest.raises(SystemExit) as error:
        locked_corpus.main()
    assert error.value.code == 2
    assert '--output' in capsys.readouterr().err
    assert output.read_bytes() == b'preserved'


def _source_manifest(tmp_path: Path, evidence: str = "LICENSE.txt") -> Path:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "source",
                        "license_id": "test-license",
                        "license_evidence": evidence,
                        "annotation_basis": "test annotations",
                        "allowed_oracles": ["known-render"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_lock_requires_hash_locked_license_evidence(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("redistribution permitted", encoding="utf-8")
    sources_path = _source_manifest(tmp_path)
    (tmp_path / "corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "corpus_id": "reviewed-v1",
                "source_manifest": "sources.json",
                "license_evidence": ["LICENSE.txt"],
                "files": {
                    "LICENSE.txt": _hash(license_path),
                    "sources.json": _hash(sources_path),
                },
            }
        ),
        encoding="utf-8",
    )

    corpus_id, files = _lock_files(tmp_path)
    sources = load_sources(tmp_path, files)

    assert corpus_id == "reviewed-v1"
    assert "LICENSE.txt" in files
    assert sources["source"].allowed_oracles == frozenset({"known-render"})


def test_lock_rejects_changed_file(tmp_path: Path) -> None:
    license_path = tmp_path / "LICENSE.txt"
    license_path.write_text("first", encoding="utf-8")
    sources_path = _source_manifest(tmp_path)
    (tmp_path / "corpus.lock.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "corpus_id": "reviewed-v1",
                "source_manifest": "sources.json",
                "license_evidence": ["LICENSE.txt"],
                "files": {
                    "LICENSE.txt": _hash(license_path),
                    "sources.json": _hash(sources_path),
                },
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
    assert result["whole_eojeol_exact_ci95_low_pct"] < 100.0
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
    assert result["correct_lemma_with_definition_in_candidates_pct"] == 100.0


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
