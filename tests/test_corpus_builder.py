import json
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.corpus_builder import (
    import_aihub,
    import_ud,
    lock_corpus,
    render_ocr,
    validate_corpus,
)
from benchmarks.locked_corpus import CorpusError


def _manifest(root: Path, allowed_oracles: list[str]) -> None:
    licenses = root / "LICENSES"
    licenses.mkdir(parents=True)
    (licenses / "source.txt").write_text("test evidence", encoding="utf-8")
    (root / "sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "source",
                        "license_id": "test-license",
                        "license_evidence": "LICENSES/source.txt",
                        "annotation_basis": "published test fixture",
                        "allowed_oracles": allowed_oracles,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _annotation(image_name: str, oracle: str) -> dict[str, object]:
    return {
        "image": image_name,
        "lines": [
            {
                "text": "어디에서",
                "box": [1, 1, 90, 30],
                "eojeols": [{"text": "어디에서", "box": [1, 1, 90, 30]}],
            }
        ],
        "provenance": {
            "source_id": "source",
            "source_sample_id": image_name,
            "source_split": "test",
            "oracle": oracle,
            "supporting_source_ids": [],
        },
    }


def test_import_ud_derives_lemma_and_labels_without_production_analyzer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _manifest(corpus, ["published-annotation-independent-map"])
    conllu = tmp_path / "sample.conllu"
    conllu.write_text(
        "\n".join(
            (
                "# sent_id = sample-1",
                "# text = 나는 밥을 먹었다.",
                "1\t나는\t나+는\tPRON\tnpp+jxt\t_\t3\tnsubj\t_\t_",
                "2\t밥을\t밥+을\tNOUN\tncn+jco\t_\t3\tobj\t_\t_",
                "3\t먹었다\t먹+었+다\tVERB\tpvg+ep+ef\t_\t0\troot\t_\tSpaceAfter=No",
                "4\t.\t.\tPUNCT\tsf\t_\t3\tpunct\t_\t_",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = import_ud([conllu], corpus, "source", "test", 3, 7)
    cases = [
        json.loads(line)
        for line in (corpus / "morphology.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result["samples"] == 3
    assert {case["expected_lemma"] for case in cases} == {"나", "밥", "먹다"}
    assert {label for case in cases for label in case["expected_labels"]} == {
        "particle",
        "past tense",
    }
    assert all(case["provenance"]["oracle"] == result["oracle"] for case in cases)


def test_import_aihub_uses_published_boxes_and_filters_non_eojeols(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _manifest(corpus, ["published-annotation"])
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (100, 40), "white").save(images / "sample.png")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "images": [{"id": "image-1", "file_name": "sample.png"}],
                "annotations": [
                    {"image_id": "image-1", "text": "어디에서", "bbox": [1, 2, 80, 25]},
                    {"image_id": "image-1", "text": "xxx", "bbox": [1, 2, 5, 5]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = import_aihub(labels, images, corpus, "complex", "source", "test", 1, 9)
    annotation = json.loads((corpus / "complex" / "0001.json").read_text(encoding="utf-8"))

    assert result["samples"] == 1
    assert annotation["lines"][0]["box"] == [1.0, 2.0, 81.0, 27.0]
    assert len(annotation["lines"]) == 1


def test_render_ocr_produces_exact_known_text_annotations(tmp_path: Path) -> None:
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.is_file():
        pytest.skip("Windows Korean font is unavailable")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _manifest(corpus, ["known-render"])
    source = tmp_path / "text.jsonl"
    source.write_text(
        json.dumps({"text": "어디에서 만나요?", "source_sample_id": "sentence-1"}) + "\n",
        encoding="utf-8",
    )

    result = render_ocr(
        source,
        corpus,
        "clean",
        "source",
        "source",
        [font],
        1,
        13,
        "test",
    )
    annotation = json.loads((corpus / "clean" / "0001.json").read_text(encoding="utf-8"))

    assert result["samples"] == 1
    assert [item["text"] for item in annotation["lines"][0]["eojeols"]] == [
        "어디에서",
        "만나요",
    ]
    assert annotation["provenance"]["oracle"] == "known-render"


def test_partial_corpus_can_be_locked_and_detects_later_changes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _manifest(
        corpus,
        [
            "known-render",
            "published-annotation",
            "published-annotation-independent-map",
        ],
    )
    for category, oracle in (
        ("clean", "known-render"),
        ("subtitles", "known-render"),
        ("complex", "published-annotation"),
    ):
        directory = corpus / category
        directory.mkdir()
        Image.new("RGB", (100, 40), "white").save(directory / "0001.png")
        (directory / "0001.json").write_text(
            json.dumps(_annotation("0001.png", oracle)), encoding="utf-8"
        )
    morphology = {
        "sentence": "어디에서 만나요?",
        "target_span": [0, 4],
        "expected_lemma": "어디",
        "expected_labels": ["particle"],
        "provenance": {
            "source_id": "source",
            "source_sample_id": "morph-1",
            "source_split": "test",
            "oracle": "published-annotation-independent-map",
            "supporting_source_ids": [],
        },
    }
    (corpus / "morphology.jsonl").write_text(
        json.dumps(morphology, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    result = lock_corpus(corpus, "automated-test", allow_incomplete=True)
    validation = validate_corpus(corpus, allow_incomplete=True)

    assert result["counts"] == {
        "clean": 1,
        "subtitles": 1,
        "complex": 1,
        "morphology": 1,
    }
    assert validation["release_sample_counts"] is False

    (corpus / "clean" / "0001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="failed verification"):
        validate_corpus(corpus, allow_incomplete=True)
