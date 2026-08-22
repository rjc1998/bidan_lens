import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from benchmarks.locked_corpus import CorpusError
from benchmarks.plain_corpus import (
    FONT_FAMILIES,
    PUNCTUATION_CLASSES,
    REQUIRED_SIZES,
    BrowserRenderer,
    DesktopRenderer,
    OracleEntry,
    PlainCandidate,
    RenderSpec,
    UdSentence,
    UdToken,
    _candidate_pool,
    _exclude_split_overlap,
    _line_records,
    _punctuate,
    _render_spec,
    _select_candidates,
    _select_language_candidates,
    acquire_plain,
    load_krdict_oracle,
)


def test_line_records_omit_rows_without_korean_eojeols() -> None:
    values = [
        {
            'index': 0,
            'raw': 'K-2020/v1',
            'core': 'K-2020/v1',
            'box': [0.0, 0.0, 20.0, 10.0],
            'core_box': [0.0, 0.0, 20.0, 10.0],
        },
        {
            'index': 1,
            'raw': '\ud55c\uad6d\uc5b4',
            'core': '\ud55c\uad6d\uc5b4',
            'box': [0.0, 20.0, 20.0, 30.0],
            'core_box': [0.0, 20.0, 20.0, 30.0],
        },
    ]

    lines, target_box = _line_records(values, 1)

    assert len(lines) == 1
    assert lines[0]['text'] == '\ud55c\uad6d\uc5b4'
    assert target_box == [0.0, 20.0, 20.0, 30.0]


def test_candidate_pool_rejects_target_surface_inside_another_eojeol() -> None:
    target = '\ub098'
    sentence = UdSentence(
        'sentence-1',
        '\ub098\ub294 \ub09c \uac04\ub2e4',
        (UdToken('1', target, target, 'NOUN', 'ncn', ''),),
    )
    oracle = {target: (OracleEntry('1', target, ((1, 'definition'),)),)}

    assert _candidate_pool((sentence,), 'source', 'dev', oracle, morphology=False) == []


def test_desktop_renderer_retains_application_for_registered_fonts(tmp_path: Path) -> None:
    from PyQt6.QtWidgets import QApplication

    font = Path('C:/Windows/Fonts/malgun.ttf')
    if not font.is_file():
        pytest.skip('Malgun Gothic is unavailable')
    renderer = DesktopRenderer()
    value = _punctuate(
        ('\uc624\ub298', '\uc5b4\ub514\uc5d0\uc11c', '\ub9cc\ub098\uc694'), 1, 'natural', 1
    )
    spec = RenderSpec(
        'desktop',
        'malgun-gothic',
        font,
        16,
        400,
        100,
        'light',
        'single-line',
        'natural',
    )

    renderer.render(value, spec, tmp_path / 'sample.png')

    assert renderer._app is QApplication.instance()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_acquire_plain_verifies_pinned_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("benchmarks.plain_corpus.platform.system", lambda: "Windows")
    data = b"held-out corpus"
    license_data = b"license"
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "id": "data",
                        "kind": "conllu",
                        "filename": "ud/data.conllu",
                        "url": "https://example.invalid/data",
                        "sha256": _hash(data),
                        "size": len(data),
                        "license_id": "test",
                        "license_artifact": "license",
                    },
                    {
                        "id": "license",
                        "kind": "license",
                        "filename": "licenses/license.txt",
                        "url": "https://example.invalid/license",
                        "sha256": _hash(license_data),
                        "size": len(license_data),
                        "license_id": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    malgun = tmp_path / "malgun.ttf"
    malgun.write_bytes(b"font")

    def fetch(url: str, destination: Path) -> None:
        destination.write_bytes(license_data if url.endswith("license") else data)

    result = acquire_plain(
        tmp_path / "acquired", source_lock=source_lock, malgun=malgun, fetcher=fetch
    )

    assert result == {"artifacts": 2, "malgun_gothic": True, "verified": True}
    assert (tmp_path / "acquired/ud/data.conllu").read_bytes() == data
    acquisition = json.loads((tmp_path / "acquired/acquisition.json").read_text())
    assert acquisition["system_fonts"]["malgun-gothic"]["redistributed"] is False


def test_acquire_plain_rejects_bad_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("benchmarks.plain_corpus.platform.system", lambda: "Windows")
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "id": "data",
                        "kind": "conllu",
                        "filename": "data.conllu",
                        "url": "https://example.invalid/data",
                        "sha256": _hash(b"expected"),
                        "size": 8,
                        "license_id": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    malgun = tmp_path / "malgun.ttf"
    malgun.write_bytes(b"font")

    with pytest.raises(CorpusError, match="failed verification"):
        acquire_plain(
            tmp_path / "acquired",
            source_lock=source_lock,
            malgun=malgun,
            fetcher=lambda _url, path: path.write_bytes(b"wrong"),
        )
    assert not (tmp_path / "acquired/data.conllu.part").exists()


def test_acquire_plain_cleans_partial_file_after_network_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("benchmarks.plain_corpus.platform.system", lambda: "Windows")
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "id": "data",
                        "kind": "conllu",
                        "filename": "data.conllu",
                        "url": "https://example.invalid/data",
                        "sha256": _hash(b"expected"),
                        "size": 8,
                        "license_id": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    malgun = tmp_path / "malgun.ttf"
    malgun.write_bytes(b"font")

    def fail(_url: str, path: Path) -> None:
        path.write_bytes(b"partial")
        raise OSError("network unavailable")

    with pytest.raises(OSError, match="network unavailable"):
        acquire_plain(
            tmp_path / "acquired", source_lock=source_lock, malgun=malgun, fetcher=fail
        )
    assert not (tmp_path / "acquired/data.conllu.part").exists()


def test_independent_krdict_oracle_preserves_definition_order(tmp_path: Path) -> None:
    archive = tmp_path / "krdict.zip"
    document = {
        "LexicalResource": {
            "Lexicon": {
                "LexicalEntry": {
                    "id": "entry-1",
                    "Lemma": {"feat": {"att": "writtenForm", "val": "먹다"}},
                    "Sense": [
                        {
                            "Equivalent": {
                                "feat": [
                                    {"att": "language", "val": "eng"},
                                    {"att": "definition", "val": " to eat "},
                                ]
                            }
                        },
                        {
                            "Equivalent": {
                                "feat": [
                                    {"att": "language", "val": "eng"},
                                    {"att": "definition", "val": "to consume"},
                                ]
                            }
                        },
                    ],
                }
            }
        }
    }
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("dictionary.json", json.dumps(document, ensure_ascii=False))

    entries = load_krdict_oracle(archive)

    assert entries["먹다"][0].entry_id == "entry-1"
    assert entries["먹다"][0].senses == ((1, "to eat"), (2, "to consume"))


def test_independent_krdict_oracle_sorts_multiple_archive_members(tmp_path: Path) -> None:
    archive = tmp_path / "krdict.zip"

    def document(entry_id: str, headword: str) -> str:
        return json.dumps(
            {
                "LexicalResource": {
                    "Lexicon": {
                        "LexicalEntry": {
                            "id": entry_id,
                            "Lemma": {"feat": {"att": "writtenForm", "val": headword}},
                            "Sense": {
                                "Equivalent": {
                                    "feat": [
                                        {"att": "language", "val": "eng"},
                                        {"att": "definition", "val": "definition"},
                                    ]
                                }
                            },
                        }
                    }
                }
            },
            ensure_ascii=False,
        )

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("z.json", document("2", "둘"))
        output.writestr("a.json", document("1", "하나"))

    entries = load_krdict_oracle(archive)

    assert set(entries) == {"하나", "둘"}


@pytest.mark.parametrize(
    ("punctuation", "prefix", "suffix"),
    [
        ("terminal", "", "!"),
        ("comma-colon", "", ":"),
        ("quotes", "\u2018", "\u2019"),
        ("brackets", "[", "]"),
        ("ellipsis", "", "\u2026"),
        ("dash-slash", "/", "/"),
    ],
)
def test_punctuation_keeps_complete_target_separate(
    punctuation: str, prefix: str, suffix: str
) -> None:
    value = _punctuate(("오늘", "어디에서", "만나요"), 1, punctuation, 0)

    assert value.target_surface == "어디에서"
    assert value.words[1] == f"{prefix}어디에서{suffix}"


def test_render_matrix_balances_size_punctuation_renderer_and_font(tmp_path: Path) -> None:
    acquisition = {
        "system_fonts": {"malgun-gothic": {"path": str(tmp_path / "malgun.ttf")}}
    }
    specs = [
        _render_spec(index, tmp_path, acquisition, REQUIRED_SIZES[index % 8])
        for index in range(2_000)
    ]

    assert {size: sum(item.size_px == size for item in specs) for size in REQUIRED_SIZES} == {
        size: 250 for size in REQUIRED_SIZES
    }
    assert {
        punctuation: sum(item.punctuation_class == punctuation for item in specs)
        for punctuation in PUNCTUATION_CLASSES
    } == {punctuation: 250 for punctuation in PUNCTUATION_CLASSES}
    assert {
        name: sum(item.renderer == name for item in specs) for name in ("browser", "desktop")
    } == {"browser": 1_000, "desktop": 1_000}
    assert {font: sum(item.font_id == font for item in specs) for font in FONT_FAMILIES} == {
        font: 400 for font in FONT_FAMILIES
    }
    assert {weight: sum(item.weight == weight for item in specs) for weight in (400, 700)} == {
        400: 1_000,
        700: 1_000,
    }
    assert {
        scale: sum(item.scale_percent == scale for item in specs)
        for scale in (100, 125, 150, 200)
    } == {100: 500, 125: 500, 150: 500, 200: 500}
    assert {theme: sum(item.theme == theme for item in specs) for theme in ("light", "dark")} == {
        "light": 1_000,
        "dark": 1_000,
    }
    assert {
        layout: sum(item.layout == layout for item in specs)
        for layout in ("single-line", "multi-line")
    } == {"single-line": 1_000, "multi-line": 1_000}


def test_selection_is_deterministic_balanced_and_sentence_unique() -> None:
    candidates = []
    for target_class in ("particle", "conjugated", "plain"):
        for index in range(10):
            candidates.append(
                PlainCandidate(
                    "source",
                    f"sentence-{target_class}-{index}:1",
                    "test",
                    ("한글", "문장", "입니다"),
                    1,
                    "문장",
                    "문장",
                    frozenset(),
                    (),
                    target_class,
                )
            )

    first = _select_candidates(candidates, 5, 17)
    second = _select_candidates(reversed(candidates), 5, 17)

    assert first == second
    assert [item.target_class for item in first].count("particle") == 2
    assert [item.target_class for item in first].count("conjugated") == 2
    assert [item.target_class for item in first].count("plain") == 1


def test_development_and_release_sentence_overlap_is_excluded() -> None:
    selected = (
        UdSentence("shared-id", "고유한 문장", ()),
        UdSentence("selected-id", "겹치는 문장", ()),
        UdSentence("kept-id", "남는 문장", ()),
    )
    held_out = (
        UdSentence("shared-id", "다른 문장", ()),
        UdSentence("other-id", "겹치는 문장", ()),
    )

    assert _exclude_split_overlap(selected, held_out) == (selected[2],)


def test_language_selection_is_balanced_and_source_sample_isolated() -> None:
    candidates = []
    for language_class in ('multi-lexical', 'auxiliary'):
        for index in range(4):
            candidates.append(
                PlainCandidate(
                    'source',
                    f'sentence-{language_class}:{index}',
                    'test',
                    ('한글', '문장'),
                    0,
                    '한글',
                    '한글',
                    frozenset(),
                    (),
                    'conjugated',
                    language_class=language_class,
                )
            )
    used = {('source', 'sentence-multi-lexical:0')}

    selected = _select_language_candidates(candidates, used, 4, 17)

    assert [item.language_class for item in selected].count('multi-lexical') == 2
    assert [item.language_class for item in selected].count('auxiliary') == 2
    assert all((item.source_id, item.source_sample_id) not in used for item in selected)


def test_desktop_renderer_returns_exact_target_box(tmp_path: Path) -> None:
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.is_file():
        pytest.skip("Malgun Gothic is unavailable")
    value = _punctuate(("오늘", "어디에서", "만나요"), 1, "quotes", 1)
    spec = RenderSpec(
        "desktop", "malgun-gothic", font, 16, 400, 100, "light", "single-line", "quotes"
    )

    lines, target_box = DesktopRenderer().render(value, spec, tmp_path / "sample.png")

    assert (tmp_path / "sample.png").is_file()
    assert target_box[2] > target_box[0]
    assert [item["text"] for item in lines[0]["eojeols"]] == ["오늘", "어디에서", "만나요"]


@pytest.mark.parametrize("size", REQUIRED_SIZES)
@pytest.mark.parametrize("scale", (100, 125, 150, 200))
def test_desktop_geometry_at_every_required_size_and_scale(
    tmp_path: Path, size: int, scale: int
) -> None:
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.is_file():
        pytest.skip("Malgun Gothic is unavailable")
    value = _punctuate(("오늘", "어디에서", "만나요"), 1, "brackets", 1)
    spec = RenderSpec(
        "desktop", "malgun-gothic", font, size, 700, scale, "dark", "multi-line", "brackets"
    )

    lines, target_box = DesktopRenderer().render(
        value, spec, tmp_path / f"desktop-{size}-{scale}.png"
    )

    pointer = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    assert any(
        item["text"] == "어디에서" and item["box"][0] <= pointer[0] <= item["box"][2]
        for line in lines
        for item in line["eojeols"]
    )


@pytest.fixture(scope="module")
def browser_renderer():
    try:
        renderer = BrowserRenderer()
    except CorpusError as error:
        pytest.skip(str(error))
    yield renderer
    renderer.close()


@pytest.mark.parametrize("size", REQUIRED_SIZES)
@pytest.mark.parametrize("scale", (100, 125, 150, 200))
def test_browser_geometry_at_every_required_size_and_scale(
    browser_renderer: BrowserRenderer, tmp_path: Path, size: int, scale: int
) -> None:
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.is_file():
        pytest.skip("Malgun Gothic is unavailable")
    value = _punctuate(("오늘", "어디에서", "만나요"), 1, "brackets", 1)
    spec = RenderSpec(
        "browser", "malgun-gothic", font, size, 700, scale, "dark", "multi-line", "brackets"
    )

    lines, target_box = browser_renderer.render(
        value, spec, tmp_path / f"browser-{size}-{scale}.png"
    )

    pointer = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    assert any(
        item["text"] == "어디에서" and item["box"][0] <= pointer[0] <= item["box"][2]
        for line in lines
        for item in line["eojeols"]
    )
