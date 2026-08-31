"""Acquire and build the deterministic v1 plain-text evaluation corpus."""

from __future__ import annotations

import hashlib
import html
import io
import json
import os
import platform
import re
import shutil
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Protocol

from benchmarks.locked_corpus import CorpusError
from bidan_lens.analysis.verified_spacing import VERIFIED_SPACING

PLAIN_COUNT = 2_000
STRESS_COUNT = 250
QUICK_COUNT = 200
LANGUAGE_COUNT = 400
LANGUAGE_PER_CLASS = 200
REQUIRED_SIZES = (12, 14, 16, 18, 20, 24, 32, 40)
STRESS_SIZE = 10
PUNCTUATION_CLASSES = (
    "natural",
    "terminal",
    "comma-colon",
    "quotes",
    "brackets",
    "ellipsis",
    "dash-slash",
    "mixed",
)
SCALE_PERCENTS = (100, 125, 150, 200)
FONT_FAMILIES = (
    "noto-sans-kr",
    "noto-serif-kr",
    "nanum-gothic",
    "nanum-myeongjo",
    "malgun-gothic",
)
IMAGE_WIDTH = 1280.0
IMAGE_HEIGHT = 720.0
CARD_TOP = 60.0
CARD_BOTTOM = 660.0
VIEWPORT_TOP = 0.0
VIEWPORT_BOTTOM = IMAGE_HEIGHT
TARGET_SAFE_TOP = 10.0
TARGET_SAFE_BOTTOM = 710.0
RENDER_POLICY_VERSION = 'viewport-v3'
PLAYWRIGHT_PACKAGE = 'playwright'
PLAIN_SOURCE_LOCK = Path(__file__).with_name("plain_sources.lock.json")
PLAIN_ORACLE = "known-render-independent-analysis"


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    kind: str
    filename: str
    url: str
    sha256: str
    size: int
    license_id: str
    split: str | None = None
    license_artifact: str | None = None
    weight: int | None = None


@dataclass(frozen=True, slots=True)
class UdToken:
    token_id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    misc: str
    dependency_relation: str = ""


@dataclass(frozen=True, slots=True)
class UdSentence:
    sent_id: str
    text: str
    tokens: tuple[UdToken, ...]


@dataclass(frozen=True, slots=True)
class OracleEntry:
    entry_id: str
    headword: str
    senses: tuple[tuple[int, str], ...]
    part_of_speech: str | None = None


@dataclass(frozen=True, slots=True)
class OracleComponent:
    surface: str
    lemma: str
    learner_role: str
    entries: tuple[OracleEntry, ...]


@dataclass(frozen=True, slots=True)
class PlainCandidate:
    source_id: str
    source_sample_id: str
    source_split: str
    words: tuple[str, ...]
    target_index: int
    surface: str
    lemma: str
    labels: frozenset[str]
    entries: tuple[OracleEntry, ...]
    target_class: str
    components: tuple[OracleComponent, ...] = ()
    expected_spacing: str | None = None
    language_class: str | None = None


@dataclass(frozen=True, slots=True)
class RenderSpec:
    renderer: str
    font_id: str
    font_path: Path
    size_px: int
    weight: int
    scale_percent: int
    theme: str
    layout: str
    punctuation_class: str


@dataclass(frozen=True, slots=True)
class RenderInput:
    words: tuple[str, ...]
    target_index: int
    target_surface: str


class PlainRenderer(Protocol):
    version: str

    def render(
        self, value: RenderInput, spec: RenderSpec, destination: Path
    ) -> tuple[list[dict[str, object]], list[float]]: ...


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read plain corpus metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise CorpusError(f"plain corpus metadata must be an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").strip()


def _artifacts(path: Path) -> tuple[Artifact, ...]:
    value = _read_object(path)
    if value.get("schema_version") != 1 or not isinstance(value.get("artifacts"), list):
        raise CorpusError("unsupported plain source lock")
    result: list[Artifact] = []
    ids: set[str] = set()
    for raw in value["artifacts"]:
        if not isinstance(raw, dict):
            raise CorpusError("plain source lock contains an invalid artifact")
        required = ("id", "kind", "filename", "url", "sha256", "size", "license_id")
        if not all(raw.get(field) not in (None, "") for field in required):
            raise CorpusError("plain source lock artifact is missing required fields")
        artifact = Artifact(
            str(raw["id"]),
            str(raw["kind"]),
            str(raw["filename"]),
            str(raw["url"]),
            str(raw["sha256"]).lower(),
            int(raw["size"]),
            str(raw["license_id"]),
            str(raw["split"]) if raw.get("split") else None,
            str(raw["license_artifact"]) if raw.get("license_artifact") else None,
            int(raw["weight"]) if raw.get("weight") else None,
        )
        if artifact.artifact_id in ids or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256):
            raise CorpusError("plain source lock has a duplicate id or invalid hash")
        ids.add(artifact.artifact_id)
        result.append(artifact)
    if any(item.license_artifact and item.license_artifact not in ids for item in result):
        raise CorpusError("plain source lock references an unknown license artifact")
    return tuple(result)


def _valid_artifact(path: Path, artifact: Artifact) -> bool:
    return (
        path.is_file() and path.stat().st_size == artifact.size and _sha256(path) == artifact.sha256
    )


def _fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "BiDan-Lens-evaluation/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def acquire_plain(
    destination: Path,
    *,
    source_lock: Path = PLAIN_SOURCE_LOCK,
    malgun: Path = Path("C:/Windows/Fonts/malgun.ttf"),
    local_krdict: Path | None = None,
    fetcher: Callable[[str, Path], None] = _fetch,
) -> dict[str, object]:
    """Explicitly acquire and verify all redistributable plain-evaluation inputs."""
    artifacts = _artifacts(source_lock)
    destination.mkdir(parents=True, exist_ok=True)
    overrides = {"krdict-english-json": local_krdict} if local_krdict is not None else {}
    for artifact in artifacts:
        output = destination / artifact.filename
        if _valid_artifact(output, artifact):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        try:
            override = overrides.get(artifact.artifact_id)
            if override is not None:
                if not override.is_file():
                    raise CorpusError("the local KRDict override does not exist")
                shutil.copy2(override, temporary)
            else:
                fetcher(artifact.url, temporary)
            if not _valid_artifact(temporary, artifact):
                raise CorpusError(
                    f"downloaded artifact failed verification: {artifact.artifact_id}"
                )
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if platform.system() != "Windows" or not malgun.is_file():
        raise CorpusError("Malgun Gothic is required to build the Windows plain-v1 corpus")
    acquisition = {
        "schema_version": 1,
        "source_lock": source_lock.name,
        "source_lock_sha256": _sha256(source_lock),
        "artifacts": {
            artifact.artifact_id: {
                "path": artifact.filename,
                "sha256": artifact.sha256,
                "size": artifact.size,
            }
            for artifact in artifacts
        },
        "system_fonts": {
            "malgun-gothic": {
                "path": str(malgun.resolve()),
                "sha256": _sha256(malgun),
                "size": malgun.stat().st_size,
                "redistributed": False,
            }
        },
    }
    shutil.copy2(source_lock, destination / source_lock.name)
    _write_json(destination / "acquisition.json", acquisition)
    return {"artifacts": len(artifacts), "malgun_gothic": True, "verified": True}


def _verify_acquisition(root: Path) -> tuple[dict[str, Artifact], dict[str, Any]]:
    lock = root / PLAIN_SOURCE_LOCK.name
    acquisition = _read_object(root / "acquisition.json")
    if acquisition.get("source_lock_sha256") != _sha256(lock):
        raise CorpusError("plain acquisition source lock has changed")
    artifacts = {item.artifact_id: item for item in _artifacts(lock)}
    for artifact in artifacts.values():
        if not _valid_artifact(root / artifact.filename, artifact):
            raise CorpusError(f"acquired artifact failed verification: {artifact.artifact_id}")
    system_font = acquisition.get("system_fonts", {}).get("malgun-gothic", {})
    font_path = Path(str(system_font.get("path", "")))
    if not font_path.is_file() or _sha256(font_path) != system_font.get("sha256"):
        raise CorpusError("Malgun Gothic differs from the acquired system-font record")
    return artifacts, acquisition


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _feature(value: Any, name: str) -> str | None:
    if not isinstance(value, dict):
        return None
    for feature in _as_list(value.get("feat")):
        if isinstance(feature, dict) and feature.get("att") == name and feature.get("val"):
            return str(feature["val"])
    return None


def _text(value: Any, *keys: str) -> str | None:
    for key in keys:
        if not isinstance(value, dict) or value.get(key) in (None, ""):
            continue
        candidate = value[key]
        if isinstance(candidate, dict):
            candidate = candidate.get("#text") or candidate.get("writtenForm")
        if candidate not in (None, ""):
            return str(candidate)
    return None


def _entry_id(value: Any, fallback: str) -> str:
    if isinstance(value, dict) and value.get("att") == "id" and value.get("val"):
        return str(value["val"])
    return str(value.get("id") or fallback) if isinstance(value, dict) else fallback


_ORACLE_PARTS_OF_SPEECH = {
    "동사": "verb",
    "형용사": "adjective",
    "명사": "noun",
    "대명사": "pronoun",
    "부사": "adverb",
    "관형사": "determiner",
    "감탄사": "interjection",
    "수사": "numeral",
    "조사": "particle",
}


def _oracle_part_of_speech(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return _ORACLE_PARTS_OF_SPEECH.get(stripped, stripped.casefold())


def load_krdict_oracle(source: Path) -> dict[str, tuple[OracleEntry, ...]]:
    """Parse KRDict independently of the production adapter."""
    if not zipfile.is_zipfile(source):
        raise CorpusError("plain-v1 requires the pinned KRDict ZIP export")
    resolved: dict[str, OracleEntry] = {}
    used_ids: dict[str, tuple[str, str | None, str | None]] = {}
    with zipfile.ZipFile(source) as archive:
        members = sorted(
            (item for item in archive.infolist() if item.filename.endswith(".json")),
            key=lambda item: item.filename,
        )
        if not members:
            raise CorpusError("KRDict oracle archive contains no JSON")
        for document_index, member in enumerate(members):
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as text:
                data = json.load(text)
            lexical = data.get("LexicalResource", data)
            lexicon = lexical.get("Lexicon", lexical) if isinstance(lexical, dict) else lexical
            raw_entries = lexicon.get("LexicalEntry", []) if isinstance(lexicon, dict) else []
            for index, raw_entry in enumerate(_as_list(raw_entries)):
                if not isinstance(raw_entry, dict):
                    continue
                lemma_value = raw_entry.get("Lemma", {})
                headword = _feature(lemma_value, "writtenForm") or _text(
                    lemma_value, "writtenForm", "#text"
                )
                if not headword:
                    continue
                senses: list[tuple[int, str]] = []
                for order, sense in enumerate(_as_list(raw_entry.get("Sense")), start=1):
                    if not isinstance(sense, dict):
                        continue
                    definitions: list[str] = []
                    for equivalent in _as_list(sense.get("Equivalent") or sense.get("equivalent")):
                        language = (
                            _feature(equivalent, "language")
                            or _text(equivalent, "language", "lang")
                            or "eng"
                        ).casefold()
                        if language not in {"eng", "en", "english", "영어"}:
                            continue
                        definition = _feature(equivalent, "definition") or _text(
                            equivalent,
                            "definition",
                            "translation",
                            "writtenForm",
                            "#text",
                        )
                        if not definition:
                            definition = _feature(equivalent, "lemma")
                        if definition:
                            definitions.append(_normalized(definition))
                    direct = _text(sense, "english_definition", "definition_en", "definition")
                    if direct and not definitions:
                        definitions.append(_normalized(direct))
                    senses.extend((order, definition) for definition in definitions)
                if senses:
                    normalized_headword = _normalized(headword)
                    part_of_speech = _oracle_part_of_speech(
                        _feature(raw_entry, "partOfSpeech")
                        or _text(raw_entry, "partOfSpeech", "part_of_speech")
                    )
                    homograph = _feature(raw_entry, "homonym_number") or _text(
                        raw_entry, "homonym_number", "homograph_number"
                    )
                    signature = (normalized_headword, part_of_speech, homograph)
                    entry_id = _entry_id(raw_entry, f"krdict-{document_index}-{index}")
                    if entry_id in used_ids and used_ids[entry_id] != signature:
                        suffix = hashlib.sha256(
                            "\0".join(item or "" for item in signature).encode("utf-8")
                        ).hexdigest()[:16]
                        entry_id = f"{entry_id}:{suffix}"
                    used_ids[entry_id] = signature
                    previous = resolved.get(entry_id)
                    combined_senses = list(previous.senses) if previous else []
                    combined_senses.extend(item for item in senses if item not in combined_senses)
                    resolved[entry_id] = OracleEntry(
                        entry_id,
                        normalized_headword,
                        tuple(combined_senses),
                        part_of_speech,
                    )
    entries: dict[str, list[OracleEntry]] = defaultdict(list)
    for entry in resolved.values():
        entries[entry.headword].append(entry)
    return {
        key: tuple(sorted(value, key=lambda item: item.entry_id)) for key, value in entries.items()
    }


def _expected_dictionary_pos(token: UdToken) -> str | None:
    tags = _morph_parts(token)[1]
    if token.upos == 'ADP' and any(
        tag.startswith('j') and tag != 'jp' for tag in tags
    ):
        return 'particle'
    if any(tag in {"px", "vx"} for tag in tags):
        return "보조 동사"
    if any(tag == "pvg" for tag in tags) or token.upos == "VERB":
        return "verb"
    if any(tag == "paa" for tag in tags) or token.upos == "ADJ":
        return "adjective"
    if any(tag.startswith("n") for tag in tags) or token.upos in {"NOUN", "PROPN"}:
        return "noun"
    if any(tag in {"mag", "maj", "mad"} for tag in tags) or token.upos == "ADV":
        return "adverb"
    if any(tag.startswith("mm") for tag in tags) or token.upos == "DET":
        return "determiner"
    return None


def _oracle_lookup(
    entries: Mapping[str, tuple[OracleEntry, ...]], lemma: str, part_of_speech: str | None
) -> tuple[OracleEntry, ...]:
    available = entries.get(lemma, ())
    if part_of_speech is None:
        return available[:10]
    matching = tuple(entry for entry in available if entry.part_of_speech in {part_of_speech, None})
    remaining = tuple(entry for entry in available if entry not in matching)
    return (*matching, *remaining)[:10]


def _component_role(tag: str) -> str | None:
    if tag in {"pvg", "vv", "xsv"}:
        return "action verb"
    if tag in {"paa", "pad", "va", "xsa", "xsm"}:
        return "descriptive verb"
    if tag in {"px", "vx"}:
        return "helping verb"
    if tag == "nnp" or tag.startswith("nq"):
        return "name or proper noun"
    if tag == "np" or tag.startswith("np"):
        return "pronoun"
    if tag == "nr" or tag.startswith(("nnc", "nno")):
        return "number"
    if tag == "nnb" or tag.startswith("nb"):
        return "dependent noun"
    if tag.startswith("n"):
        return "noun"
    if tag in {"mag", "maj", "mad"}:
        return "adverb"
    if tag.startswith("mm"):
        return "determiner"
    return None


def _upos_component_role(upos: str) -> str:
    return {
        "NOUN": "noun",
        "PROPN": "name or proper noun",
        "PRON": "pronoun",
        "NUM": "number",
        "VERB": "action verb",
        "ADJ": "descriptive verb",
        "AUX": "helping verb",
        "ADV": "adverb",
        "DET": "determiner",
    }.get(upos, "word")


def _component_positions(tag: str, role: str | None = None) -> tuple[str, ...]:
    if role == 'adverb':
        return ('adverb',)
    if role == "helping verb":
        return ("보조 동사", "보조 형용사")
    if role == "descriptive verb":
        return ("adjective",)
    if role == "action verb":
        return ("verb",)
    if tag in {"px", "vx"}:
        return ("보조 동사", "보조 형용사")
    if tag in {"pvg", "vv", "xsv"}:
        return ("verb",)
    if tag in {"paa", "pad", "va", "xsa", "xsm"}:
        return ("adjective",)
    if tag.startswith("n"):
        return ("noun",)
    if tag in {"mag", "maj", "mad"}:
        return ("adverb",)
    if tag.startswith("mm"):
        return ("determiner",)
    return ()


def _ordered_oracle_entries(
    oracle: Mapping[str, tuple[OracleEntry, ...]],
    lemma: str,
    positions: tuple[str, ...],
) -> tuple[OracleEntry, ...]:
    available = oracle.get(lemma, ())
    ordered: list[OracleEntry] = []
    seen: set[str] = set()
    for position in positions:
        for entry in available:
            if entry.part_of_speech in {position, None} and entry.entry_id not in seen:
                seen.add(entry.entry_id)
                ordered.append(entry)
    ordered.extend(entry for entry in available if entry.entry_id not in seen)
    return tuple(ordered[:10])


def _expected_components(
    token: UdToken, oracle: Mapping[str, tuple[OracleEntry, ...]]
) -> tuple[OracleComponent, ...]:
    forms, tags = _morph_parts(token)
    components: list[OracleComponent] = []
    index = 0
    while index < min(len(forms), len(tags)):
        form = _normalized(forms[index])
        tag = tags[index]
        role = _component_role(tag)
        if role is None:
            index += 1
            continue
        if token.upos == "ADJ" and role == "action verb":
            role = "descriptive verb"
        elif token.upos == "AUX" and role in {"action verb", "descriptive verb"}:
            role = "helping verb"
        if (
            token.upos == 'ADV'
            and role == 'noun'
            and len(tags) == 1
            and token.dependency_relation == 'advmod'
        ):
            role = 'adverb'
        surface = form
        lemma = form
        component_tag = tag
        if tag.startswith("n") and index + 1 < min(len(forms), len(tags)):
            following_tag = tags[index + 1]
            if following_tag in {"xsv", "xsa", "xsm"}:
                surface += _normalized(forms[index + 1])
                lemma = surface + "다"
                component_tag = following_tag
                role = _component_role(component_tag) or role
                index += 1
            elif following_tag == "xsn":
                suffix_closes_noun = index + 2 >= min(len(forms), len(tags)) or (
                    tags[index + 2].startswith("j") and tags[index + 2] != "jp"
                )
                if forms[index + 1] != "들" and suffix_closes_noun:
                    surface += _normalized(forms[index + 1])
                    lemma = surface
                    index += 1
        elif tag in {
            "pvg",
            "paa",
            "pad",
            "px",
            "vv",
            "va",
            "vx",
            "xsv",
            "xsa",
        }:
            lemma = form if form.endswith("다") else form + "다"
        entries = _ordered_oracle_entries(oracle, lemma, _component_positions(component_tag, role))
        components.append(OracleComponent(surface, lemma, role, entries))
        index += 1
    if not components:
        for form, tag in zip(forms, tags, strict=False):
            if tag.startswith('j') and tag != 'jp':
                particle = _normalized(form)
                components.append(
                    OracleComponent(
                        particle,
                        particle,
                        'particle',
                        _oracle_lookup(oracle, particle, 'particle'),
                    )
                )
                break
    if not components:
        for form, tag in zip(forms, tags, strict=False):
            if tag == 'jp':
                lemma = _normalized(form)
                if not lemma.endswith('\ub2e4'):
                    lemma += '\ub2e4'
                components.append(
                    OracleComponent(
                        _normalized(token.form),
                        lemma,
                        'linking word',
                        _ordered_oracle_entries(oracle, lemma, ()),
                    )
                )
                break
    return tuple(components)


def _parse_ud(path: Path) -> tuple[UdSentence, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CorpusError(f"cannot read CoNLL-U source: {path.name}") from error
    sentences: list[UdSentence] = []
    for block in raw.split("\n\n"):
        sent_id = text = ""
        tokens: list[UdToken] = []
        for line in block.splitlines():
            if line.startswith("# sent_id = "):
                sent_id = line.removeprefix("# sent_id = ").strip()
            elif line.startswith("# text = "):
                text = line.removeprefix("# text = ").strip()
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) == 10 and "-" not in fields[0] and "." not in fields[0]:
                    tokens.append(
                        UdToken(
                            fields[0],
                            fields[1],
                            fields[2],
                            fields[3],
                            fields[4],
                            fields[9],
                            fields[7],
                        )
                    )
        if sent_id and text and tokens:
            sentences.append(UdSentence(sent_id, _normalized(text), tuple(tokens)))
    return tuple(sentences)


def _exclude_split_overlap(
    selected: Iterable[UdSentence], held_out: Iterable[UdSentence]
) -> tuple[UdSentence, ...]:
    held_out_values = tuple(held_out)
    held_out_ids = {sentence.sent_id for sentence in held_out_values}
    held_out_text = {sentence.text for sentence in held_out_values}
    return tuple(
        sentence
        for sentence in selected
        if sentence.sent_id not in held_out_ids and sentence.text not in held_out_text
    )


def _misc_value(misc: str, key: str) -> str | None:
    for item in misc.split("|"):
        name, separator, value = item.partition("=")
        if separator and name == key:
            return value
    return None


def _morph_parts(token: UdToken) -> tuple[list[str], list[str]]:
    lemma = _misc_value(token.misc, "OrigLemma") or token.lemma
    forms = lemma.split("+") if lemma and lemma != "_" else []
    tags = token.xpos.casefold().split("+") if token.xpos and token.xpos != "_" else []
    return forms, tags


def _expected_lemma(token: UdToken) -> str | None:
    forms, tags = _morph_parts(token)
    if not forms:
        return None
    for index, tag in enumerate(tags):
        if (
            tag.startswith("n")
            and index + 1 < min(len(forms), len(tags))
            and tags[index + 1] == "xsm"
        ):
            return forms[index] + forms[index + 1] + "다"
    verb_roots = {"pvg", "paa", "pad", "px", "vv", "va", "vx", "xsv", "xsa"}
    for index, tag in enumerate(tags):
        if (
            tag.startswith("n")
            and index + 1 < min(len(forms), len(tags))
            and tags[index + 1]
            in {
                "xsv",
                "xsa",
            }
        ):
            return forms[index] + forms[index + 1] + "다"
        if tag in verb_roots:
            return forms[index] if forms[index].endswith("다") else forms[index] + "다"
    for index, tag in enumerate(tags):
        if tag.startswith("n") and index < len(forms):
            return forms[index]
    if token.upos in {"VERB", "ADJ", "AUX"}:
        return forms[0] if forms[0].endswith("다") else forms[0] + "다"
    return forms[0]


def _expected_labels(token: UdToken) -> frozenset[str]:
    forms, tags = _morph_parts(token)
    labels: set[str] = set()
    for form, tag in zip(forms, tags, strict=False):
        if tag == "jp":
            continue
        if tag.startswith("j"):
            labels.add("particle")
        if tag.startswith("e"):
            labels.add("verb ending")
        if tag == "ep":
            if any(marker in form for marker in ("았", "었", "였")):
                labels.add("past tense")
            if "겠" in form:
                labels.add("future or intention")
            if "시" in form:
                labels.add("honorific")
        if tag in {"ec", "ecc", "ecs", "ecx"}:
            if form == "고":
                labels.add("connecting ending")
            if form.endswith("지만"):
                labels.add("contrast ending")
            if form.endswith("면"):
                labels.add("conditional ending")
    if token.form.endswith("세요"):
        labels.add("polite request")
    elif token.form.endswith(("어요", "아요")):
        labels.add("polite style")
    if token.form.endswith(("습니다", "ㅂ니다")):
        labels.add("formal polite style")
    return frozenset(labels)


def _trim_word(value: str) -> tuple[str, int, int]:
    start = 0
    end = len(value)
    while start < end and unicodedata.category(value[start])[0] in {"P", "Z"}:
        start += 1
    while end > start and unicodedata.category(value[end - 1])[0] in {"P", "Z"}:
        end -= 1
    return value[start:end], start, end


def _contains_hangul(value: str) -> bool:
    return any(0xAC00 <= ord(character) <= 0xD7A3 for character in value)


def _candidate_pool(
    sentences: Iterable[UdSentence],
    source_id: str,
    split: str,
    oracle: Mapping[str, tuple[OracleEntry, ...]],
    *,
    morphology: bool,
) -> list[PlainCandidate]:
    result: list[PlainCandidate] = []
    for sentence in sentences:
        words = tuple(sentence.text.split())
        if not 3 <= len(words) <= 35:
            continue
        available: dict[str, list[int]] = defaultdict(list)
        for word_index, word in enumerate(words):
            core, _, _ = _trim_word(word)
            available[_normalized(core)].append(word_index)
        sentence_candidates: list[PlainCandidate] = []
        for token in sentence.tokens:
            surface = _normalized(token.form)
            if sum(_normalized(word).count(surface) for word in words) != 1:
                continue
            indices = available.get(surface, [])
            if len(indices) != 1 or not _contains_hangul(surface):
                continue
            lemma = _expected_lemma(token) if morphology else surface
            if not lemma:
                continue
            lemma = _normalized(lemma)
            dictionary_entries = _oracle_lookup(oracle, lemma, _expected_dictionary_pos(token))
            if not dictionary_entries:
                continue
            labels = _expected_labels(token) if morphology else frozenset()
            components = _expected_components(token, oracle)
            if (
                "particle" in labels
                and len(components) == 1
                and components[0].learner_role
                in {
                    "noun",
                    "name or proper noun",
                    "pronoun",
                    "number",
                    "dependent noun",
                }
                and components[0].surface == surface
            ):
                labels -= {"particle"}
            tags = _morph_parts(token)[1]
            if "particle" in labels:
                target_class = "particle"
            elif (
                morphology
                and token.upos in {"VERB", "ADJ", "AUX"}
                and any(tag.startswith("e") for tag in tags)
            ):
                target_class = "conjugated"
            elif not labels and surface == lemma:
                target_class = "plain"
            else:
                continue
            if not components:
                components = (
                    OracleComponent(
                        surface,
                        lemma,
                        (target_class if morphology else _upos_component_role(token.upos)),
                        dictionary_entries,
                    ),
                )
            lemma = components[0].lemma
            if components[0].entries:
                dictionary_entries = components[0].entries
            if len(components) >= 2:
                language_class = "multi-lexical"
            elif any(component.learner_role == "helping verb" for component in components):
                language_class = "auxiliary"
            else:
                language_class = None
            sentence_candidates.append(
                PlainCandidate(
                    source_id,
                    f"{sentence.sent_id}:{token.token_id}",
                    split,
                    words,
                    indices[0],
                    surface,
                    lemma,
                    labels,
                    dictionary_entries,
                    target_class,
                    components,
                    VERIFIED_SPACING.get(surface),
                    language_class,
                )
            )
        sentence_candidates.sort(key=lambda item: (item.target_class, item.source_sample_id))
        result.extend(sentence_candidates)
    return result


def _stable(values: Iterable[PlainCandidate], seed: int) -> list[PlainCandidate]:
    return sorted(
        values,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.source_id}:{item.source_sample_id}".encode()
        ).digest(),
    )


def _select_candidates(
    candidates: Iterable[PlainCandidate], count: int, seed: int
) -> tuple[PlainCandidate, ...]:
    desired = {
        "particle": count * 2 // 5,
        "conjugated": count * 2 // 5,
        "plain": count - 2 * (count * 2 // 5),
    }
    pools: dict[str, list[PlainCandidate]] = defaultdict(list)
    for candidate in candidates:
        pools[candidate.target_class].append(candidate)
    selected: list[PlainCandidate] = []
    used_sentences: set[tuple[str, str]] = set()
    for offset, target_class in enumerate(("particle", "conjugated", "plain")):
        selected_for_class = 0
        for candidate in _stable(pools[target_class], seed + offset):
            sentence_id = (candidate.source_id, candidate.source_sample_id.rsplit(":", 1)[0])
            if sentence_id in used_sentences:
                continue
            selected.append(candidate)
            used_sentences.add(sentence_id)
            selected_for_class += 1
            if selected_for_class == desired[target_class]:
                break
        if selected_for_class != desired[target_class]:
            raise CorpusError(
                f"not enough {target_class} plain-text targets for the requested corpus"
            )
    return tuple(_stable(selected, seed + 97))


def _select_language_candidates(
    candidates: Iterable[PlainCandidate],
    used_identities: set[tuple[str, str]],
    count: int,
    seed: int,
) -> tuple[PlainCandidate, ...]:
    if count == 0:
        return ()
    if count % 2:
        raise CorpusError("focused language count must be even")
    per_class = count // 2
    selected: list[PlainCandidate] = []
    selected_identities = set(used_identities)
    for offset, language_class in enumerate(("multi-lexical", "auxiliary")):
        available = (
            candidate for candidate in candidates if candidate.language_class == language_class
        )
        class_count = 0
        for candidate in _stable(available, seed + offset):
            identity = (candidate.source_id, candidate.source_sample_id)
            if identity in selected_identities:
                continue
            selected.append(candidate)
            selected_identities.add(identity)
            class_count += 1
            if class_count == per_class:
                break
        if class_count != per_class:
            raise CorpusError(f"not enough {language_class} focused language targets")
    return tuple(_stable(selected, seed + 97))


def _punctuate(
    words: tuple[str, ...], target_index: int, punctuation_class: str, variant: int
) -> RenderInput:
    result = list(words)
    core, _, _ = _trim_word(result[target_index])
    wrappers = {
        "terminal": ("", "?" if variant % 2 else "!"),
        "comma-colon": ("", "," if variant % 2 else ":"),
        "quotes": (("\u201c", "\u201d") if variant % 2 else ("\u2018", "\u2019")),
        "brackets": (("(", ")") if variant % 2 else ("[", "]")),
        "ellipsis": ("", "\u2026"),
        "dash-slash": (("\u2014", "\u2014") if variant % 2 else ("/", "/")),
    }
    if punctuation_class in wrappers:
        prefix, suffix = wrappers[punctuation_class]
        result[target_index] = f"{prefix}{core}{suffix}"
    elif punctuation_class == "mixed":
        result.insert(min(len(result), target_index + 1), f"K-{2020 + variant % 10}/v{variant % 5}")
    elif punctuation_class != "natural":
        raise CorpusError("unknown punctuation class")
    return RenderInput(tuple(result), target_index, core)


def _render_spec(
    index: int, acquired: Path, acquisition: Mapping[str, Any], size: int
) -> RenderSpec:
    stress = size == STRESS_SIZE
    block = index if stress else index // len(REQUIRED_SIZES)
    size_index = 0 if stress else index % len(REQUIRED_SIZES)
    renderer = "browser" if block % 2 == 0 else "desktop"
    font_id = FONT_FAMILIES[(index // 16) % len(FONT_FAMILIES)]
    weight = 400 if (block + size_index) % 2 == 0 else 700
    scale = SCALE_PERCENTS[(block + size_index) % len(SCALE_PERCENTS)]
    theme = "light" if (block // 2 + size_index) % 2 == 0 else "dark"
    layout = "single-line" if (block // 4 + size_index) % 2 == 0 else "multi-line"
    punctuation_class = PUNCTUATION_CLASSES[
        (index % len(REQUIRED_SIZES) + index // len(REQUIRED_SIZES)) % len(PUNCTUATION_CLASSES)
    ]
    if font_id == "noto-sans-kr":
        font_path = acquired / "fonts/NotoSansKR-wght.ttf"
    elif font_id == "noto-serif-kr":
        font_path = acquired / "fonts/NotoSerifKR-wght.ttf"
    elif font_id == "nanum-gothic":
        font_path = acquired / f"fonts/NanumGothic-{'Bold' if weight == 700 else 'Regular'}.ttf"
    elif font_id == "nanum-myeongjo":
        font_path = acquired / f"fonts/NanumMyeongjo-{'Bold' if weight == 700 else 'Regular'}.ttf"
    else:
        font_path = Path(str(acquisition["system_fonts"]["malgun-gothic"]["path"]))
    return RenderSpec(
        renderer,
        font_id,
        font_path,
        size,
        weight,
        scale,
        theme,
        layout,
        punctuation_class,
    )


def _line_records(
    values: list[dict[str, object]], target_index: int
) -> tuple[list[dict[str, object]], list[float]]:
    rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    target_box: list[float] | None = None
    for value in values:
        box = [float(item) for item in value["box"]]  # type: ignore[arg-type]
        rows[round(box[1] / 3)].append(value)
        if int(value["index"]) == target_index:
            target_box = [float(item) for item in value["core_box"]]  # type: ignore[arg-type]
    if target_box is None:
        raise CorpusError("renderer did not return the selected target")
    lines: list[dict[str, object]] = []
    for row in sorted(rows):
        words = sorted(rows[row], key=lambda item: float(item["box"][0]))  # type: ignore[index]
        boxes = [[float(part) for part in item["box"]] for item in words]  # type: ignore[arg-type]
        eojeols = [
            {"text": item["core"], "box": [float(part) for part in item["core_box"]]}
            for item in words
            if item["core"] and _contains_hangul(str(item["core"]))
        ]
        if not eojeols:
            continue
        lines.append(
            {
                "text": " ".join(str(item["raw"]) for item in words),
                "box": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "eojeols": eojeols,
                "regions": [
                    {
                        "raw_box": [float(part) for part in item["box"]],
                        "core_box": [float(part) for part in item["core_box"]],
                        "contains_hangul": _contains_hangul(str(item["core"])),
                    }
                    for item in words
                ],
            }
        )
    return lines, target_box


def _position_visible_values(
    values: list[dict[str, object]], target_index: int
) -> tuple[list[dict[str, object]], float]:
    """Move an off-screen target into the image and omit clipped words."""
    target = next((value for value in values if int(value["index"]) == target_index), None)
    if target is None:
        raise CorpusError("renderer did not return the selected target")
    target_raw_box = [float(item) for item in target["box"]]
    target_height = target_raw_box[3] - target_raw_box[1]
    if target_height <= 0.0 or target_height > TARGET_SAFE_BOTTOM - TARGET_SAFE_TOP:
        raise CorpusError("selected target cannot fit inside the render viewport")
    offset = 0.0
    if target_raw_box[1] < TARGET_SAFE_TOP:
        offset = TARGET_SAFE_TOP - target_raw_box[1]
    elif target_raw_box[3] > TARGET_SAFE_BOTTOM:
        offset = TARGET_SAFE_BOTTOM - target_raw_box[3]

    positioned: list[dict[str, object]] = []
    for value in values:
        raw_box = [float(item) for item in value["box"]]
        core_box = [float(item) for item in value["core_box"]]
        raw_box[1] += offset
        raw_box[3] += offset
        core_box[1] += offset
        core_box[3] += offset
        if raw_box[1] < VIEWPORT_TOP or raw_box[3] > VIEWPORT_BOTTOM:
            continue
        positioned.append({**value, "box": raw_box, "core_box": core_box})

    positioned_target = next(
        (value for value in positioned if int(value["index"]) == target_index), None
    )
    if positioned_target is None:
        raise CorpusError("renderer could not retain the selected target in the viewport")
    target_box = [float(item) for item in positioned_target["core_box"]]
    if not (
        0.0 <= target_box[0] < target_box[2] <= IMAGE_WIDTH
        and 0.0 <= target_box[1] < target_box[3] <= IMAGE_HEIGHT
    ):
        raise CorpusError("selected target lies outside the render viewport")
    return positioned, offset


class DesktopRenderer:
    def __init__(self) -> None:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

        self.version = (
            f'PyQt6-{PYQT_VERSION_STR}-Qt-{QT_VERSION_STR}-QPainter-'
            f'{RENDER_POLICY_VERSION}'
        )
        self._families: dict[Path, str] = {}
        self._app: Any | None = None

    def render(
        self, value: RenderInput, spec: RenderSpec, destination: Path
    ) -> tuple[list[dict[str, object]], list[float]]:
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QPainter
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
        self._app = app
        family = self._families.get(spec.font_path)
        if family is None:
            font_id = QFontDatabase.addApplicationFont(str(spec.font_path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            family = families[0] if families else "Malgun Gothic"
            self._families[spec.font_path] = family
        font = QFont(family)
        font.setPixelSize(max(1, round(spec.size_px * spec.scale_percent / 100)))
        font.setWeight(QFont.Weight.Bold if spec.weight == 700 else QFont.Weight.Normal)
        metrics = QFontMetricsF(font)
        background, foreground, border = (
            ("#ffffff", "#202124", "#c7cbd1")
            if spec.theme == "light"
            else ("#202124", "#f1f3f4", "#5f6368")
        )
        left = 120.0
        top = 150.0
        max_width = 1040.0 if spec.layout == "single-line" else 650.0
        space = metrics.horizontalAdvance(" ")
        line_height = metrics.height() + max(6.0, metrics.height() * 0.4)
        x = left
        y = top
        values: list[dict[str, object]] = []
        for index, raw_word in enumerate(value.words):
            width = metrics.horizontalAdvance(raw_word)
            if x > left and x + width > left + max_width:
                x = left
                y += line_height
            core, start, end = _trim_word(raw_word)
            core_left = x + metrics.horizontalAdvance(raw_word[:start])
            core_right = core_left + metrics.horizontalAdvance(raw_word[start:end])
            values.append(
                {
                    "index": index,
                    "raw": raw_word,
                    "core": core,
                    "box": [x, y, x + width, y + metrics.height()],
                    "core_box": [core_left, y, core_right, y + metrics.height()],
                }
            )
            x += width + space
        values, _ = _position_visible_values(values, value.target_index)
        image = QImage(round(IMAGE_WIDTH), round(IMAGE_HEIGHT), QImage.Format.Format_RGB32)
        image.fill(QColor(background))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(QColor(border))
        painter.drawRoundedRect(QRectF(70, CARD_TOP, 1140, 600), 16, 16)
        painter.setPen(QColor(foreground))
        chrome_font = QFont("Segoe UI")
        chrome_font.setPixelSize(14)
        painter.setFont(chrome_font)
        painter.drawText(QPointF(90, 100), "Settings")
        painter.setFont(font)
        painter.save()
        painter.setClipRect(QRectF(0, VIEWPORT_TOP, IMAGE_WIDTH, VIEWPORT_BOTTOM))
        for rendered_value in values:
            box = rendered_value["box"]
            painter.drawText(
                QPointF(float(box[0]), float(box[1]) + metrics.ascent()),
                str(rendered_value["raw"]),
            )
        painter.restore()
        painter.end()
        if not image.save(str(destination), "PNG"):
            raise CorpusError("Qt could not save a rendered plain-text sample")
        return _line_records(values, value.target_index)


class BrowserRenderer:
    def __init__(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise CorpusError(
                "browser rendering requires `pip install -e .[evaluation]` and "
                "`playwright install chromium`"
            ) from error
        self._temporary = tempfile.TemporaryDirectory(prefix="bidan-plain-browser-")
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=True)
            self._context = self._browser.new_context(viewport={"width": 1280, "height": 720})
            self._page = self._context.new_page()
        except Exception as error:
            self.close()
            raise CorpusError("pinned Playwright Chromium is not installed") from error
        self.version = (
            f'Playwright-{package_version(PLAYWRIGHT_PACKAGE)}-Chromium-'
            f'{self._browser.version}-{RENDER_POLICY_VERSION}'
        )

    def close(self) -> None:
        context = getattr(self, "_context", None)
        if context is not None:
            context.close()
            self._context = None
        browser = getattr(self, "_browser", None)
        if browser is not None:
            browser.close()
            self._browser = None
        playwright = getattr(self, "_playwright", None)
        if playwright is not None:
            playwright.stop()
            self._playwright = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def render(
        self, value: RenderInput, spec: RenderSpec, destination: Path
    ) -> tuple[list[dict[str, object]], list[float]]:
        page = self._page
        font_uri = spec.font_path.resolve().as_uri()
        spans: list[str] = []
        for index, raw_word in enumerate(value.words):
            core, start, end = _trim_word(raw_word)
            spans.append(
                f'<span class="word" data-index="{index}">'
                + html.escape(raw_word[:start])
                + f'<span class="core">{html.escape(core)}</span>'
                + html.escape(raw_word[end:])
                + "</span>"
            )
        background, foreground, border = (
            ("#ffffff", "#202124", "#c7cbd1")
            if spec.theme == "light"
            else ("#202124", "#f1f3f4", "#5f6368")
        )
        width = 1040 if spec.layout == "single-line" else 650
        effective_size = spec.size_px * spec.scale_percent / 100
        document = f"""<!doctype html><meta charset="utf-8"><style>
@font-face {{ font-family: Fixture; src: url('{font_uri}'); font-weight: 100 900; }}
html,body {{ margin:0; width:1280px; height:720px; background:{background}; }}
#card {{ box-sizing:border-box; position:absolute; left:70px; top:60px; width:1140px;
height:600px; padding:90px 50px; border:1px solid {border}; border-radius:16px; }}
#chrome {{ position:absolute; left:90px; top:82px; font:14px 'Segoe UI'; color:{foreground}; }}
#text {{ width:{width}px; color:{foreground}; font-family:Fixture; font-size:{effective_size}px;
font-weight:{spec.weight}; line-height:1.45; }}
.word {{ display:inline-block; white-space:nowrap; }}
</style><div id="chrome">Settings</div>
<div id="card"><div id="text">{" ".join(spans)}</div></div>"""
        html_path = Path(self._temporary.name) / "fixture.html"
        html_path.write_text(document, encoding="utf-8")
        page.goto(html_path.as_uri())
        page.evaluate("document.fonts.ready")
        values = page.locator(".word").evaluate_all(
            """items => items.map(item => {
                const box = item.getBoundingClientRect();
                const core = item.querySelector('.core');
                const coreBox = core.getBoundingClientRect();
                return {index:Number(item.dataset.index), raw:item.textContent,
                  core:core.textContent, box:[box.left,box.top,box.right,box.bottom],
                  core_box:[coreBox.left,coreBox.top,coreBox.right,coreBox.bottom]};
            })"""
        )
        _, offset = _position_visible_values(values, value.target_index)
        if offset:
            page.locator("#text").evaluate(
                "(item, delta) => item.style.transform = `translateY(${delta}px)`",
                offset,
            )
            values = page.locator(".word").evaluate_all(
                """items => items.map(item => {
                    const box = item.getBoundingClientRect();
                    const core = item.querySelector(`.core`);
                    const coreBox = core.getBoundingClientRect();
                    return {index:Number(item.dataset.index), raw:item.textContent,
                      core:core.textContent, box:[box.left,box.top,box.right,box.bottom],
                      core_box:[coreBox.left,coreBox.top,coreBox.right,coreBox.bottom]};
                })"""
            )
        values, _ = _position_visible_values(values, value.target_index)
        visible_indices = [int(item["index"]) for item in values]
        page.locator(".word").evaluate_all(
            """(items, visible) => items.forEach(item => {
                item.style.visibility = visible.includes(Number(item.dataset.index))
                  ? `visible` : `hidden`;
            })""",
            visible_indices,
        )
        page.screenshot(path=str(destination))
        return _line_records(values, value.target_index)


def _oracle_json(entries: tuple[OracleEntry, ...]) -> list[dict[str, object]]:
    return [
        {
            "entry_id": entry.entry_id,
            "headword": entry.headword,
            "senses": [
                {"order": order, "definition": definition} for order, definition in entry.senses
            ],
        }
        for entry in entries
    ]


def _component_json(components: tuple[OracleComponent, ...]) -> list[dict[str, object]]:
    return [
        {
            "surface": component.surface,
            "lemma": component.lemma,
            "learner_role": component.learner_role,
            "expected_dictionary_entries": _oracle_json(component.entries),
        }
        for component in components
    ]


def _sources_manifest(
    acquired: Path, artifacts: Mapping[str, Artifact], acquisition: Mapping[str, Any]
) -> dict[str, object]:
    malgun_evidence = acquired / "licenses/Malgun-Gothic-system-source.txt"
    malgun = acquisition["system_fonts"]["malgun-gothic"]
    malgun_evidence.parent.mkdir(parents=True, exist_ok=True)
    malgun_evidence.write_text(
        "Windows system font used locally for rendering only; not redistributed.\n"
        f"Path: {malgun['path']}\nSHA-256: {malgun['sha256']}\n",
        encoding="utf-8",
    )
    source_specs = (
        (
            "ud-korean-gsd-2.18",
            "ud-korean-gsd-license",
            "published GSD text",
            "ud-korean-gsd-dev",
        ),
        (
            "ud-korean-kaist-2.18",
            "ud-korean-kaist-license",
            "published KAIST lemma and XPOS annotations",
            "ud-korean-kaist-dev",
        ),
        ("noto-sans-kr", "noto-sans-kr-license", "rendering font", "noto-sans-kr"),
        ("noto-serif-kr", "noto-serif-kr-license", "rendering font", "noto-serif-kr"),
        ("nanum-gothic", "nanum-gothic-license", "rendering font", "nanum-gothic-regular"),
        (
            "nanum-myeongjo",
            "nanum-myeongjo-license",
            "rendering font",
            "nanum-myeongjo-regular",
        ),
        (
            "krdict-english-json",
            "krdict-license",
            "published English dictionary oracle",
            "krdict-english-json",
        ),
    )
    entries: list[dict[str, object]] = []
    for source_id, license_id, basis, artifact_id in source_specs:
        entries.append(
            {
                "id": source_id,
                "license_id": artifacts[artifact_id].license_id,
                "license_evidence": artifacts[license_id].filename,
                "annotation_basis": basis,
                "allowed_oracles": [PLAIN_ORACLE],
            }
        )
    entries.append(
        {
            "id": "malgun-gothic",
            "license_id": "Microsoft-Windows-system-font",
            "license_evidence": "licenses/Malgun-Gothic-system-source.txt",
            "annotation_basis": "locally installed rendering font; not redistributed",
            "allowed_oracles": [PLAIN_ORACLE],
        }
    )
    return {"schema_version": 1, "sources": entries}


def _copy_evidence(acquired: Path, corpus: Path, artifacts: Mapping[str, Artifact]) -> None:
    for artifact in artifacts.values():
        if artifact.kind != "license":
            continue
        destination = corpus / artifact.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(acquired / artifact.filename, destination)
    shutil.copy2(
        acquired / "licenses/Malgun-Gothic-system-source.txt",
        corpus / "licenses/Malgun-Gothic-system-source.txt",
    )
    shutil.copy2(acquired / "acquisition.json", corpus / "acquisition.json")
    shutil.copy2(acquired / PLAIN_SOURCE_LOCK.name, corpus / PLAIN_SOURCE_LOCK.name)


def build_plain(
    acquired: Path,
    corpus: Path,
    profile: str,
    *,
    seed: int = 20260822,
    count: int = PLAIN_COUNT,
    stress_count: int = STRESS_COUNT,
    language_count: int | None = None,
    renderers: Mapping[str, PlainRenderer] | None = None,
) -> dict[str, object]:
    if profile not in {"dev", "release"}:
        raise CorpusError("plain profile must be dev or release")
    if corpus.exists() and any(corpus.iterdir()):
        raise CorpusError("plain corpus output must be empty")
    if language_count is None:
        language_count = LANGUAGE_COUNT if count == PLAIN_COUNT else 0
    corpus.mkdir(parents=True, exist_ok=True)
    artifacts, acquisition = _verify_acquisition(acquired)
    split = "dev" if profile == "dev" else "test"
    krdict = load_krdict_oracle(acquired / artifacts["krdict-english-json"].filename)
    other_split = "test" if split == "dev" else "dev"
    kaist = _exclude_split_overlap(
        _parse_ud(acquired / artifacts[f"ud-korean-kaist-{split}"].filename),
        _parse_ud(acquired / artifacts[f"ud-korean-kaist-{other_split}"].filename),
    )
    gsd = _exclude_split_overlap(
        _parse_ud(acquired / artifacts[f"ud-korean-gsd-{split}"].filename),
        _parse_ud(acquired / artifacts[f"ud-korean-gsd-{other_split}"].filename),
    )
    candidates = _candidate_pool(kaist, "ud-korean-kaist-2.18", split, krdict, morphology=True)
    candidates.extend(_candidate_pool(gsd, "ud-korean-gsd-2.18", split, krdict, morphology=False))
    selected = _select_candidates(candidates, count, seed)
    used_sentences = {
        (item.source_id, item.source_sample_id.rsplit(":", 1)[0]) for item in selected
    }
    stress_candidates = (
        item
        for item in candidates
        if (item.source_id, item.source_sample_id.rsplit(":", 1)[0]) not in used_sentences
    )
    stress_selected = _select_candidates(stress_candidates, stress_count, seed + 1_009)
    used_sentences.update(
        (item.source_id, item.source_sample_id.rsplit(":", 1)[0]) for item in stress_selected
    )
    used_identities = {
        (item.source_id, item.source_sample_id) for item in (*selected, *stress_selected)
    }
    language_selected = _select_language_candidates(
        candidates, used_identities, language_count, seed + 2_017
    )
    _write_json(corpus / "sources.json", _sources_manifest(acquired, artifacts, acquisition))
    _copy_evidence(acquired, corpus, artifacts)

    language_root = corpus / "language"
    language_root.mkdir()
    for index, candidate in enumerate(language_selected, 1):
        sentence = " ".join(candidate.words)
        start = sentence.find(candidate.surface)
        if start < 0 or sentence.find(candidate.surface, start + 1) >= 0:
            raise CorpusError("focused language target is not unique in its sentence")
        _write_json(
            language_root / f"{index:04d}.json",
            {
                "schema_version": 4,
                "sample_id": f"{profile}-language-{index:04d}",
                "sentence": sentence,
                "sentence_span": [start, start + len(candidate.surface)],
                "target": {
                    "text": candidate.surface,
                    "box": [0.0, 0.0, 1.0, 1.0],
                    "pointer": [0.5, 0.5],
                    "sentence": sentence,
                    "sentence_span": [start, start + len(candidate.surface)],
                    "expected_lemma": candidate.lemma,
                    "expected_labels": sorted(candidate.labels),
                    "expected_dictionary_entries": _oracle_json(candidate.entries),
                    "expected_components": _component_json(candidate.components),
                    "expected_spacing": candidate.expected_spacing,
                    "target_class": candidate.target_class,
                    "language_class": candidate.language_class,
                },
                "provenance": {
                    "source_id": candidate.source_id,
                    "source_sample_id": candidate.source_sample_id,
                    "source_split": candidate.source_split,
                    "oracle": PLAIN_ORACLE,
                    "supporting_source_ids": ["krdict-english-json"],
                },
            },
        )

    owned_renderers = renderers is None
    active: dict[str, PlainRenderer] = (
        {"desktop": DesktopRenderer(), "browser": BrowserRenderer()}
        if renderers is None
        else dict(renderers)
    )
    if set(active) != {"desktop", "browser"}:
        raise CorpusError("plain builder requires desktop and browser renderers")
    renderer_versions = {name: renderer.version for name, renderer in active.items()}
    font_hashes: dict[Path, str] = {}
    _write_json(
        corpus / "renderer.json",
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "renderers": renderer_versions,
            "required_sizes_px": list(REQUIRED_SIZES),
            "stress_size_px": STRESS_SIZE,
            "scales_percent": list(SCALE_PERCENTS),
            "punctuation_classes": list(PUNCTUATION_CLASSES),
        },
    )
    quick: list[str] = []
    try:
        for category, values in (
            ("plain", selected),
            ("plain_stress", stress_selected),
        ):
            output = corpus / category
            output.mkdir()
            for index, candidate in enumerate(values):
                size = (
                    STRESS_SIZE
                    if category == "plain_stress"
                    else REQUIRED_SIZES[index % len(REQUIRED_SIZES)]
                )
                spec = _render_spec(index, acquired, acquisition, size)
                if spec.font_path not in font_hashes:
                    font_hashes[spec.font_path] = _sha256(spec.font_path)
                rendered = _punctuate(
                    candidate.words, candidate.target_index, spec.punctuation_class, index
                )
                name = f"{index + 1:04d}"
                lines, target_box = active[spec.renderer].render(
                    rendered, spec, output / f"{name}.png"
                )
                pointer = [
                    (target_box[0] + target_box[2]) / 2,
                    (target_box[1] + target_box[3]) / 2,
                ]
                target_line = next(
                    (
                        line
                        for line in lines
                        if _box_contains(line["box"], pointer)
                        and rendered.target_surface in str(line["text"])
                    ),
                    None,
                )
                if target_line is None:
                    raise CorpusError("renderer did not retain target sentence geometry")
                sentence = str(target_line["text"])
                target_start = sentence.find(rendered.target_surface)
                duplicate_start = sentence.find(rendered.target_surface, target_start + 1)
                if target_start < 0 or duplicate_start >= 0:
                    raise CorpusError("rendered target is not unique in its containing line")
                _write_json(
                    output / f"{name}.json",
                    {
                        "schema_version": 4,
                        "sample_id": f"{profile}-{category}-{name}",
                        "image": f"{name}.png",
                        "lines": lines,
                        "target": {
                            "text": rendered.target_surface,
                            "box": target_box,
                            "pointer": pointer,
                            "sentence": sentence,
                            "sentence_span": [
                                target_start,
                                target_start + len(rendered.target_surface),
                            ],
                            "expected_lemma": candidate.lemma,
                            "expected_labels": sorted(candidate.labels),
                            "expected_dictionary_entries": _oracle_json(candidate.entries),
                            "expected_components": _component_json(candidate.components),
                            "expected_spacing": candidate.expected_spacing,
                            "target_class": candidate.target_class,
                            "language_class": candidate.language_class,
                        },
                        "negative_probes": _negative_probes(lines, target_line, target_box),
                        "render": {
                            "renderer": spec.renderer,
                            "renderer_version": renderer_versions[spec.renderer],
                            "font": spec.font_id,
                            "font_sha256": font_hashes[spec.font_path],
                            "size_px": spec.size_px,
                            "weight": spec.weight,
                            "scale_percent": spec.scale_percent,
                            "theme": spec.theme,
                            "layout": spec.layout,
                            "punctuation": spec.punctuation_class,
                            "stress": category == "plain_stress",
                        },
                        "provenance": {
                            "source_id": candidate.source_id,
                            "source_sample_id": candidate.source_sample_id,
                            "source_split": candidate.source_split,
                            "oracle": PLAIN_ORACLE,
                            "supporting_source_ids": [spec.font_id, "krdict-english-json"],
                        },
                    },
                )
                if category == "plain" and len(quick) < min(QUICK_COUNT, count):
                    quick.append(f"plain/{name}.json")
    finally:
        if owned_renderers:
            browser = active.get("browser")
            close = getattr(browser, "close", None)
            if close:
                close()
    _write_json(corpus / "quick.json", {"schema_version": 1, "samples": quick})
    return {
        "profile": profile,
        "plain": count,
        "plain_stress": stress_count,
        "language": language_count,
        "quick": len(quick),
        "language_classes": {
            language_class: sum(item.language_class == language_class for item in language_selected)
            for language_class in ("multi-lexical", "auxiliary")
        },
        "main_language_coverage": {
            language_class: sum(item.language_class == language_class for item in selected)
            for language_class in ("multi-lexical", "auxiliary")
        },
        "target_classes": {
            target_class: sum(item.target_class == target_class for item in selected[:count])
            for target_class in ("particle", "conjugated", "plain")
        },
        "renderers": renderer_versions,
    }


def _box_contains(value: object, point: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    left, top, right, bottom = (float(item) for item in value)
    return left <= point[0] <= right and top <= point[1] <= bottom


def _negative_probes(
    lines: list[dict[str, object]],
    target_line: dict[str, object],
    target_box: list[float],
) -> list[dict[str, object]]:
    probes: list[dict[str, object]] = [
        {"kind": "english", "pointer": [110.0, 92.0]},
        {"kind": "blank", "pointer": [1_100.0, 100.0]},
    ]
    regions = target_line.get("regions")
    if isinstance(regions, list):
        raw_boxes = sorted(
            (
                [float(part) for part in region["raw_box"]]
                for region in regions
                if isinstance(region, dict) and isinstance(region.get("raw_box"), list)
            ),
            key=lambda box: box[0],
        )
        for left, right in zip(raw_boxes, raw_boxes[1:], strict=False):
            if right[0] - left[2] >= 3:
                probes.append(
                    {
                        "kind": "whitespace",
                        "pointer": [(left[2] + right[0]) / 2, (left[1] + left[3]) / 2],
                    }
                )
                break
        for region in regions:
            if not isinstance(region, dict):
                continue
            raw = region.get("raw_box")
            core = region.get("core_box")
            if not isinstance(raw, list) or not isinstance(core, list):
                continue
            raw_box = [float(part) for part in raw]
            core_box = [float(part) for part in core]
            if any(abs(a - b) > 0.01 for a, b in zip(core_box, target_box, strict=True)):
                continue
            if core_box[0] - raw_box[0] >= 2:
                x = (raw_box[0] + core_box[0]) / 2
            elif raw_box[2] - core_box[2] >= 2:
                x = (core_box[2] + raw_box[2]) / 2
            else:
                break
            probes.append({"kind": "punctuation", "pointer": [x, (raw_box[1] + raw_box[3]) / 2]})
            break
    all_boxes = [
        [float(part) for part in eojeol["box"]]
        for line in lines
        for eojeol in line.get("eojeols", [])  # type: ignore[union-attr]
        if isinstance(eojeol, dict) and isinstance(eojeol.get("box"), list)
    ]
    height = target_box[3] - target_box[1]
    offset = max(4.0, height * 0.25)
    center_x = (target_box[0] + target_box[2]) / 2
    center_y = (target_box[1] + target_box[3]) / 2
    near_candidates = (
        [center_x, target_box[3] + offset],
        [center_x, target_box[1] - offset],
        [target_box[2] + offset, center_y],
        [target_box[0] - offset, center_y],
    )
    near = next(
        (
            point
            for point in near_candidates
            if 5.0 <= point[0] <= 1_275.0
            and 5.0 <= point[1] <= 715.0
            and not any(_box_contains(box, point) for box in all_boxes)
        ),
        None,
    )
    if near is not None:
        probes.append({'kind': 'near-miss', 'pointer': near})
    return probes
