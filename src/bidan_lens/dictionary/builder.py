from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import unicodedata
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceSense:
    definition: str
    order: int


@dataclass(frozen=True, slots=True)
class SourceEntry:
    entry_id: str
    headword: str
    part_of_speech: str | None
    homograph_number: str | None
    vocabulary_level: str | None
    senses: tuple[SourceSense, ...]
    aliases: tuple[str, ...] = ()


class DictionarySourceAdapter(ABC):
    @abstractmethod
    def entries(self, source_path: Path) -> Iterator[SourceEntry]: ...


def _text(value: Any, *keys: str) -> str | None:
    for key in keys:
        if isinstance(value, dict) and value.get(key) not in (None, ""):
            candidate = value[key]
            if isinstance(candidate, dict):
                candidate = candidate.get("#text") or candidate.get("writtenForm")
            if candidate not in (None, ""):
                return str(candidate)
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _feature(value: Any, name: str) -> str | None:
    if not isinstance(value, dict):
        return None
    for feature in _as_list(value.get("feat")):
        if isinstance(feature, dict) and feature.get("att") == name:
            result = feature.get("val")
            if result not in (None, ""):
                return str(result)
    return None


def _node_id(value: Any, fallback: str) -> str:
    if not isinstance(value, dict):
        return fallback
    if value.get("att") == "id" and value.get("val") not in (None, ""):
        return str(value["val"])
    return str(value.get("id") or fallback)


def _documents(source_path: Path) -> Iterator[dict[str, Any]]:
    if not zipfile.is_zipfile(source_path):
        yield json.loads(source_path.read_text(encoding="utf-8-sig"))
        return
    with zipfile.ZipFile(source_path) as archive:
        members = sorted(
            (item for item in archive.infolist() if item.filename.lower().endswith(".json")),
            key=lambda item: item.filename,
        )
        if not members:
            raise ValueError("KRDict archive contains no JSON documents")
        for member in members:
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as text:
                yield json.load(text)


class KrdictJsonAdapter(DictionarySourceAdapter):
    """Read the official KRDict JSON export while tolerating minor schema variations."""

    def entries(self, source_path: Path) -> Iterator[SourceEntry]:
        for document_index, data in enumerate(_documents(source_path)):
            lexical = data.get("LexicalResource", data)
            lexicon = lexical.get("Lexicon", lexical) if isinstance(lexical, dict) else lexical
            raw_entries = lexicon.get("LexicalEntry", []) if isinstance(lexicon, dict) else []
            for index, raw in enumerate(_as_list(raw_entries)):
                lemma_value = raw.get("Lemma", {})
                headword = (
                    _feature(lemma_value, "writtenForm")
                    or _text(lemma_value, "writtenForm", "#text")
                    or _text(raw, "headword", "word")
                )
                if not headword:
                    continue
                senses: list[SourceSense] = []
                for order, sense in enumerate(_as_list(raw.get("Sense")), start=1):
                    equivalents = _as_list(sense.get("Equivalent") or sense.get("equivalent"))
                    definitions = []
                    for equivalent in equivalents:
                        language = (
                            _feature(equivalent, "language")
                            or _text(equivalent, "language", "lang")
                            or "eng"
                        ).lower()
                        if language not in {"eng", "en", "english", "영어"}:
                            continue
                        definition = _feature(equivalent, "definition") or _text(
                            equivalent, "definition", "translation", "writtenForm", "#text"
                        )
                        if not definition:
                            definition = _feature(equivalent, "lemma")
                        if definition:
                            definitions.append(definition)
                    direct = _text(sense, "english_definition", "definition_en", "definition")
                    if direct and not definitions:
                        definitions.append(direct)
                    for definition in definitions:
                        senses.append(SourceSense(definition.strip(), order))
                if not senses:
                    continue
                yield SourceEntry(
                    entry_id=_node_id(raw, f"krdict-{document_index}-{index}"),
                    headword=headword.strip(),
                    part_of_speech=normalize_part_of_speech(
                        _feature(raw, "partOfSpeech")
                        or _text(raw, "partOfSpeech", "part_of_speech")
                    ),
                    homograph_number=_feature(raw, "homonym_number")
                    or _text(raw, "homonym_number", "homograph_number"),
                    vocabulary_level=_feature(raw, "vocabularyLevel")
                    or _text(raw, "vocabularyLevel", "vocabulary_level"),
                    senses=tuple(senses),
                )


SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE entries (
    id TEXT PRIMARY KEY,
    headword TEXT NOT NULL,
    normalized_headword TEXT NOT NULL,
    part_of_speech TEXT,
    homograph_number TEXT,
    vocabulary_level TEXT,
    frequency_rank INTEGER,
    source TEXT NOT NULL
);
CREATE INDEX entries_headword_idx ON entries(normalized_headword);
CREATE TABLE senses (
    entry_id TEXT NOT NULL REFERENCES entries(id),
    sense_order INTEGER NOT NULL,
    definition TEXT NOT NULL,
    PRIMARY KEY (entry_id, sense_order, definition)
);
CREATE TABLE aliases (
    entry_id TEXT NOT NULL REFERENCES entries(id),
    form TEXT NOT NULL,
    normalized_form TEXT NOT NULL,
    PRIMARY KEY (entry_id, normalized_form)
);
CREATE INDEX aliases_form_idx ON aliases(normalized_form);
"""

_PARTS_OF_SPEECH = {
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


def normalize_part_of_speech(value: str | None) -> str | None:
    if value is None:
        return None
    return _PARTS_OF_SPEECH.get(value.strip(), value.strip().lower())


def normalize_lookup(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def build_database(
    entries: Iterable[SourceEntry], destination: Path, *, source_version: str
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    count = 0
    used_ids: dict[str, tuple[str, str | None, str | None]] = {}
    try:
        connection.executescript(SCHEMA)
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", "1"))
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("source", "KRDict"))
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("source_version", source_version))
        for entry in entries:
            signature = (
                normalize_lookup(entry.headword),
                entry.part_of_speech,
                entry.homograph_number,
            )
            entry_id = entry.entry_id
            if entry_id in used_ids and used_ids[entry_id] != signature:
                suffix = hashlib.sha256(
                    "\0".join(value or "" for value in signature).encode("utf-8")
                ).hexdigest()[:16]
                entry_id = f"{entry.entry_id}:{suffix}"
            is_new = entry_id not in used_ids
            used_ids[entry_id] = signature
            connection.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    entry.headword,
                    normalize_lookup(entry.headword),
                    entry.part_of_speech,
                    entry.homograph_number,
                    entry.vocabulary_level,
                    None,
                    "KRDict",
                ),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO senses VALUES (?, ?, ?)",
                ((entry_id, sense.order, sense.definition) for sense in entry.senses),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO aliases VALUES (?, ?, ?)",
                ((entry_id, alias, normalize_lookup(alias)) for alias in entry.aliases),
            )
            count += int(is_new)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    temporary.replace(destination)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BiDan Lens KRDict database")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-version", required=True)
    arguments = parser.parse_args()
    count = build_database(
        KrdictJsonAdapter().entries(arguments.source),
        arguments.destination,
        source_version=arguments.source_version,
    )
    print(f"Built {arguments.destination} with {count} entries")


if __name__ == "__main__":
    main()
