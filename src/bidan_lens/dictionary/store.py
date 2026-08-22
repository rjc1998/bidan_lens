from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from bidan_lens.models import DictionaryEntry, DictionarySense


class DictionaryStore(ABC):
    @abstractmethod
    def lookup(
        self, lemma: str, part_of_speech: str | None = None, limit: int = 10
    ) -> tuple[DictionaryEntry, ...]: ...


class SqliteDictionaryStore(DictionaryStore):
    def __init__(self, path: Path) -> None:
        self.path = path

    def lookup(
        self, lemma: str, part_of_speech: str | None = None, limit: int = 10
    ) -> tuple[DictionaryEntry, ...]:
        if not self.path.exists():
            return ()
        connection = sqlite3.connect(
            f"file:{self.path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            query = """
                SELECT DISTINCT e.id, e.headword, e.part_of_speech,
                       e.homograph_number, e.vocabulary_level, e.source
                FROM entries e
                LEFT JOIN aliases a ON a.entry_id = e.id
                WHERE (e.normalized_headword = ? OR a.normalized_form = ?)
            """
            params: list[object] = [lemma, lemma]
            if part_of_speech:
                query += " AND (e.part_of_speech = ? OR e.part_of_speech IS NULL)"
                params.append(part_of_speech)
            query += " ORDER BY e.frequency_rank IS NULL, e.frequency_rank, e.id LIMIT ?"
            params.append(limit)
            rows = connection.execute(query, params).fetchall()
            entries = []
            for row in rows:
                senses = connection.execute(
                    "SELECT definition, sense_order FROM senses WHERE entry_id = ? "
                    "ORDER BY sense_order",
                    (row["id"],),
                ).fetchall()
                entries.append(
                    DictionaryEntry(
                        entry_id=row["id"],
                        headword=row["headword"],
                        part_of_speech=row["part_of_speech"],
                        homograph_number=row["homograph_number"],
                        vocabulary_level=row["vocabulary_level"],
                        senses=tuple(
                            DictionarySense(s["definition"], s["sense_order"]) for s in senses
                        ),
                        source=row["source"],
                    )
                )
            return tuple(entries)
        finally:
            connection.close()
