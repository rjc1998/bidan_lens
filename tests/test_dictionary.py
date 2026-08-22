import json
import zipfile

from bidan_lens.dictionary.builder import (
    KrdictJsonAdapter,
    SourceEntry,
    SourceSense,
    build_database,
)
from bidan_lens.dictionary.store import SqliteDictionaryStore


def test_build_and_lookup_krdict_database(tmp_path) -> None:
    source = tmp_path / "krdict.json"
    source.write_text(
        json.dumps(
            {
                "LexicalResource": {
                    "Lexicon": {
                        "LexicalEntry": [
                            {
                                "id": "42",
                                "Lemma": {"writtenForm": "먹다"},
                                "partOfSpeech": "동사",
                                "vocabularyLevel": "초급",
                                "Sense": [
                                    {
                                        "Equivalent": {
                                            "language": "eng",
                                            "definition": "to eat",
                                        }
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "dictionary.sqlite3"
    count = build_database(
        KrdictJsonAdapter().entries(source), database, source_version="test-version"
    )
    assert count == 1
    results = SqliteDictionaryStore(database).lookup("먹다", "verb")
    assert results[0].headword == "먹다"
    assert results[0].senses[0].definition == "to eat"
    assert results[0].vocabulary_level == "초급"


def test_official_krdict_archive_shape_is_supported(tmp_path) -> None:
    source = tmp_path / "krdict.zip"
    document = {
        "LexicalResource": {
            "Lexicon": {
                "LexicalEntry": [
                    {
                        "Lemma": {"feat": {"att": "writtenForm", "val": "먹다"}},
                        "Sense": {
                            "Equivalent": [
                                {
                                    "feat": [
                                        {"att": "language", "val": "영어"},
                                        {"att": "lemma", "val": "eat"},
                                        {"att": "definition", "val": "To consume food."},
                                    ]
                                }
                            ]
                        },
                        "att": "id",
                        "val": "42",
                        "feat": [
                            {"att": "partOfSpeech", "val": "동사"},
                            {"att": "vocabularyLevel", "val": "초급"},
                        ],
                    }
                ]
            }
        }
    }
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("1_5000.json", json.dumps(document, ensure_ascii=False))

    entries = tuple(KrdictJsonAdapter().entries(source))

    assert len(entries) == 1
    assert entries[0].entry_id == "42"
    assert entries[0].headword == "먹다"
    assert entries[0].part_of_speech == "verb"
    assert entries[0].vocabulary_level == "초급"
    assert entries[0].senses[0].definition == "To consume food."


def test_reused_source_ids_do_not_overwrite_related_entries(tmp_path) -> None:
    entries = (
        SourceEntry("42", "첫", "determiner", "0", None, (SourceSense("first", 1),)),
        SourceEntry(
            "42",
            "첫 단추를 끼우다",
            None,
            None,
            None,
            (SourceSense("to make a good start", 1),),
        ),
    )
    database = tmp_path / "dictionary.sqlite3"

    assert build_database(entries, database, source_version="test") == 2
    assert SqliteDictionaryStore(database).lookup("첫", "determiner")
    assert SqliteDictionaryStore(database).lookup("첫 단추를 끼우다")
