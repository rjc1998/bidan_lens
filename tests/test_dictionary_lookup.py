from bidan_lens.dictionary.builder import SourceEntry, SourceSense, build_database
from bidan_lens.dictionary.store import SqliteDictionaryStore


def test_exact_headword_takes_precedence_over_aliases(tmp_path) -> None:
    entries = (
        SourceEntry(
            'direct',
            'ê°€ë‹¤',
            'verb',
            None,
            None,
            (SourceSense('to go', 1),),
        ),
        SourceEntry(
            'alias',
            'ê°€ë‹¤ë“¬ë‹¤',
            'verb',
            None,
            None,
            (SourceSense('to trim', 1),),
            aliases=('ê°€ë‹¤',),
        ),
    )
    database = tmp_path / 'dictionary.sqlite3'
    build_database(entries, database, source_version='test')

    results = SqliteDictionaryStore(database).lookup('ê°€ë‹¤', 'verb')

    assert [entry.entry_id for entry in results] == ['direct']


def test_dictionary_alias_remains_available_as_fallback(tmp_path) -> None:
    entry = SourceEntry(
        'alias',
        'ê°€ë‹¤ë“¬ë‹¤',
        'verb',
        None,
        None,
        (SourceSense('to trim', 1),),
        aliases=('ë‹¤ë“¬ë‹¤',),
    )
    database = tmp_path / 'dictionary.sqlite3'
    build_database((entry,), database, source_version='test')

    results = SqliteDictionaryStore(database).lookup('ë‹¤ë“¬ë‹¤', 'verb')

    assert [item.entry_id for item in results] == ['alias']
