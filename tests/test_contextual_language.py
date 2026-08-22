from __future__ import annotations

from dataclasses import dataclass

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.gui.popup import _breakdown_text, _definitions_text
from bidan_lens.models import DictionaryEntry, DictionarySense


@dataclass
class Token:
    form: str
    tag: str
    start: int
    len: int


class ContextKiwi:
    def __init__(self, analyses):  # type: ignore[no-untyped-def]
        self.analyses = analyses

    def analyze(self, text: str, top_n: int = 1):  # type: ignore[no-untyped-def]
        return self.analyses[text][:top_n]

    def space(self, text: str, reset_whitespace: bool = False) -> str:
        return text


def _entry(entry_id: str, headword: str, pos: str, definition: str) -> DictionaryEntry:
    return DictionaryEntry(
        entry_id,
        headword,
        pos,
        None,
        None,
        (DictionarySense(definition, 1),),
    )


class RoleDictionary(DictionaryStore):
    values = {
        '가다': (_entry('go', '가다', 'verb', 'to go'),),
        '오다': (_entry('come', '오다', 'verb', 'to come'),),
        '버리다': (
            _entry('discard', '버리다', 'verb', 'to throw away'),
            _entry('complete', '버리다', '보조 동사', 'marks completion'),
        ),
        '먹다': (
            _entry('hear', '먹다', 'verb', 'to become unable to hear'),
            _entry('eat', '먹다', 'verb', 'to eat'),
        ),
    }

    def lookup(
        self, lemma: str, part_of_speech: str | None = None, limit: int = 10
    ) -> tuple[DictionaryEntry, ...]:
        entries = self.values.get(lemma, ())
        if part_of_speech is not None:
            entries = tuple(
                entry for entry in entries if entry.part_of_speech == part_of_speech
            )
        return entries[:limit]


def _analyzer() -> KoreanAnalyzer:
    analyses = {
        '갔다오다': [
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VV', 2, 1),
                    Token('다', 'EF', 3, 1),
                ],
                -1.0,
            )
        ],
        '먹어 버리다': [
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('어', 'EC', 1, 1),
                    Token('버리', 'VX', 3, 2),
                    Token('다', 'EF', 5, 1),
                ],
                -1.0,
            )
        ],
        '쓰레기를 버리다': [
            (
                [
                    Token('쓰레기', 'NNG', 0, 3),
                    Token('를', 'JKO', 3, 1),
                    Token('버리', 'VV', 5, 2),
                    Token('다', 'EF', 7, 1),
                ],
                -1.0,
            )
        ],
    }
    return KoreanAnalyzer(RoleDictionary(), ContextKiwi(analyses))


def test_unspaced_multi_lexical_eojeol_keeps_both_definitions() -> None:
    candidate = _analyzer().analyze('갔다오다', (0, 4))[0]

    assert [part.lemma for part in candidate.lexical_components] == ['가다', '오다']
    assert candidate.lexical_components[1].dictionary_entries[0].senses[0].definition == 'to come'
    assert candidate.interpreted_surface == '갔다 오다'
    assert '오 → 오다: action verb — to come' in _breakdown_text(candidate)


def test_context_selects_auxiliary_entry_and_explanation() -> None:
    candidate = _analyzer().analyze('먹어 버리다', (3, 6))[0]

    component = candidate.lexical_components[0]
    assert component.learner_role == 'helping verb'
    assert component.dictionary_entries[0].entry_id == 'complete'
    assert component.contextual_explanation == 'indicates completion of the preceding action'
    rendered = _definitions_text(candidate)
    assert rendered.index('marks completion') < rendered.index('to throw away')


def test_main_verb_leads_with_ordinary_dictionary_group() -> None:
    candidate = _analyzer().analyze('쓰레기를 버리다', (5, 8))[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'
    assert candidate.dictionary_entries[0].entry_id == 'discard'
    assert candidate.dictionary_entries[1].entry_id == 'complete'


def test_same_role_homographs_remain_source_ordered() -> None:
    entries = KoreanAnalyzer(RoleDictionary())._ordered_entries('먹다', 'VV')

    assert [entry.entry_id for entry in entries] == ['hear', 'eat']
