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
        return self.analyses.get(text, ())[:top_n]

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
        '먹어 치우다': [
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('어', 'EC', 1, 1),
                    Token('치우', 'VV', 3, 2),
                    Token('다', 'EF', 5, 1),
                ],
                -1.0,
            )
        ],
        '먹어 놓다': [
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('어', 'EC', 1, 1),
                    Token('놓', 'VV', 3, 1),
                    Token('다', 'EF', 4, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('어', 'EC', 1, 1),
                    Token('놓', 'VX', 3, 1),
                    Token('다', 'EF', 4, 1),
                ],
                -2.0,
            ),
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


def test_context_recovers_auxiliary_role_when_kiwi_returns_action_verb() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '치우다': (
            _entry('clear', '치우다', 'verb', 'to clear away'),
            _entry('finish', '치우다', '보조 동사', 'marks emphatic completion'),
        ),
    }
    analyzer = KoreanAnalyzer(dictionary, _analyzer().kiwi)

    candidate = analyzer.analyze('먹어 치우다', (3, 6))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'
    assert candidate.lexical_components[0].dictionary_entries[0].entry_id == 'finish'


def test_reported_speech_verb_is_not_relabelled_as_an_auxiliary() -> None:
    sentence = '말한다고 하다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '하다': (
            _entry('do', '하다', 'verb', 'to do'),
            _entry('aux', '하다', '보조 동사', 'auxiliary use'),
        ),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('말하', 'VV', 0, 2),
                    Token('다고', 'EC', 2, 2),
                    Token('하', 'VV', 5, 1),
                    Token('다', 'EF', 6, 1),
                ],
                -1.0,
            )
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (5, 7)
    )[0]

    component = candidate.lexical_components[0]
    assert component.learner_role == 'action verb'
    assert component.dictionary_entries[0].entry_id == 'do'


def test_prefixed_reported_speech_verb_is_not_relabelled_as_an_auxiliary() -> None:
    sentence = '찍으라고 하다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '하다': (
            _entry('do', '하다', 'verb', 'to do'),
            _entry('aux', '하다', '보조 동사', 'auxiliary use'),
        ),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('찍', 'VV', 0, 1),
                    Token('으라고', 'EC', 1, 3),
                    Token('하', 'VV', 5, 1),
                    Token('다', 'EF', 6, 1),
                ],
                -1.0,
            )
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (5, 7)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'
    assert candidate.dictionary_entries[0].entry_id == 'do'


def test_nominal_rado_context_does_not_relabel_a_verb_as_auxiliary() -> None:
    sentence = '올이라도 가지다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '가지다': (
            _entry('have', '가지다', 'verb', 'to have'),
            _entry('aux-have', '가지다', '보조 동사', 'auxiliary use'),
        ),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('올', 'NNG', 0, 1),
                    Token('이', 'VCP', 1, 1),
                    Token('라도', 'EC', 2, 2),
                    Token('가지', 'VV', 5, 2),
                    Token('다', 'EF', 7, 1),
                ],
                -1.0,
            )
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (5, 8)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'
    assert candidate.dictionary_entries[0].entry_id == 'have'


def test_gido_context_promotes_a_close_helping_verb_alternative() -> None:
    sentence = '먹기도 하다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '하다': (
            _entry('do', '하다', 'verb', 'to do'),
            _entry('aux', '하다', '보조 동사', 'auxiliary use'),
        ),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('기', 'ETN', 1, 1),
                    Token('도', 'JX', 2, 1),
                    Token('하', 'VV', 4, 1),
                    Token('다', 'EF', 5, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('먹', 'VV', 0, 1),
                    Token('기', 'ETN', 1, 1),
                    Token('도', 'JX', 2, 1),
                    Token('하', 'VX', 4, 1),
                    Token('다', 'EF', 5, 1),
                ],
                -1.7,
            ),
        ],
        '하다': [
            (
                [
                    Token('하', 'VV', 0, 1),
                    Token('다', 'EF', 1, 1),
                ],
                -1.0,
            )
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (4, 6)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'
    assert candidate.dictionary_entries[0].entry_id == 'aux'


def test_context_reranks_kiwi_helping_alternative_after_connective() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '놓다': (_entry('put', '놓다', 'verb', 'to put'),),
    }
    analyzer = KoreanAnalyzer(dictionary, _analyzer().kiwi)

    candidates = analyzer.analyze('먹어 놓다', (3, 5))

    assert candidates[0].lexical_components[0].learner_role == 'helping verb'
    assert candidates[1].lexical_components[0].learner_role == 'action verb'


def test_context_does_not_promote_distant_same_lemma_helping_alternative() -> None:
    sentence = '\uba39\uc5b4 \ub193\ub2e4'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\ub193\ub2e4': (_entry('put', '\ub193\ub2e4', 'verb', 'to put'),),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('\uba39', 'VV', 0, 1),
                    Token('\uc5b4', 'EC', 1, 1),
                    Token('\ub193', 'VV', 3, 1),
                    Token('\ub2e4', 'EF', 4, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('\uba39', 'VV', 0, 1),
                    Token('\uc5b4', 'EC', 1, 1),
                    Token('\ub193', 'VX', 3, 1),
                    Token('\ub2e4', 'EF', 4, 1),
                ],
                -4.0,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 5)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'


def test_ge_doeda_promotes_distant_same_lemma_helping_alternative() -> None:
    sentence = '\ubcf4\uac8c \ub418\ub294'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\ub418\ub2e4': (_entry('become', '\ub418\ub2e4', 'verb', 'to become'),),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('\ubcf4', 'VV', 0, 1),
                    Token('\uac8c', 'EC', 1, 1),
                    Token('\ub418', 'VV', 3, 1),
                    Token('\ub294', 'ETM', 4, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('\ubcf4', 'VV', 0, 1),
                    Token('\uac8c', 'EC', 1, 1),
                    Token('\ub418', 'VX', 3, 1),
                    Token('\ub294', 'ETM', 4, 1),
                ],
                -10.5,
            ),
        ],
        '\ub418\ub294': [
            (
                [
                    Token('\ub418', 'VV', 0, 1),
                    Token('\ub294', 'ETM', 1, 1),
                ],
                -1.0,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 5)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_ge_doeda_promotes_a_contracted_helping_alternative() -> None:
    sentence = '보게 된다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '되다': (_entry('become', '되다', 'verb', 'to become'),),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('보', 'VV', 0, 1),
                    Token('게', 'EC', 1, 1),
                    Token('되', 'VV', 3, 1),
                    Token('ᆫ다', 'EF', 3, 2),
                ],
                -1.0,
            ),
            (
                [
                    Token('보', 'VV', 0, 1),
                    Token('게', 'EC', 1, 1),
                    Token('되', 'VX', 3, 1),
                    Token('ᆫ다', 'EF', 3, 2),
                ],
                -10.5,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 5)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_context_still_promotes_distant_different_lemma_helping_analysis() -> None:
    sentence = '\uba39\uc5b4 \ub193\ub2e4'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\ub193\ub2e4': (_entry('put', '\ub193\ub2e4', 'verb', 'to put'),),
        '\ub193\uc544\ub450\ub2e4': (
            _entry('keep', '\ub193\uc544\ub450\ub2e4', 'verb', 'to leave in place'),
        ),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('\uba39', 'VV', 0, 1),
                    Token('\uc5b4', 'EC', 1, 1),
                    Token('\ub193', 'VV', 3, 1),
                    Token('\ub2e4', 'EF', 4, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('\uba39', 'VV', 0, 1),
                    Token('\uc5b4', 'EC', 1, 1),
                    Token('\ub193\uc544\ub450', 'VX', 3, 1),
                    Token('\ub2e4', 'EF', 4, 1),
                ],
                -4.0,
            ),
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 5)
    )[0]

    assert candidate.lemma == '\ub193\uc544\ub450\ub2e4'
    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_context_recovers_auxiliary_role_across_punctuation() -> None:
    sentence = '\uba39\uc5b4 [\ubc84\ub9ac\ub2e4]'
    analyses = {
        sentence: [
            (
                [
                    Token('\uba39', 'VV', 0, 1),
                    Token('\uc5b4', 'EC', 1, 1),
                    Token('[', 'SSO', 3, 1),
                    Token('\ubc84\ub9ac', 'VV', 4, 2),
                    Token('\ub2e4', 'EF', 6, 1),
                    Token(']', 'SSC', 7, 1),
                ],
                -1.0,
            )
        ]
    }

    candidate = KoreanAnalyzer(RoleDictionary(), ContextKiwi(analyses)).analyze(
        sentence, (4, 7)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_close_candidate_supported_by_unwrapped_context_is_promoted() -> None:
    sentence = '\ub9d0\ud560 \uc218 \u2018\uc788\ub2e4\u2019'
    unwrapped = '\ub9d0\ud560 \uc218 \uc788\ub2e4'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\uc788\ub2e4': (
            _entry('remain', '\uc788\ub2e4', 'verb', 'to remain'),
            _entry('exist', '\uc788\ub2e4', 'adjective', 'to exist'),
        ),
    }
    analyses = {
        sentence: [
            ([Token('\uc788', 'VV', 6, 1), Token('\ub2e4', 'EF', 7, 1)], -1.0),
            ([Token('\uc788', 'VA', 6, 1), Token('\ub2e4', 'EF', 7, 1)], -1.4),
        ],
        unwrapped: [
            ([Token('\uc788', 'VA', 5, 1), Token('\ub2e4', 'EF', 6, 1)], -1.0),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (6, 8)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'descriptive verb'


def test_unwrapped_context_can_supply_a_missing_role_only_candidate() -> None:
    sentence = '“안” 먹다'
    unwrapped = '안 먹다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '안': (
            _entry('not', '안', 'adverb', 'not'),
            _entry('inside', '안', 'noun', 'inside'),
        ),
    }
    analyses = {
        sentence: [([Token('안', 'NNP', 1, 1)], -1.0)],
        unwrapped: [([Token('안', 'MAG', 0, 1)], -1.0)],
    }

    candidates = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (1, 2)
    )

    assert candidates[0].lexical_components[0].learner_role == 'adverb'
    assert candidates[1].lexical_components[0].learner_role == 'name or proper noun'


def test_unwrapped_context_does_not_supply_a_different_lemma() -> None:
    sentence = '“한”'
    unwrapped = '한'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '한': (_entry('han', '한', 'noun', 'a name'),),
        '하다': (_entry('do', '하다', 'verb', 'to do'),),
    }
    analyses = {
        sentence: [([Token('한', 'NNP', 1, 1)], -1.0)],
        unwrapped: [
            ([Token('하', 'VV', 0, 1), Token('ᆫ', 'ETM', 0, 1)], -1.0)
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (1, 2)
    )[0]

    assert candidate.lemma == '한'
    assert candidate.lexical_components[0].learner_role == 'name or proper noun'


def test_unwrapped_context_does_not_promote_distant_candidate() -> None:
    sentence = '\ub9d0\ud560 \uc218 \u2018\uc788\ub2e4\u2019'
    unwrapped = '\ub9d0\ud560 \uc218 \uc788\ub2e4'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\uc788\ub2e4': (
            _entry('remain', '\uc788\ub2e4', 'verb', 'to remain'),
            _entry('exist', '\uc788\ub2e4', 'adjective', 'to exist'),
        ),
    }
    analyses = {
        sentence: [
            ([Token('\uc788', 'VV', 6, 1), Token('\ub2e4', 'EF', 7, 1)], -1.0),
            ([Token('\uc788', 'VA', 6, 1), Token('\ub2e4', 'EF', 7, 1)], -3.0),
        ],
        unwrapped: [
            ([Token('\uc788', 'VA', 5, 1), Token('\ub2e4', 'EF', 6, 1)], -1.0),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (6, 8)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'


def test_isolated_analysis_can_disambiguate_a_close_verb_role() -> None:
    sentence = '“있는” 말'
    surface = '있는'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '있다': (
            _entry('remain', '있다', 'verb', 'to remain'),
            _entry('exist', '있다', 'adjective', 'to exist'),
        ),
    }
    analyses = {
        sentence: [
            ([Token('있', 'VV', 1, 1), Token('는', 'ETM', 2, 1)], -1.0),
            ([Token('있', 'VA', 1, 1), Token('는', 'ETM', 2, 1)], -1.5),
        ],
        surface: [
            ([Token('있', 'VA', 0, 1), Token('는', 'ETM', 1, 1)], -1.0),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (1, 3)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'descriptive verb'


def test_isolated_verb_role_support_does_not_override_a_distant_candidate() -> None:
    sentence = '“있는” 말'
    surface = '있는'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '있다': (
            _entry('remain', '있다', 'verb', 'to remain'),
            _entry('exist', '있다', 'adjective', 'to exist'),
        ),
    }
    analyses = {
        sentence: [
            ([Token('있', 'VV', 1, 1), Token('는', 'ETM', 2, 1)], -1.0),
            ([Token('있', 'VA', 1, 1), Token('는', 'ETM', 2, 1)], -3.5),
        ],
        surface: [
            ([Token('있', 'VA', 0, 1), Token('는', 'ETM', 1, 1)], -1.0),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (1, 3)
    )[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'


def test_isolated_fallback_recovers_a_contracted_dependent_noun_and_copula() -> None:
    sentence = '[테야]'
    surface = '테야'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '터': (_entry('intention', '터', 'noun', 'intention'),),
    }
    analyses = {
        sentence: [([Token('테야', 'NNG', 1, 2)], -1.0)],
        surface: [
            (
                [
                    Token('터', 'NNB', 0, 1),
                    Token('이', 'VCP', 0, 1),
                    Token('야', 'EF', 1, 1),
                ],
                -1.0,
            )
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (1, 3)
    )[0]

    assert candidate.lemma == '터'
    assert candidate.lexical_components[0].learner_role == 'dependent noun'


def test_main_verb_leads_with_ordinary_dictionary_group() -> None:
    candidate = _analyzer().analyze('쓰레기를 버리다', (5, 8))[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'
    assert candidate.dictionary_entries[0].entry_id == 'discard'
    assert candidate.dictionary_entries[1].entry_id == 'complete'


def test_same_role_homographs_remain_source_ordered() -> None:
    entries = KoreanAnalyzer(RoleDictionary())._ordered_entries('먹다', 'VV')

    assert [entry.entry_id for entry in entries] == ['hear', 'eat']


def test_close_complete_multi_component_analysis_is_promoted() -> None:
    analyses = {
        '갔다오다': [
            ([Token('갔다오', 'VV', 0, 3), Token('다', 'EF', 3, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VV', 2, 1),
                    Token('다', 'EF', 3, 1),
                ],
                -1.5,
            ),
        ]
    }

    candidate = KoreanAnalyzer(RoleDictionary(), ContextKiwi(analyses)).analyze(
        '갔다오다', (0, 4)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '가다',
        '오다',
    ]


def test_complete_multi_component_analysis_is_not_fragmented_further() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '이중': (_entry('double', '이중', 'noun', 'double'),),
        '의도': (_entry('intention', '의도', 'noun', 'intention'),),
        '이': (_entry('this', '이', 'determiner', 'this'),),
        '중': (_entry('middle', '중', 'noun', 'middle'),),
    }
    analyses = {
        '이중의도이다': [
            (
                [
                    Token('이중', 'NNG', 0, 2),
                    Token('의도', 'NNG', 2, 2),
                    Token('이', 'VCP', 4, 1),
                    Token('다', 'EF', 5, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('이', 'MM', 0, 1),
                    Token('중', 'NNG', 1, 1),
                    Token('의도', 'NNG', 2, 2),
                    Token('이', 'VCP', 4, 1),
                    Token('다', 'EF', 5, 1),
                ],
                -2.5,
            ),
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        '이중의도이다', (0, 6)
    )[0]

    assert candidate.lemma == '이중'
    assert [component.lemma for component in candidate.lexical_components] == [
        '이중',
        '의도',
    ]


def test_distant_multi_component_analysis_is_not_promoted() -> None:
    analyses = {
        '갔다오다': [
            ([Token('갔다오', 'VV', 0, 3), Token('다', 'EF', 3, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VV', 2, 1),
                    Token('다', 'EF', 3, 1),
                ],
                -3.5,
            ),
        ]
    }

    candidate = KoreanAnalyzer(RoleDictionary(), ContextKiwi(analyses)).analyze(
        '갔다오다', (0, 4)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == ['갔다오다']


def test_isolated_analysis_can_corroborate_a_contextually_distant_decomposition() -> None:
    sentence = '문맥 갔다오는'
    surface = '갔다오는'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '갔다오다': (_entry('round-trip', '갔다오다', 'verb', 'make a round trip'),),
    }
    compound = [Token('갔다오', 'VV', 3, 3), Token('는', 'ETM', 6, 1)]
    decomposed = [
        Token('가', 'VV', 3, 1),
        Token('었', 'EP', 3, 1),
        Token('다', 'EC', 4, 1),
        Token('오', 'VX', 5, 1),
        Token('는', 'ETM', 6, 1),
    ]
    isolated_compound = [Token('갔다오', 'VV', 0, 3), Token('는', 'ETM', 3, 1)]
    isolated_decomposed = [
        Token('가', 'VV', 0, 1),
        Token('었', 'EP', 0, 1),
        Token('다', 'EC', 1, 1),
        Token('오', 'VX', 2, 1),
        Token('는', 'ETM', 3, 1),
    ]
    analyses = {
        sentence: [(compound, -1.0), (decomposed, -6.0)],
        surface: [(isolated_compound, -1.0), (isolated_decomposed, -4.5)],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 7)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '가다',
        '오다',
    ]


def test_distant_isolated_decomposition_does_not_override_context() -> None:
    sentence = '문맥 갔다오는'
    surface = '갔다오는'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '갔다오다': (_entry('round-trip', '갔다오다', 'verb', 'make a round trip'),),
    }
    analyses = {
        sentence: [
            ([Token('갔다오', 'VV', 3, 3), Token('는', 'ETM', 6, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 3, 1),
                    Token('었', 'EP', 3, 1),
                    Token('다', 'EC', 4, 1),
                    Token('오', 'VX', 5, 1),
                    Token('는', 'ETM', 6, 1),
                ],
                -6.0,
            ),
        ],
        surface: [
            ([Token('갔다오', 'VV', 0, 3), Token('는', 'ETM', 3, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VX', 2, 1),
                    Token('는', 'ETM', 3, 1),
                ],
                -6.0,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 7)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '갔다오다'
    ]


def test_isolated_decomposition_does_not_override_a_dictionary_base_form() -> None:
    sentence = '문맥 갔다오다'
    surface = '갔다오다'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '갔다오다': (_entry('round-trip', '갔다오다', 'verb', 'make a round trip'),),
    }
    analyses = {
        sentence: [
            ([Token('갔다오', 'VV', 3, 3), Token('다', 'EF', 6, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 3, 1),
                    Token('었', 'EP', 3, 1),
                    Token('다', 'EC', 4, 1),
                    Token('오', 'VX', 5, 1),
                    Token('다', 'EF', 6, 1),
                ],
                -6.0,
            ),
        ],
        surface: [
            ([Token('갔다오', 'VV', 0, 3), Token('다', 'EF', 3, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VX', 2, 1),
                    Token('다', 'EF', 3, 1),
                ],
                -4.5,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (3, 7)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '갔다오다'
    ]


def test_multi_component_promotion_does_not_discard_particle_feature() -> None:
    sentence = '\uc11c\uc6b8\ub3c4'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\uc11c\uc6b8': (_entry('place', '\uc11c\uc6b8', 'noun', 'place'),),
        '\ub3c4': (_entry('province', '\ub3c4', 'noun', 'province'),),
    }
    analyses = {
        sentence: [
            (
                [Token('\uc11c\uc6b8', 'NNP', 0, 2), Token('\ub3c4', 'JX', 2, 1)],
                -1.0,
            ),
            (
                [Token('\uc11c\uc6b8', 'NNP', 0, 2), Token('\ub3c4', 'NNG', 2, 1)],
                -1.5,
            ),
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (0, 3)
    )[0]

    assert len(candidate.lexical_components) == 1
    assert 'particle' in {feature.label for feature in candidate.features}


def test_defined_pronoun_is_not_replaced_by_fragmented_analysis() -> None:
    sentence = '\uadf8\uac83\uc740'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '\uadf8': (_entry('that-determiner', '\uadf8', 'determiner', 'that'),),
        '\uac83': (_entry('thing', '\uac83', 'noun', 'thing'),),
        '\uadf8\uac83': (_entry('that-pronoun', '\uadf8\uac83', 'noun', 'that'),),
    }
    analyses = {
        sentence: [
            (
                [
                    Token('\uadf8', 'MM', 0, 1),
                    Token('\uac83', 'NNB', 1, 1),
                    Token('\uc740', 'JX', 2, 1),
                ],
                -1.0,
            ),
            (
                [Token('\uadf8\uac83', 'NP', 0, 2), Token('\uc740', 'JX', 2, 1)],
                -0.7,
            ),
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (0, 3)
    )[0]

    assert candidate.lemma == '\uadf8\uac83'
    assert [component.learner_role for component in candidate.lexical_components] == [
        'pronoun'
    ]


def test_defined_noun_is_not_replaced_by_fragmented_analysis() -> None:
    sentence = '고전에'
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '고': (_entry('old', '고', 'noun', 'old'),),
        '전': (_entry('former', '전', 'noun', 'former'),),
        '고전': (_entry('classic', '고전', 'noun', 'classic'),),
    }
    analyses = {
        sentence: [
            (
                [Token('고전', 'NNG', 0, 2), Token('에', 'JKB', 2, 1)],
                -1.0,
            ),
            (
                [
                    Token('고', 'NNG', 0, 1),
                    Token('전', 'NNG', 1, 1),
                    Token('에', 'JKB', 2, 1),
                ],
                -1.3,
            ),
        ]
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        sentence, (0, 3)
    )[0]

    assert candidate.lemma == '고전'
    assert [component.lemma for component in candidate.lexical_components] == ['고전']


def test_close_inflected_verb_after_particle_leads_noun_homograph() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '대해': (_entry('ocean', '대해', 'noun', 'ocean'),),
        '대하다': (_entry('regarding', '대하다', 'verb', 'regard'),),
    }
    analyses = {
        '문제에 [대해]': [
            (
                [
                    Token('문제', 'NNG', 0, 2),
                    Token('에', 'JKB', 2, 1),
                    Token('[', 'SS', 4, 1),
                    Token('대해', 'NNG', 5, 2),
                    Token(']', 'SS', 7, 1),
                ],
                -1.0,
            ),
            (
                [
                    Token('문제', 'NNG', 0, 2),
                    Token('에', 'JKB', 2, 1),
                    Token('[', 'SS', 4, 1),
                    Token('대하', 'VV', 5, 2),
                    Token('어', 'EC', 6, 1),
                    Token(']', 'SS', 7, 1),
                ],
                -1.3,
            ),
        ],
        '문제에 대해': [
            (
                [
                    Token('문제', 'NNG', 0, 2),
                    Token('에', 'JKB', 2, 1),
                    Token('대하', 'VV', 4, 2),
                    Token('어', 'EC', 5, 1),
                ],
                -1.0,
            )
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        '문제에 [대해]', (5, 7)
    )[0]

    assert candidate.lemma == '대하다'


def test_isolated_fallback_rejects_unrepresented_word_part() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '마피아': (_entry('mafia', '마피아', 'noun', 'mafia'),),
        '의': (_entry('meaning', '의', 'noun', 'meaning'),),
    }
    analyses = {
        '오늘 마피아끼리의': [
            ([Token('마피아끼리', 'NNG', 3, 5), Token('의', 'JX', 8, 1)], -1.0)
        ],
        '마피아끼리의': [
            ([Token('마피아끼리', 'NNG', 0, 5), Token('의', 'JX', 5, 1)], -1.0),
            (
                [
                    Token('마피아', 'NNG', 0, 3),
                    Token('끼리', 'XSN', 3, 2),
                    Token('의', 'NNG', 5, 1),
                ],
                -3.0,
            ),
        ],
    }

    candidate = KoreanAnalyzer(dictionary, ContextKiwi(analyses)).analyze(
        '오늘 마피아끼리의', (3, 9)
    )[0]

    assert candidate.lemma == '마피아끼리'


def test_isolated_analysis_recovers_defined_components_for_undefined_leader() -> None:
    analyses = {
        '오늘 갔다오다': [
            (
                [Token('갔다오', 'VV', 3, 3), Token('다', 'EF', 6, 1)],
                -1.0,
            )
        ],
        '갔다오다': [
            ([Token('갔다오', 'VV', 0, 3), Token('다', 'EF', 3, 1)], -1.0),
            (
                [
                    Token('가', 'VV', 0, 1),
                    Token('었', 'EP', 0, 1),
                    Token('다', 'EC', 1, 1),
                    Token('오', 'VV', 2, 1),
                    Token('다', 'EF', 3, 1),
                ],
                -3.0,
            ),
        ],
    }

    candidate = KoreanAnalyzer(RoleDictionary(), ContextKiwi(analyses)).analyze(
        '오늘 갔다오다', (3, 7)
    )[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '가다',
        '오다',
    ]


def test_isolated_analysis_does_not_replace_defined_contextual_leader() -> None:
    dictionary = RoleDictionary()
    dictionary.values = {
        **dictionary.values,
        '갔다오다': (_entry('round-trip', '갔다오다', 'verb', 'to go and return'),),
    }
    analyses = {
        '오늘 갔다오다': [
            (
                [Token('갔다오', 'VV', 3, 3), Token('다', 'EF', 6, 1)],
                -1.0,
            )
        ],
    }
    kiwi = ContextKiwi(analyses)

    candidate = KoreanAnalyzer(dictionary, kiwi).analyze('오늘 갔다오다', (3, 7))[0]

    assert [component.lemma for component in candidate.lexical_components] == ['갔다오다']
    assert list(kiwi.analyses) == ['오늘 갔다오다']
