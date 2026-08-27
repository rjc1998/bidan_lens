from dataclasses import dataclass

import pytest

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import (
    AnalysisCandidate,
    DictionaryEntry,
    DictionarySense,
    LearnerFeature,
    LexicalComponent,
    MorphemeExplanation,
)


@dataclass
class Token:
    form: str
    tag: str
    start: int
    len: int


class FakeKiwi:
    def __init__(self, analyses):
        self.analyses = analyses

    def analyze(self, _text: str, top_n: int = 1):
        return self.analyses[:top_n]

    def space(self, text: str, reset_whitespace: bool = False) -> str:
        return {"먹고싶어요": "먹고 싶어요"}.get(text, text)


class RecordingKiwi(FakeKiwi):
    requested_top_n: int | None = None

    def analyze(self, _text: str, top_n: int = 1):
        self.requested_top_n = top_n
        return self.analyses[:top_n]


class FakeDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        known = {
            "먹다": "to eat",
            "싶다": "to want",
            "공부하다": "to study",
            "어디": "where",
        }
        if lemma not in known:
            return ()
        return (
            DictionaryEntry(
                lemma,
                lemma,
                part_of_speech,
                None,
                "beginner",
                (DictionarySense(known[lemma]),),
            ),
        )


def analyze(tokens, sentence, span):
    kiwi = FakeKiwi([(tokens, -1.0)])
    return KoreanAnalyzer(FakeDictionary(), kiwi).analyze(sentence, span)[0]


def test_past_form_recovers_lemma_and_beginner_features() -> None:
    candidate = analyze(
        [Token("먹", "VV", 0, 1), Token("었", "EP", 1, 1), Token("습니다", "EF", 2, 3)],
        "먹었습니다",
        (0, 5),
    )
    assert candidate.lemma == "먹다"
    assert [feature.label for feature in candidate.features] == [
        "past tense",
        "formal polite style",
    ]
    assert candidate.dictionary_entries[0].senses[0].definition == "to eat"
    assert all(part.learner_label != "VV" for part in candidate.morphemes)


def test_honorific_polite_form_is_explained() -> None:
    candidate = analyze(
        [
            Token("먹", "VV", 0, 1),
            Token("으시", "EP", 1, 2),
            Token("었", "EP", 3, 1),
            Token("어요", "EF", 4, 2),
        ],
        "먹으시었어요",
        (0, 6),
    )
    labels = {feature.label for feature in candidate.features}
    assert {"honorific", "past tense", "polite style"} <= labels


def test_noun_plus_hada_recovers_compound_dictionary_form() -> None:
    candidate = analyze(
        [Token("공부", "NNG", 0, 2), Token("하", "XSV", 2, 1), Token("어요", "EF", 3, 2)],
        "공부해요",
        (0, 5),
    )
    assert candidate.lemma == "공부하다"


@pytest.mark.parametrize(
    ('verb_suffix', 'expected_lemma'),
    [('하', '국유화하다'), ('되', '국유화되다')],
)
def test_dictionary_backed_hwa_derivation_forms_one_action_verb(
    verb_suffix: str,
    expected_lemma: str,
) -> None:
    class DerivedVerbDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma != expected_lemma or part_of_speech not in {None, 'verb'}:
                return ()
            return (
                DictionaryEntry(
                    expected_lemma,
                    expected_lemma,
                    'verb',
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('국유', 'NNG', 0, 2),
        Token('화', 'XSN', 2, 1),
        Token(verb_suffix, 'XSV', 3, 1),
        Token('었', 'EP', 4, 1),
        Token('다', 'EF', 5, 1),
    ]
    candidate = KoreanAnalyzer(
        DerivedVerbDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('국유화' + verb_suffix + '었다', (0, 6))[0]

    assert candidate.lemma == expected_lemma
    assert candidate.lexical_components[0].surface == '국유화' + verb_suffix
    assert candidate.lexical_components[0].learner_role == 'action verb'
    assert candidate.lexical_components[0].dictionary_entries


def test_hwa_derivation_requires_an_exact_dictionary_entry() -> None:
    tokens = [
        Token('국유', 'NNG', 0, 2),
        Token('화', 'XSN', 2, 1),
        Token('되', 'XSV', 3, 1),
        Token('다', 'EF', 4, 1),
    ]
    candidate = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('국유화되다', (0, 5))[0]

    assert candidate.lemma == '국유'
    assert [component.lemma for component in candidate.lexical_components] == [
        '국유',
        '되다',
    ]


@pytest.mark.parametrize(
    ('stem', 'suffix_tag', 'part_of_speech', 'expected_role'),
    [
        ('가득', 'XSA', 'adjective', 'descriptive verb'),
        ('못', 'XSV', 'verb', 'action verb'),
    ],
)
def test_dictionary_backed_adverb_hada_derivation_forms_one_component(
    stem: str,
    suffix_tag: str,
    part_of_speech: str,
    expected_role: str,
) -> None:
    lemma = stem + '하다'

    class DerivedAdverbDictionary(DictionaryStore):
        def lookup(self, value: str, position=None, limit: int = 10):
            if value != lemma or position not in {None, part_of_speech}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    part_of_speech,
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token(stem, 'MAG', 0, len(stem)),
        Token('하', suffix_tag, len(stem), 1),
        Token('다', 'EF', len(stem) + 1, 1),
    ]
    candidate = KoreanAnalyzer(
        DerivedAdverbDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze(stem + '하다', (0, len(stem) + 2))[0]

    assert candidate.lemma == lemma
    assert len(candidate.lexical_components) == 1
    assert candidate.lexical_components[0].learner_role == expected_role


def test_dictionary_backed_adverb_hada_preserves_contextual_auxiliary_role() -> None:
    class AuxiliaryDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma != '못하다' or part_of_speech not in {'verb', '보조 동사'}:
                return ()
            return (
                DictionaryEntry(
                    part_of_speech,
                    lemma,
                    part_of_speech,
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('지', 'EC', 0, 1),
        Token('못', 'MAG', 2, 1),
        Token('하', 'XSV', 3, 1),
        Token('었', 'EP', 3, 1),
        Token('다', 'EF', 4, 1),
    ]
    candidate = KoreanAnalyzer(
        AuxiliaryDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('지 못했다', (2, 5))[0]

    assert candidate.lemma == '못하다'
    assert candidate.lexical_components[0].learner_role == 'helping verb'
    assert candidate.dictionary_entries[0].part_of_speech == '보조 동사'


def test_adverb_hada_derivation_requires_an_exact_dictionary_entry() -> None:
    tokens = [
        Token('가득', 'MAG', 0, 2),
        Token('하', 'XSA', 2, 1),
        Token('다', 'EF', 3, 1),
    ]
    candidate = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('가득하다', (0, 4))[0]

    assert candidate.lemma == '가득'
    assert [component.lemma for component in candidate.lexical_components] == [
        '가득',
        '하다',
    ]


def test_isolated_decomposition_does_not_displace_complete_adverb_hada() -> None:
    sentence = '지 못했다'
    surface = '못했다'

    class DerivationDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            known = {
                ('못', 'adverb'): '못',
                ('하다', '보조 동사'): '하다',
                ('못하다', 'verb'): '못하다',
                ('못하다', '보조 동사'): '못하다',
            }
            headword = known.get((lemma, part_of_speech))
            if headword is None:
                return ()
            return (
                DictionaryEntry(
                    lemma + str(part_of_speech),
                    headword,
                    part_of_speech,
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    full_context = [
        Token('지', 'EC', 0, 1),
        Token('못', 'MAG', 2, 1),
        Token('하', 'XSV', 3, 1),
        Token('었', 'EP', 3, 1),
        Token('다', 'EF', 4, 1),
    ]
    split_context = [
        Token('지', 'EC', 0, 1),
        Token('못', 'MAG', 2, 1),
        Token('하', 'VX', 3, 1),
        Token('었', 'EP', 3, 1),
        Token('다', 'EF', 4, 1),
    ]
    full_isolated = [
        Token('못', 'MAG', 0, 1),
        Token('하', 'XSV', 1, 1),
        Token('었', 'EP', 1, 1),
        Token('다', 'EF', 2, 1),
    ]
    split_isolated = [
        Token('못', 'MAG', 0, 1),
        Token('하', 'VX', 1, 1),
        Token('었', 'EP', 1, 1),
        Token('다', 'EF', 2, 1),
    ]

    class DerivationKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            analyses = {
                sentence: [(full_context, -1.0), (split_context, -4.0)],
                surface: [(split_isolated, -1.0), (full_isolated, -2.0)],
            }
            return analyses[text][:top_n]

    candidate = KoreanAnalyzer(
        DerivationDictionary(),
        DerivationKiwi([]),
    ).analyze(sentence, (2, 5))[0]

    assert candidate.lemma == '못하다'
    assert len(candidate.lexical_components) == 1
    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_zero_length_kiwi_insertion_is_not_a_lexical_component() -> None:
    class InsertionDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            known = {
                ('불과하다', 'adjective'),
                ('하다', 'verb'),
            }
            if (lemma, part_of_speech) not in known:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    part_of_speech,
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('불과', 'MAG', 0, 2),
        Token('하', 'XSA', 2, 1),
        Token('다고', 'EC', 3, 1),
        Token('하', 'VV', 4, 0),
        Token('는', 'ETM', 4, 1),
    ]
    candidate = KoreanAnalyzer(
        InsertionDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('불과하다는', (0, 5))[0]

    assert candidate.lemma == '불과하다'
    assert [component.lemma for component in candidate.lexical_components] == [
        '불과하다'
    ]


def test_nonduplicate_zero_length_kiwi_ellipsis_remains_a_component() -> None:
    tokens = [
        Token('살', 'VV', 0, 1),
        Token('ㄴ다고', 'EC', 0, 2),
        Token('하', 'VV', 2, 0),
        Token('고', 'EC', 2, 1),
    ]
    candidate = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('산다고', (0, 3))[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '살다',
        '하다',
    ]


def test_multi_eojeol_construction_remains_two_targets_with_shared_context() -> None:
    tokens = [
        Token("먹", "VV", 0, 1),
        Token("고", "EC", 1, 1),
        Token("싶", "VX", 3, 1),
        Token("어요", "EF", 4, 2),
    ]
    kiwi = FakeKiwi([(tokens, -1.0)])
    analyzer = KoreanAnalyzer(FakeDictionary(), kiwi)
    first = analyzer.analyze("먹고 싶어요", (0, 2))[0]
    second = analyzer.analyze("먹고 싶어요", (3, 6))[0]
    assert first.surface == "먹고" and first.lemma == "먹다"
    assert second.surface == "싶어요" and second.lemma == "싶다"


def test_only_provenance_backed_spacing_is_marked() -> None:
    analyzer = KoreanAnalyzer(FakeDictionary(), FakeKiwi([]))
    assert analyzer.conservative_correction("갔다오다") == "갔다 오다"
    assert analyzer.conservative_correction("먹고싶어요") is None


class PosFallbackDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma == "먹다" and part_of_speech is None:
            return (
                DictionaryEntry(
                    "42",
                    lemma,
                    "verb",
                    None,
                    "beginner",
                    (DictionarySense("to eat"),),
                ),
            )
        return ()


def test_dictionary_lookup_falls_back_when_kiwi_pos_is_ambiguous() -> None:
    kiwi = FakeKiwi([([Token("먹", "VA", 0, 1), Token("어요", "EF", 1, 2)], -1.0)])
    candidate = KoreanAnalyzer(PosFallbackDictionary(), kiwi).analyze("먹어요", (0, 3))[0]

    assert candidate.lemma == "먹다"
    assert candidate.dictionary_entries[0].senses[0].definition == "to eat"


def test_dictionary_backed_noun_segmentation_recovers_particle_feature() -> None:
    kiwi = FakeKiwi(
        [
            (
                [
                    Token("어디", "NP", 0, 2),
                    # An offset at the target boundary models a tokenizer alignment
                    # that excludes the attached particle from the selected tokens.
                    Token("에", "JKB", 3, 1),
                ],
                -1.0,
            )
        ]
    )

    candidate = KoreanAnalyzer(FakeDictionary(), kiwi).analyze("어디에", (0, 3))[0]

    assert candidate.lemma == "어디"
    assert {feature.label for feature in candidate.features} == {"particle"}
    assert all(part.learner_label != "particle" for part in candidate.morphemes)


def test_whole_dictionary_noun_ending_like_particle_is_not_split() -> None:
    class WholeNounDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma != "사과":
                return ()
            return (
                DictionaryEntry(
                    "apple",
                    lemma,
                    "noun",
                    None,
                    "beginner",
                    (DictionarySense("apple"),),
                ),
            )

    kiwi = FakeKiwi([([Token("사과", "NNG", 0, 2)], -1.0)])

    candidate = KoreanAnalyzer(WholeNounDictionary(), kiwi).analyze("사과", (0, 2))[0]

    assert {feature.label for feature in candidate.features} == set()
    assert candidate.lemma == "사과"


def test_krdict_particle_recovers_feature_after_multiple_noun_components() -> None:
    class CompoundParticleDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            known = {
                "학교": ("school", "noun"),
                "생활": ("life", "noun"),
                "부터": ("starting from", "particle"),
            }
            value = known.get(lemma)
            if value is None or part_of_speech not in {None, value[1]}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    value[1],
                    None,
                    "beginner",
                    (DictionarySense(value[0]),),
                ),
            )

    kiwi = FakeKiwi(
        [
            (
                [
                    Token("학교", "NNG", 0, 2),
                    Token("생활", "NNG", 2, 2),
                    Token("부터", "ETM", 4, 2),
                ],
                -1.0,
            )
        ]
    )

    candidate = KoreanAnalyzer(CompoundParticleDictionary(), kiwi).analyze("학교생활부터", (0, 6))[
        0
    ]

    assert [part.lemma for part in candidate.lexical_components] == ["학교", "생활"]
    assert "particle" in {feature.label for feature in candidate.features}


def test_internal_kiwi_search_is_deeper_than_popup_candidate_limit() -> None:
    kiwi = RecordingKiwi([([Token("먹", "VV", 0, 1), Token("어요", "EF", 1, 2)], -1.0)])

    candidates = KoreanAnalyzer(FakeDictionary(), kiwi).analyze("먹어요", (0, 3), max_candidates=5)

    assert kiwi.requested_top_n == 10
    assert len(candidates) <= 5


def test_particle_fallback_precedes_isolated_component_recovery() -> None:
    class SentenceKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            analyses = {
                "오늘 어디에": [([Token("미상", "NNG", 3, 2)], -1.0)],
                "어디에": [
                    (
                        [Token("어디", "NP", 0, 2), Token("에", "NNG", 2, 1)],
                        -1.0,
                    )
                ],
            }
            return analyses[text][:top_n]

    candidate = KoreanAnalyzer(FakeDictionary(), SentenceKiwi([])).analyze("오늘 어디에", (3, 6))[0]

    assert candidate.lemma == "어디"
    assert {feature.label for feature in candidate.features} == {"particle"}
    assert len(candidate.lexical_components) == 1


def test_terminal_derivational_noun_suffix_extends_component_surface() -> None:
    base = "\uac00\ub2a5"
    suffix = "\uc131"
    kiwi = FakeKiwi([([Token(base, "NNG", 0, 2), Token(suffix, "XSN", 2, 1)], -1.0)])

    candidate = KoreanAnalyzer(FakeDictionary(), kiwi).analyze(base + suffix, (0, 3))[0]

    assert candidate.lexical_components[0].surface == base + suffix
    assert candidate.lexical_components[0].lemma == base + suffix


def test_between_noun_suffix_is_not_assigned_to_either_component() -> None:
    first = "\uc5b8\uc5b4"
    suffix = "\uc801"
    second = "\ub300\uad00"
    kiwi = FakeKiwi(
        [
            (
                [
                    Token(first, "NNG", 0, 2),
                    Token(suffix, "XSN", 2, 1),
                    Token(second, "NNG", 3, 2),
                ],
                -1.0,
            )
        ]
    )

    candidate = KoreanAnalyzer(FakeDictionary(), kiwi).analyze(first + suffix + second, (0, 5))[0]

    assert [item.surface for item in candidate.lexical_components] == [first, second]


def test_dictionary_backed_internal_noun_suffix_extends_preceding_component() -> None:
    class InternalSuffixDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma not in {'민주화', '운동'} or part_of_speech not in {None, 'noun'}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    'noun',
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('민주', 'NNG', 0, 2),
        Token('화', 'XSN', 2, 1),
        Token('운동', 'NNG', 3, 2),
        Token('의', 'JKG', 5, 1),
    ]
    candidate = KoreanAnalyzer(
        InternalSuffixDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('민주화운동의', (0, 6))[0]

    assert candidate.lemma == '민주화'
    assert [item.surface for item in candidate.lexical_components] == [
        '민주화',
        '운동',
    ]


def test_dictionary_backed_internal_non_hwa_suffix_remains_unattached() -> None:
    class InternalSuffixDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma not in {'언어적', '대관'} or part_of_speech not in {None, 'noun'}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    'noun',
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('언어', 'NNG', 0, 2),
        Token('적', 'XSN', 2, 1),
        Token('대관', 'NNG', 3, 2),
    ]
    candidate = KoreanAnalyzer(
        InternalSuffixDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('언어적대관', (0, 5))[0]

    assert candidate.lemma == '언어'
    assert [item.surface for item in candidate.lexical_components] == [
        '언어',
        '대관',
    ]


def test_noun_suffix_before_copula_does_not_extend_component() -> None:
    noun = "\uc0dd\uc0b0"
    suffix = "\uc801"
    kiwi = FakeKiwi(
        [
            (
                [
                    Token(noun, "NNG", 0, 2),
                    Token(suffix, "XSN", 2, 1),
                    Token("\uc774", "VCP", 3, 1),
                ],
                -1.0,
            )
        ]
    )

    candidate = KoreanAnalyzer(FakeDictionary(), kiwi).analyze(noun + suffix + "\uc778", (0, 4))[0]

    assert candidate.lexical_components[0].surface == noun


def test_dictionary_backed_noun_suffix_before_copula_remains_unattached() -> None:
    class CopularNounDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma != '현실적' or part_of_speech not in {None, 'noun'}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    'noun',
                    None,
                    None,
                    (DictionarySense('definition'),),
                ),
            )

    tokens = [
        Token('현실', 'NNG', 0, 2),
        Token('적', 'XSN', 2, 1),
        Token('이', 'VCP', 3, 1),
        Token('ㄴ', 'ETM', 3, 1),
    ]
    candidate = KoreanAnalyzer(
        CopularNounDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    ).analyze('현실적인', (0, 4))[0]

    assert candidate.lemma == '현실'
    assert candidate.lexical_components[0].surface == '현실'


def test_dictionary_backed_noun_prefix_recovers_complete_component() -> None:
    prefix = "\uc7ac"
    noun = "\uc0dd\uc0b0"
    combined = prefix + noun

    class PrefixDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma != combined or part_of_speech not in {None, "noun"}:
                return ()
            return (
                DictionaryEntry(
                    combined,
                    combined,
                    "noun",
                    None,
                    None,
                    (DictionarySense("reproduction"),),
                ),
            )

    kiwi = FakeKiwi([([Token(prefix, "XPN", 0, 1), Token(noun, "NNG", 1, 2)], -1.0)])
    candidate = KoreanAnalyzer(PrefixDictionary(), kiwi).analyze(combined, (0, 3))[0]

    assert candidate.lemma == combined
    assert candidate.lexical_components[0].surface == combined
    assert candidate.lexical_components[0].lemma == combined


def test_dictionary_backed_prefix_survives_noun_adjective_merge() -> None:
    prefix = "\ubb34"
    noun = "\uc758\ubbf8"
    suffix = "\ud558"
    lemma = prefix + noun + suffix + "\ub2e4"

    class PrefixedAdjectiveDictionary(DictionaryStore):
        def lookup(self, value: str, part_of_speech=None, limit: int = 10):
            if value != lemma or part_of_speech not in {None, "adjective"}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    "adjective",
                    None,
                    None,
                    (DictionarySense("meaningless"),),
                ),
            )

    kiwi = FakeKiwi(
        [
            (
                [
                    Token(prefix, "XPN", 0, 1),
                    Token(noun, "NNG", 1, 2),
                    Token(suffix, "XSA", 3, 1),
                ],
                -1.0,
            )
        ]
    )
    candidate = KoreanAnalyzer(PrefixedAdjectiveDictionary(), kiwi).analyze(
        prefix + noun + suffix, (0, 4)
    )[0]

    assert candidate.lemma == lemma
    assert candidate.lexical_components[0].surface == prefix + noun + suffix
    assert candidate.lexical_components[0].learner_role == "descriptive verb"


@pytest.mark.parametrize(
    ("tag", "role", "part_of_speech"),
    [
        ("MAG", "adverb", "adverb"),
        ("MAJ", "adverb", "adverb"),
        ("MM", "determiner", "determiner"),
        ("VCN", "descriptive verb", "adjective"),
    ],
)
def test_non_noun_lexical_tags_produce_dictionary_components(
    tag: str,
    role: str,
    part_of_speech: str,
) -> None:
    surface = "\uc544\ub2c8" if tag == "VCN" else "\ub2e4\uc2dc"
    lemma = surface + "\ub2e4" if tag == "VCN" else surface

    class RoleDictionary(DictionaryStore):
        def lookup(self, value: str, position=None, limit: int = 10):
            if value != lemma or position not in {None, part_of_speech}:
                return ()
            return (
                DictionaryEntry(
                    lemma,
                    lemma,
                    part_of_speech,
                    None,
                    None,
                    (DictionarySense("definition"),),
                ),
            )

    tokens = [Token(surface, tag, 0, len(surface))]
    if tag == "VCN":
        tokens.append(Token("\ub2e4", "EF", len(surface), 1))
    candidate = KoreanAnalyzer(RoleDictionary(), FakeKiwi([(tokens, -1.0)])).analyze(
        surface + ("\ub2e4" if tag == "VCN" else ""), (0, len(lemma))
    )[0]

    assert candidate.lemma == lemma
    assert candidate.lexical_components[0].learner_role == role
    assert candidate.lexical_components[0].dictionary_entries


class InflectedCandidateDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma not in {'\ud558\ub2e4', '\uac00\ub2a5\ud558\ub2e4'}:
            return ()
        return (
            DictionaryEntry(
                lemma,
                lemma,
                part_of_speech,
                None,
                None,
                (DictionarySense('definition'),),
            ),
        )


def test_dictionary_backed_bound_root_and_adjective_suffix_form_one_component() -> None:
    analyses = [
        (
            [
                Token('\uac00\ub2a5', 'XR', 0, 2),
                Token('\ud558', 'XSA', 2, 1),
                Token('\uc5b4', 'EF', 2, 1),
            ],
            -1.0,
        )
    ]

    candidate = KoreanAnalyzer(
        InflectedCandidateDictionary(),
        FakeKiwi(analyses),
    ).analyze('\uac00\ub2a5\ud574', (0, 3))[0]

    assert candidate.lemma == '\uac00\ub2a5\ud558\ub2e4'
    assert candidate.lexical_components[0].surface == '\uac00\ub2a5\ud558'
    assert candidate.lexical_components[0].learner_role == 'descriptive verb'


@pytest.mark.parametrize(
    ('alternative_score', 'expected_lemma'),
    [(-2.5, '\uac00\ub2a5\ud558\ub2e4'), (-3.0, '\ud558\ub2e4')],
)
def test_complete_inflected_candidate_promotion_is_score_bounded(
    alternative_score: float,
    expected_lemma: str,
) -> None:
    analyses = [
        (
            [
                Token('\uac00\ub2a5', 'XPN', 0, 2),
                Token('\ud558', 'XSA', 2, 1),
                Token('\uc5b4', 'EF', 2, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('\uac00\ub2a5\ud558', 'VA', 0, 3),
                Token('\uc5b4', 'EF', 2, 1),
            ],
            alternative_score,
        ),
    ]

    candidate = KoreanAnalyzer(
        InflectedCandidateDictionary(),
        FakeKiwi(analyses),
    ).analyze('\uac00\ub2a5\ud574', (0, 3))[0]

    assert candidate.lemma == expected_lemma


class NominalRoleDictionary(DictionaryStore):
    def __init__(self, preferred_role: str) -> None:
        self.preferred_role = preferred_role

    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma != '그':
            return ()
        entries = (
            DictionaryEntry(
                self.preferred_role,
                lemma,
                self.preferred_role,
                None,
                None,
                (DictionarySense('preferred'),),
            ),
            DictionaryEntry(
                'other',
                lemma,
                'determiner' if self.preferred_role == 'pronoun' else 'pronoun',
                None,
                None,
                (DictionarySense('other'),),
            ),
        )
        if part_of_speech is not None:
            entries = tuple(
                entry for entry in entries if entry.part_of_speech == part_of_speech
            )
        return entries[:limit]


@pytest.mark.parametrize(
    ('preferred_role', 'expected_role'),
    [('determiner', 'determiner'), ('pronoun', 'pronoun')],
)
def test_dictionary_preferred_nominal_role_is_score_bounded(
    preferred_role: str,
    expected_role: str,
) -> None:
    analyses = [
        ([Token('그', 'NP', 0, 1), Token('돈', 'NNG', 2, 1)], -1.0),
        ([Token('그', 'MM', 0, 1), Token('돈', 'NNG', 2, 1)], -1.4),
    ]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary(preferred_role),
        FakeKiwi(analyses),
    ).analyze('그 돈', (0, 1))[0]

    assert candidate.lexical_components[0].learner_role == expected_role


@pytest.mark.parametrize(
    ('alternative_score', 'expected_role'),
    [(-4.1, 'noun'), (-4.2, 'name or proper noun')],
)
def test_dictionary_noun_entry_can_disambiguate_a_proper_noun_tag(
    alternative_score: float,
    expected_role: str,
) -> None:
    analyses = [
        ([Token('그', 'NNP', 0, 1)], -1.0),
        ([Token('그', 'NNG', 0, 1)], alternative_score),
    ]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary('noun'),
        FakeKiwi(analyses),
    ).analyze('그', (0, 1))[0]

    assert candidate.lexical_components[0].learner_role == expected_role


def test_dictionary_nominal_role_is_reapplied_after_wrapper_context() -> None:
    sentence = "'그' 돈"
    wrapped = [
        (
            [
                Token("'", 'SS', 0, 1),
                Token('그', 'NNP', 1, 1),
                Token("'", 'SS', 2, 1),
                Token('돈', 'NNG', 4, 1),
            ],
            -1.0,
        ),
        (
            [
                Token("'", 'SS', 0, 1),
                Token('그', 'NNG', 1, 1),
                Token("'", 'SS', 2, 1),
                Token('돈', 'NNG', 4, 1),
            ],
            -1.5,
        ),
    ]
    unwrapped = [
        ([Token('그', 'NNP', 0, 1), Token('돈', 'NNG', 2, 1)], -1.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = unwrapped if text == '그 돈' else wrapped
            return values[:top_n]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary('noun'),
        ContextKiwi([]),
    ).analyze(sentence, (1, 2))[0]

    assert candidate.lexical_components[0].learner_role == 'noun'


class PrenominalRoleDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        roles = {
            '\uba87': ('numeral', 'determiner'),
        }.get(lemma, ())
        if part_of_speech is not None:
            roles = tuple(role for role in roles if role == part_of_speech)
        return tuple(
            DictionaryEntry(
                lemma + role,
                lemma,
                role,
                None,
                None,
                (DictionarySense('definition'),),
            )
            for role in roles[:limit]
        )


@pytest.mark.parametrize(
    ('alternative_score', 'expected_role'),
    [(-4.9, 'determiner'), (-5.0, 'number')],
)
def test_prenominal_determiner_promotion_is_score_bounded(
    alternative_score: float,
    expected_role: str,
) -> None:
    analyses = [
        (
            [
                Token('\uba87', 'NR', 0, 1),
                Token('?', 'SF', 1, 1),
                Token('\ub144', 'NNB', 3, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('\uba87', 'MM', 0, 1),
                Token('?', 'SF', 1, 1),
                Token('\ub144', 'NNB', 3, 1),
            ],
            alternative_score,
        ),
    ]

    candidate = KoreanAnalyzer(
        PrenominalRoleDictionary(),
        FakeKiwi(analyses),
    )._analyze_candidates('\uba87? \ub144', (0, 1), 5)[0]

    assert candidate.lexical_components[0].learner_role == expected_role


class AdverbNounRoleDictionary(DictionaryStore):
    def __init__(self, preferred_role: str) -> None:
        self.preferred_role = preferred_role

    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma != '깊이':
            return ()
        other_role = 'noun' if self.preferred_role == 'adverb' else 'adverb'
        entries = (
            DictionaryEntry(
                self.preferred_role,
                lemma,
                self.preferred_role,
                None,
                None,
                (DictionarySense('preferred'),),
            ),
            DictionaryEntry(
                other_role,
                lemma,
                other_role,
                None,
                None,
                (DictionarySense('other'),),
            ),
        )
        if part_of_speech is not None:
            entries = tuple(
                entry for entry in entries if entry.part_of_speech == part_of_speech
            )
        return entries[:limit]


@pytest.mark.parametrize(
    (
        'preferred_role',
        'first_tag',
        'alternative_tag',
        'alternative_score',
        'expected_role',
    ),
    [
        ('noun', 'MAG', 'NNG', -3.3, 'adverb'),
        ('adverb', 'NNG', 'MAG', -3.3, 'adverb'),
        ('adverb', 'NNG', 'MAG', -3.5, 'noun'),
    ],
)
def test_dictionary_preferred_adverb_noun_role_is_score_bounded(
    preferred_role: str,
    first_tag: str,
    alternative_tag: str,
    alternative_score: float,
    expected_role: str,
) -> None:
    analyses = [
        ([Token('깊이', first_tag, 0, 2)], -1.0),
        ([Token('깊이', alternative_tag, 0, 2)], alternative_score),
    ]

    candidate = KoreanAnalyzer(
        AdverbNounRoleDictionary(preferred_role),
        FakeKiwi(analyses),
    ).analyze('깊이', (0, 2))[0]

    assert candidate.lexical_components[0].learner_role == expected_role


class VerbRoleDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        roles = {
            '만들다': ('verb',),
            '만': ('noun',),
            '들다': ('보조 동사', 'verb'),
            '보다': ('보조 동사', 'verb'),
            '되다': ('verb',),
        }.get(lemma, ())
        if not roles:
            return ()
        if part_of_speech is not None and part_of_speech not in roles:
            return ()
        role = part_of_speech or roles[0]
        return (
            DictionaryEntry(
                lemma + role,
                lemma,
                role,
                None,
                None,
                (DictionarySense('definition'),),
            ),
        )


def test_vx_without_an_auxiliary_entry_uses_lexical_dictionary_role() -> None:
    sentence = '제도를 만든다'
    contextual = [
        (
            [
                Token('제도', 'NNG', 0, 2),
                Token('를', 'JKO', 2, 1),
                Token('만들', 'VX', 4, 2),
                Token('ㄴ다', 'EF', 6, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('제도', 'NNG', 0, 2),
                Token('를', 'JKO', 2, 1),
                Token('만들', 'VV', 4, 2),
                Token('ㄴ다', 'EF', 6, 1),
            ],
            -7.4,
        ),
        (
            [
                Token('제도', 'NNG', 0, 2),
                Token('를', 'JKO', 2, 1),
                Token('만', 'NR', 4, 1),
                Token('들', 'VX', 5, 1),
                Token('ㄴ다', 'EF', 6, 1),
            ],
            -8.0,
        ),
    ]
    isolated = [
        ([Token('만들', 'VX', 0, 2), Token('ㄴ다', 'EF', 2, 1)], -1.0),
        ([Token('만들', 'VV', 0, 2), Token('ㄴ다', 'EF', 2, 1)], -7.4),
        (
            [
                Token('만', 'NR', 0, 1),
                Token('들', 'VX', 1, 1),
                Token('ㄴ다', 'EF', 2, 1),
            ],
            -8.0,
        ),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = contextual if text == sentence else isolated
            return values[:top_n]

    candidate = KoreanAnalyzer(
        VerbRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (4, 7))[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'


def test_unsupported_auxiliary_promotion_is_score_bounded() -> None:
    analyses = [
        ([Token('만들', 'VX', 0, 2), Token('ㄴ다', 'EF', 2, 1)], -1.0),
        ([Token('만들', 'VV', 0, 2), Token('ㄴ다', 'EF', 2, 1)], -8.1),
    ]

    candidate = KoreanAnalyzer(
        VerbRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('만든다', (0, 3))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_contextual_helping_verb_is_not_demoted() -> None:
    sentence = '먹어 본다'
    analyses = [
        (
            [Token('먹어', 'EC', 0, 2), Token('보', 'VX', 3, 1), Token('ㄴ다', 'EF', 4, 1)],
            -1.0,
        ),
        (
            [Token('먹어', 'EC', 0, 2), Token('보', 'VV', 3, 1), Token('ㄴ다', 'EF', 4, 1)],
            -1.5,
        ),
    ]

    candidate = KoreanAnalyzer(
        VerbRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze(sentence, (3, 5))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_ge_doeda_auxiliary_cue_ignores_wrapper_punctuation() -> None:
    sentence = '게 /된다/'
    analyses = [
        (
            [Token('게', 'NNG', 0, 1), Token('되', 'VV', 3, 1), Token('ㄴ다', 'EF', 4, 1)],
            -1.0,
        ),
        (
            [Token('게', 'EC', 0, 1), Token('되', 'VX', 3, 1), Token('ㄴ다', 'EF', 4, 1)],
            -3.5,
        ),
    ]

    candidate = KoreanAnalyzer(
        VerbRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze(sentence, (3, 5))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_dictionary_preference_does_not_demote_a_determiner() -> None:
    analyses = [
        ([Token('그', 'MM', 0, 1), Token('돈', 'NNG', 2, 1)], -1.0),
        ([Token('그', 'NNG', 0, 1), Token('돈', 'NNG', 2, 1)], -1.4),
    ]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary('noun'),
        FakeKiwi(analyses),
    ).analyze('그 돈', (0, 1))[0]

    assert candidate.lexical_components[0].learner_role == 'determiner'


class ReviewedRoleDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        roles = {
            '적': ('의존 명사', 'noun'),
            '부정하다': ('verb', 'adjective'),
            '있다': ('보조 동사', 'verb', 'adjective'),
            '하다': ('보조 동사', 'verb'),
            '가다': ('verb', '보조 동사'),
            '듯하다': ('보조 형용사',),
            '회전': ('noun',),
            '식': ('noun', '의존 명사'),
            '고려': ('noun',),
            '말': ('noun', '의존 명사'),
            '이루다': ('verb',),
            '지다': ('보조 동사',),
            '이루어지다': ('verb',),
            '알': ('noun', '의존 명사'),
            '알다': ('verb',),
            '대한': ('noun',),
            '대하다': ('verb',),
            '한': ('determiner',),
            '층': ('noun',),
            '한층': ('adverb',),
            '바라다': ('verb',),
            '보다': ('보조 동사',),
            '바라보다': ('verb',),
            '거듭': ('adverb',),
            '거듭되다': ('verb',),
            '해': ('noun',),
            '그런': ('determiner',),
            '그러다': ('verb',),
            '바뀌다': ('verb',),
        }.get(lemma, ())
        if part_of_speech is not None:
            roles = tuple(role for role in roles if role == part_of_speech)
        return tuple(
            DictionaryEntry(
                lemma + role,
                lemma,
                role,
                None,
                None,
                (DictionarySense('definition'),),
            )
            for role in roles[:limit]
        )


@pytest.mark.parametrize(
    ('sentence', 'target_span', 'tokens'),
    [
        (
            '\ub9e1\uc544 \ud558\ub294',
            (3, 5),
            [
                Token('\ub9e1', 'VV', 0, 1),
                Token('\uc5b4', 'EC', 1, 1),
                Token('\ud558', 'VV', 3, 1),
                Token('\ub294', 'ETM', 4, 1),
            ],
        ),
        (
            '\uadf8\ub807\uac8c \ud558\uc9c0',
            (4, 6),
            [
                Token('\uadf8\ub807', 'VA', 0, 2),
                Token('\uac8c', 'EC', 2, 1),
                Token('\ud558', 'VV', 4, 1),
                Token('\uc9c0', 'EC', 5, 1),
            ],
        ),
    ],
)
def test_lexical_hada_is_not_promoted_by_plain_connective_context(
    sentence: str,
    target_span: tuple[int, int],
    tokens: list[Token],
) -> None:
    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    )._analyze_candidates(sentence, target_span, 5)[0]

    assert candidate.lexical_components[0].learner_role == 'action verb'


def test_obligative_connective_still_promotes_hada_to_auxiliary() -> None:
    tokens = [
        Token('\uc5b4\uc57c', 'EC', 0, 2),
        Token('\ud558', 'VV', 3, 1),
        Token('\u11af', 'ETM', 3, 1),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi([(tokens, -1.0)]),
    )._analyze_candidates('\uc5b4\uc57c \ud560', (3, 4), 5)[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


@pytest.mark.parametrize(
    ('alternative_score', 'expected_role'),
    [(-3.6, 'dependent noun'), (-3.7, 'noun')],
)
def test_dictionary_preferred_dependent_noun_is_score_bounded(
    alternative_score: float,
    expected_role: str,
) -> None:
    analyses = [
        ([Token('적', 'NNG', 0, 1), Token('은', 'JX', 1, 1)], -1.0),
        ([Token('적', 'NNB', 0, 1), Token('은', 'JX', 1, 1)], alternative_score),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('적은', (0, 2))[0]

    assert candidate.lexical_components[0].learner_role == expected_role


def test_dictionary_preference_does_not_demote_a_dependent_noun() -> None:
    class NounFirstDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            roles = ('noun', '의존 명사')
            if part_of_speech is not None:
                roles = tuple(role for role in roles if role == part_of_speech)
            return tuple(
                DictionaryEntry(
                    lemma + role,
                    lemma,
                    role,
                    None,
                    None,
                    (DictionarySense('definition'),),
                )
                for role in roles[:limit]
            )

    analyses = [
        ([Token('수', 'NNB', 0, 1), Token('가', 'JKS', 1, 1)], -1.0),
        ([Token('수', 'NNG', 0, 1), Token('가', 'JKS', 1, 1)], -1.2),
    ]

    candidate = KoreanAnalyzer(
        NounFirstDictionary(),
        FakeKiwi(analyses),
    ).analyze('수가', (0, 2))[0]

    assert candidate.lexical_components[0].learner_role == 'dependent noun'


def test_dictionary_rejects_an_unsupported_dependent_noun_role() -> None:
    analyses = [
        ([Token('회전', 'NNB', 0, 2), Token('도', 'JX', 2, 1)], -1.0),
        ([Token('회전', 'NNG', 0, 2), Token('도', 'JX', 2, 1)], -4.0),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('회전도', (0, 3))[0]

    assert candidate.lexical_components[0].learner_role == 'noun'


@pytest.mark.parametrize(
    ('surface', 'tokens'),
    [
        (
            '식이다',
            (
                Token('식', 'NNB', 0, 1),
                Token('이', 'VCP', 1, 1),
                Token('다', 'EF', 2, 1),
            ),
        ),
        ('식으로', (Token('식', 'NNB', 0, 1), Token('으로', 'JKB', 1, 2))),
    ],
)
def test_contextual_sik_promotes_an_ordinary_noun(
    surface: str,
    tokens: tuple[Token, ...],
) -> None:
    alternative = tuple(
        Token(token.form, 'NNG' if index == 0 else token.tag, token.start, token.len)
        for index, token in enumerate(tokens)
    )
    analyses = [(list(tokens), -1.0), (list(alternative), -5.8)]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze(surface, (0, len(surface)))[0]

    assert candidate.lexical_components[0].learner_role == 'noun'


def test_attached_compound_promotes_a_terminal_ordinary_noun() -> None:
    analyses = [
        (
            [
                Token('고려', 'NNP', 0, 2),
                Token('말', 'NNB', 2, 1),
                Token('의', 'JKG', 3, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('고려', 'NNP', 0, 2),
                Token('말', 'NNG', 2, 1),
                Token('의', 'JKG', 3, 1),
            ],
            -6.7,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('고려말의', (0, 4))[0]

    assert [component.learner_role for component in candidate.lexical_components] == [
        'name or proper noun',
        'noun',
    ]


@pytest.mark.parametrize(
    ('alternative_score', 'expected_role'),
    [(-1.8, 'action verb'), (-2.0, 'descriptive verb')],
)
def test_dictionary_preferred_predicate_role_is_score_bounded(
    alternative_score: float,
    expected_role: str,
) -> None:
    analyses = [
        ([Token('부정하', 'VA', 0, 3), Token('ㄴ', 'ETM', 3, 1)], -1.0),
        ([Token('부정하', 'VV', 0, 3), Token('ㄴ', 'ETM', 3, 1)], alternative_score),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('부정한', (0, 4))[0]

    assert candidate.lexical_components[0].learner_role == expected_role


def test_locative_itda_prefers_descriptive_role() -> None:
    analyses = [
        (
            [
                Token('왕', 'NNG', 0, 1),
                Token('에게', 'JKB', 1, 2),
                Token('있', 'VV', 4, 1),
                Token('었', 'EP', 5, 1),
                Token('다', 'EF', 6, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('왕', 'NNG', 0, 1),
                Token('에게', 'JKB', 1, 2),
                Token('있', 'VA', 4, 1),
                Token('었', 'EP', 5, 1),
                Token('다', 'EF', 6, 1),
            ],
            -6.0,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('왕에게 있었다', (4, 7))[0]

    assert candidate.lexical_components[0].learner_role == 'descriptive verb'


def test_hayaman_particle_preserves_auxiliary_context() -> None:
    analyses = [
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('어야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VV', 6, 1),
                Token('ㄹ', 'ETM', 6, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('어야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VX', 6, 1),
                Token('ㄹ', 'ETM', 6, 1),
            ],
            -1.8,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('기울여야만 할', (6, 7))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_wrapper_context_can_replace_learner_identical_candidates() -> None:
    wrapped = '휩쓸어 -간다-'
    unwrapped = '휩쓸어 간다'
    contextual = [
        (
            [Token('휩쓸어', 'NNG', 0, 3), Token('가', 'VV', 5, 1), Token('ㄴ다', 'EF', 6, 1)],
            -1.0,
        ),
        (
            [Token('휩쓸어', 'NNG', 0, 3), Token('가', 'VV', 5, 1), Token('ㄴ다', 'EF', 6, 1)],
            -1.5,
        ),
    ]
    unwrapped_analysis = [
        (
            [
                Token('휩쓸', 'VV', 0, 2),
                Token('어', 'EC', 2, 1),
                Token('가', 'VX', 4, 1),
                Token('ㄴ다', 'EF', 5, 1),
            ],
            -1.0,
        )
    ]

    class WrapperKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            if text == unwrapped:
                values = unwrapped_analysis
            elif text == '간다':
                values = [
                    ([Token('가', 'VV', 0, 1), Token('ㄴ다', 'EF', 1, 1)], -1.0)
                ]
            else:
                values = contextual
            return values[:top_n]

    analyzer = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        WrapperKiwi([]),
    )
    initial = analyzer._analyze_candidates(wrapped, (5, 7), 5)
    promoted = analyzer._promote_close_wrapper_context_candidate(
        wrapped,
        (5, 7),
        initial,
        5,
    )

    assert promoted[0].lexical_components[0].learner_role == 'helping verb'

    candidate = analyzer.analyze(wrapped, (5, 7))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


class WrapperGrammarDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        roles = {
            '경우': ('noun',),
            '일': ('noun',),
            '의': ('noun', 'particle'),
        }.get(lemma, ())
        if part_of_speech is not None:
            roles = tuple(role for role in roles if role == part_of_speech)
        return tuple(
            DictionaryEntry(
                lemma + role,
                lemma,
                role,
                None,
                None,
                (DictionarySense('definition'),),
            )
            for role in roles[:limit]
        )


def test_wrapper_context_recovers_a_dictionary_backed_standalone_particle() -> None:
    wrapped = '대부 [의] 요체'
    unwrapped = '대부 의 요체'
    wrapped_analyses = [
        (
            [
                Token('대부', 'NNG', 0, 2),
                Token('[', 'SSO', 3, 1),
                Token('의', 'NNG', 4, 1),
                Token(']', 'SSC', 5, 1),
                Token('요체', 'NNG', 7, 2),
            ],
            -1.0,
        ),
        (
            [
                Token('대부', 'NNG', 0, 2),
                Token('[', 'SSO', 3, 1),
                Token('의', 'JKG', 4, 1),
                Token(']', 'SSC', 5, 1),
                Token('요체', 'NNG', 7, 2),
            ],
            -6.0,
        ),
    ]
    unwrapped_analyses = [
        (
            [
                Token('대부', 'NNG', 0, 2),
                Token('의', 'JKG', 3, 1),
                Token('요체', 'NNG', 5, 2),
            ],
            -1.0,
        )
    ]

    class WrapperKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            if text == unwrapped:
                values = unwrapped_analyses
            elif text == '의':
                values = [([Token('의', 'NNG', 0, 1)], -1.0)]
            else:
                values = wrapped_analyses
            return values[:top_n]

    candidate = KoreanAnalyzer(
        WrapperGrammarDictionary(),
        WrapperKiwi([]),
    ).analyze(wrapped, (4, 5))[0]

    assert candidate.lemma == '의'
    assert candidate.lexical_components[0].learner_role == 'particle'
    assert candidate.dictionary_entries[0].part_of_speech == 'particle'
    assert [entry.part_of_speech for entry in candidate.dictionary_entries] == [
        'particle',
        'noun',
    ]


def test_wrapper_context_recovers_a_copular_adnominal_before_a_dependent_noun() -> None:
    wrapped = '전자의 /경우일/ 터'
    unwrapped = '전자의 경우일 터'
    wrapped_analyses = [
        (
            [
                Token('전자', 'NNG', 0, 2),
                Token('의', 'JKG', 2, 1),
                Token('/', 'SP', 4, 1),
                Token('경우', 'NNG', 5, 2),
                Token('일', 'NNG', 7, 1),
                Token('/', 'SP', 8, 1),
                Token('터', 'NNB', 10, 1),
            ],
            -1.0,
        )
    ]
    unwrapped_analyses = [
        (
            [
                Token('전자', 'NNG', 0, 2),
                Token('의', 'JKG', 2, 1),
                Token('경우', 'NNG', 4, 2),
                Token('이', 'VCP', 6, 1),
                Token('ㄹ', 'ETM', 6, 1),
                Token('터', 'NNB', 8, 1),
            ],
            -1.0,
        )
    ]

    class WrapperKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            if text == unwrapped:
                values = unwrapped_analyses
            elif text == '경우일':
                values = [
                    (
                        [Token('경우', 'NNG', 0, 2), Token('일', 'NNG', 2, 1)],
                        -1.0,
                    )
                ]
            else:
                values = wrapped_analyses
            return values[:top_n]

    candidate = KoreanAnalyzer(
        WrapperGrammarDictionary(),
        WrapperKiwi([]),
    ).analyze(wrapped, (5, 8))[0]

    assert candidate.lemma == '경우'
    assert [component.lemma for component in candidate.lexical_components] == [
        '경우'
    ]
    assert 'verb ending' in {feature.label for feature in candidate.features}


def test_contracted_hayaman_context_promotes_auxiliary_hada() -> None:
    analyses = [
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('여야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VV', 7, 1),
                Token('ㄹ', 'ETM', 7, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('여야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VX', 7, 1),
                Token('ㄹ', 'ETM', 7, 1),
            ],
            -1.8,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('기울여야만 /할/', (7, 8))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_exclusively_auxiliary_dictionary_lemma_uses_helping_role() -> None:
    analyses = [
        (
            [
                Token('복권되', 'VV', 0, 3),
                Token('ㄴ', 'ETM', 3, 1),
                Token('듯하', 'VA', 5, 2),
                Token('었', 'EP', 7, 1),
                Token('다', 'EF', 8, 1),
            ],
            -1.0,
        )
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('복권된 듯했다', (5, 9))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


def test_isolated_role_does_not_override_complete_unwrapped_context() -> None:
    sentence = '아파트치고 커서 찾기도 쉽다'
    contextual = [
        ([Token('크', 'VA', 6, 1), Token('어서', 'EC', 7, 1)], -1.0),
        ([Token('크', 'VV', 6, 1), Token('어서', 'EC', 7, 1)], -2.5),
    ]
    isolated = [
        ([Token('크', 'VV', 0, 1), Token('어서', 'EC', 1, 1)], -1.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == '커서' else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (6, 8))[0]

    assert candidate.lexical_components[0].learner_role == 'descriptive verb'


def test_isolated_decomposition_still_applies_in_complete_unwrapped_context() -> None:
    sentence = '문제가 이루어진다'
    contextual = [
        ([Token('이루어지', 'VV', 4, 4), Token('ㄴ다', 'EF', 8, 1)], -1.0),
        (
            [
                Token('이루', 'VV', 4, 2),
                Token('어', 'EC', 6, 1),
                Token('지', 'VX', 7, 1),
                Token('ㄴ다', 'EF', 8, 1),
            ],
            -4.8,
        ),
    ]
    isolated = [
        (
            [
                Token('이루', 'VV', 0, 2),
                Token('어', 'EC', 2, 1),
                Token('지', 'VX', 3, 1),
                Token('ㄴ다', 'EF', 4, 1),
            ],
            -1.0,
        )
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == '이루어진다' else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (4, 9))[0]

    assert [component.lemma for component in candidate.lexical_components] == [
        '이루다',
        '지다',
    ]


def test_adnominal_before_dependent_noun_promotes_an_inflected_verb() -> None:
    analyses = [
        ([Token('알', 'NNG', 0, 1), Token('수', 'NNB', 2, 1)], -1.0),
        (
            [
                Token('알', 'VV', 0, 1),
                Token('ㄹ', 'ETM', 0, 1),
                Token('수', 'NNB', 2, 1),
            ],
            -1.8,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('알 수', (0, 1))[0]

    assert candidate.lemma == '알다'


def test_boundary_isolated_evidence_promotes_an_inflected_predicate() -> None:
    sentence = '대한 자료'
    contextual = [
        ([Token('대한', 'NNG', 0, 2), Token('자료', 'NNG', 3, 2)], -1.0),
        (
            [
                Token('대하', 'VV', 0, 2),
                Token('ㄴ', 'ETM', 1, 1),
                Token('자료', 'NNG', 3, 2),
            ],
            -6.8,
        ),
    ]
    isolated = [
        ([Token('대하', 'VV', 0, 2), Token('ㄴ', 'ETM', 1, 1)], -1.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == '대한' else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (0, 2))[0]

    assert candidate.lemma == '대하다'


def test_wrapper_context_synthesizes_a_repeated_nominal_as_a_predicate() -> None:
    sentence = '[대한]'
    wrapped = [
        ([Token('대한', 'NNG', 1, 2)], -1.0),
        ([Token('대한', 'NNP', 1, 2)], -2.0),
    ]
    unwrapped = [
        ([Token('대하', 'VV', 0, 2), Token('ㄴ', 'ETM', 1, 1)], -1.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = unwrapped if text == '대한' else wrapped
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (1, 3))[0]

    assert candidate.lemma == '대하다'


def test_complete_lexical_adverb_is_not_resplit() -> None:
    analyses = [
        ([Token('한', 'MM', 0, 1), Token('층', 'NNG', 1, 1)], -1.0),
        ([Token('한층', 'MAG', 0, 2)], -5.4),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('한층', (0, 2))[0]

    assert candidate.lemma == '한층'
    assert candidate.lexical_components[0].learner_role == 'adverb'


def test_isolated_auxiliary_decomposition_requires_a_connective() -> None:
    sentence = '바라보는 시선'
    contextual = [
        ([Token('바라보', 'VV', 0, 3), Token('는', 'ETM', 3, 1)], -1.0),
        (
            [
                Token('바라', 'VV', 0, 2),
                Token('보', 'VX', 2, 1),
                Token('는', 'ETM', 3, 1),
            ],
            -3.5,
        ),
    ]
    isolated = [
        ([Token('바라보', 'VV', 0, 3), Token('는', 'ETM', 3, 1)], -1.0),
        (
            [
                Token('바라', 'VV', 0, 2),
                Token('보', 'VX', 2, 1),
                Token('는', 'ETM', 3, 1),
            ],
            -4.0,
        ),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == '바라보는' else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (0, 4))[0]

    assert candidate.lemma == '바라보다'
    assert len(candidate.lexical_components) == 1


def test_isolated_auxiliary_decomposition_rejects_duplicate_components() -> None:
    hada_entries = (
        DictionaryEntry(
            '하다-verb',
            '하다',
            'verb',
            None,
            None,
            (DictionarySense('definition'),),
        ),
    )
    oda_entries = (
        DictionaryEntry(
            '오다-auxiliary',
            '오다',
            '보조 동사',
            None,
            None,
            (DictionarySense('definition'),),
        ),
    )
    action = LexicalComponent('하', '하다', 'action verb', hada_entries)
    auxiliary = LexicalComponent('오', '오다', 'helping verb', oda_entries)
    correct = AnalysisCandidate(
        '해왔다는',
        '하다',
        -1.0,
        lexical_components=(action, auxiliary),
        dictionary_entries=hada_entries,
    )
    duplicate = AnalysisCandidate(
        '해왔다는',
        '하다',
        -5.0,
        morphemes=(
            MorphemeExplanation('하', '하다', 'action verb'),
            MorphemeExplanation('어', '어', 'verb ending'),
            MorphemeExplanation('오', '오다', 'helping verb'),
            MorphemeExplanation('다는', '다는', 'verb ending'),
            MorphemeExplanation('하', '하다', 'action verb'),
        ),
        lexical_components=(action, auxiliary, action),
        dictionary_entries=hada_entries,
    )

    class IsolatedEvidenceAnalyzer(KoreanAnalyzer):
        def _analyze_candidates(self, sentence, target_span, max_candidates):
            del target_span, max_candidates
            return (duplicate,) if sentence == '해왔다는' else (correct, duplicate)

    analyzer = IsolatedEvidenceAnalyzer(FakeDictionary(), FakeKiwi([]))
    candidates = analyzer._promote_isolated_verb_role_candidate(
        '역할을 해왔다는 점',
        (4, 8),
        '해왔다는',
        (correct, duplicate),
        5,
    )

    assert candidates[0] is correct


@pytest.mark.parametrize(
    ('alternative_score', 'expected_component_count'),
    [(-4.2, 1), (-4.3, 2)],
)
def test_terminal_adverb_ending_promotion_is_score_bounded(
    alternative_score: float,
    expected_component_count: int,
) -> None:
    predicate_entries = (
        DictionaryEntry(
            '생각하다-verb',
            '생각하다',
            'verb',
            None,
            None,
            (DictionarySense('definition'),),
        ),
    )
    adverb_entries = (
        DictionaryEntry(
            '다-adverb',
            '다',
            'adverb',
            None,
            None,
            (DictionarySense('definition'),),
        ),
    )
    predicate = LexicalComponent(
        '생각하',
        '생각하다',
        'action verb',
        predicate_entries,
    )
    terminal_adverb = LexicalComponent('다', '다', 'adverb', adverb_entries)
    shared_morphemes = (
        MorphemeExplanation('생각', '생각', 'noun'),
        MorphemeExplanation('하', '하다', 'word part'),
        MorphemeExplanation('어서', '어서', 'verb ending'),
    )
    first = AnalysisCandidate(
        '생각해서다',
        '생각하다',
        -1.0,
        morphemes=(
            *shared_morphemes,
            MorphemeExplanation('다', '다', 'adverb'),
        ),
        lexical_components=(predicate, terminal_adverb),
        dictionary_entries=predicate_entries,
    )
    alternative = AnalysisCandidate(
        '생각해서다',
        '생각하다',
        alternative_score,
        morphemes=(
            *shared_morphemes,
            MorphemeExplanation('다', '다', 'verb ending'),
        ),
        lexical_components=(predicate,),
        dictionary_entries=predicate_entries,
    )

    candidates = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([]),
    )._promote_terminal_adverb_ending_candidate([first, alternative])

    assert len(candidates[0].lexical_components) == expected_component_count


def test_terminal_adverb_ending_promotion_survives_candidate_ranking() -> None:
    class EndingDictionary(DictionaryStore):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            roles = {
                '생각하다': ('verb',),
                '다': ('adverb',),
            }.get(lemma, ())
            if part_of_speech is not None:
                roles = tuple(role for role in roles if role == part_of_speech)
            return tuple(
                DictionaryEntry(
                    lemma + role,
                    lemma,
                    role,
                    None,
                    None,
                    (DictionarySense('definition'),),
                )
                for role in roles[:limit]
            )

    analyses = [
        (
            [
                Token('생각', 'NNG', 0, 2),
                Token('하', 'XSV', 2, 1),
                Token('어서', 'EC', 2, 2),
                Token('다', 'MAG', 4, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('생각', 'NNG', 0, 2),
                Token('하', 'XSV', 2, 1),
                Token('어서', 'EC', 2, 2),
                Token('하', 'VV', 4, 0),
                Token('다', 'EF', 4, 1),
            ],
            -3.8,
        ),
    ]

    candidate = KoreanAnalyzer(
        EndingDictionary(),
        FakeKiwi(analyses),
    ).analyze('생각해서다', (0, 5))[0]

    assert candidate.lemma == '생각하다'
    assert [component.lemma for component in candidate.lexical_components] == [
        '생각하다'
    ]
    assert candidate.morphemes[-1].learner_label == 'verb ending'


def test_complete_dictionary_predicate_replaces_a_word_part_derivation() -> None:
    analyses = [
        (
            [
                Token('거듭', 'MAG', 0, 2),
                Token('되', 'XSV', 2, 1),
                Token('었', 'EP', 3, 1),
                Token('다', 'EF', 4, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('거듭되', 'VV', 0, 3),
                Token('었', 'EP', 3, 1),
                Token('다', 'EF', 4, 1),
            ],
            -5.7,
        ),
    ]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi(analyses),
    ).analyze('거듭되었다', (0, 5))[0]

    assert candidate.lemma == '거듭되다'


@pytest.mark.parametrize(
    ('alternative_score', 'expected_lemma'),
    [(-4.19, '\ub9d0'), (-4.21, '\ub9d0\ub85c')],
)
def test_close_noun_particle_promotion_is_score_bounded(
    alternative_score: float,
    expected_lemma: str,
) -> None:
    proper_entries = (
        DictionaryEntry(
            'proper',
            '\ub9d0\ub85c',
            'proper noun',
            None,
            None,
            (DictionarySense('proper'),),
        ),
    )
    noun_entries = (
        DictionaryEntry(
            'noun',
            '\ub9d0',
            'noun',
            None,
            None,
            (DictionarySense('noun'),),
        ),
    )
    particle_entries = (
        DictionaryEntry(
            'particle',
            '\ub85c',
            'particle',
            None,
            None,
            (DictionarySense('particle'),),
        ),
    )

    class NounParticleDictionary(FakeDictionary):
        def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
            if lemma == '\ub85c' and part_of_speech in {None, 'particle'}:
                return particle_entries[:limit]
            return super().lookup(lemma, part_of_speech, limit)

    first = AnalysisCandidate(
        '\ub9d0\ub85c',
        '\ub9d0\ub85c',
        -1.0,
        lexical_components=(
            LexicalComponent(
                '\ub9d0\ub85c',
                '\ub9d0\ub85c',
                'name or proper noun',
                proper_entries,
            ),
        ),
        dictionary_entries=proper_entries,
    )
    alternative = AnalysisCandidate(
        '\ub9d0\ub85c',
        '\ub9d0',
        alternative_score,
        morphemes=(
            MorphemeExplanation('\ub9d0', '\ub9d0', 'noun'),
            MorphemeExplanation('\ub85c', '\ub85c', 'particle'),
        ),
        features=(LearnerFeature('particle', 'particle', '\ub85c'),),
        dictionary_entries=noun_entries,
        lexical_components=(
            LexicalComponent('\ub9d0', '\ub9d0', 'noun', noun_entries),
        ),
    )

    candidates = KoreanAnalyzer(
        NounParticleDictionary(),
        FakeKiwi([]),
    )._promote_close_noun_particle_candidate([first, alternative])

    assert candidates[0].lemma == expected_lemma


def test_complete_inflected_predicate_is_not_replaced_by_richer_split() -> None:
    whole_entries = (
        DictionaryEntry(
            'whole',
            '\ubc1b\uc544\ub4e4\uc774\ub2e4',
            'verb',
            None,
            None,
            (DictionarySense('whole'),),
        ),
    )
    part_entries = (
        DictionaryEntry(
            'part',
            '\ubc1b\ub2e4',
            'verb',
            None,
            None,
            (DictionarySense('part'),),
        ),
    )
    first = AnalysisCandidate(
        '\ubc1b\uc544\ub4e4\uc778\ub2e4',
        '\ubc1b\uc544\ub4e4\uc774\ub2e4',
        -1.0,
        morphemes=(
            MorphemeExplanation(
                '\ubc1b\uc544\ub4e4\uc774',
                '\ubc1b\uc544\ub4e4\uc774\ub2e4',
                'action verb',
            ),
            MorphemeExplanation('\u3134\ub2e4', '\u3134\ub2e4', 'verb ending'),
        ),
        features=(LearnerFeature('verb ending', 'ending'),),
        dictionary_entries=whole_entries,
        lexical_components=(
            LexicalComponent(
                '\ubc1b\uc544\ub4e4\uc774',
                '\ubc1b\uc544\ub4e4\uc774\ub2e4',
                'action verb',
                whole_entries,
            ),
        ),
    )
    split = AnalysisCandidate(
        '\ubc1b\uc544\ub4e4\uc778\ub2e4',
        '\ubc1b\ub2e4',
        -2.9,
        features=(LearnerFeature('verb ending', 'ending'),),
        dictionary_entries=part_entries,
        lexical_components=(
            LexicalComponent(
                '\ubc1b',
                '\ubc1b\ub2e4',
                'action verb',
                part_entries,
            ),
            LexicalComponent(
                '\ub4e4\uc774',
                '\ub4e4\uc774\ub2e4',
                'action verb',
                part_entries,
            ),
        ),
    )

    candidates = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([]),
    )._promote_close_complete_multi_component([first, split], set())

    assert candidates[0] is first


@pytest.mark.parametrize(
    ('whole_surface', 'alternative_role'),
    [
        ('\uac00\uc838\uc624', 'helping verb'),
        ('\uc0ac', 'action verb'),
    ],
)
def test_complete_predicate_allows_supported_multi_component_exceptions(
    whole_surface: str,
    alternative_role: str,
) -> None:
    whole_entries = (
        DictionaryEntry(
            'whole',
            whole_surface + '\ub2e4',
            'verb',
            None,
            None,
            (DictionarySense('whole'),),
        ),
    )
    part_entries = (
        DictionaryEntry(
            'part',
            '\uac00\uc9c0\ub2e4',
            'verb',
            None,
            None,
            (DictionarySense('part'),),
        ),
    )
    first = AnalysisCandidate(
        whole_surface + '\ub2e4',
        whole_surface + '\ub2e4',
        -1.0,
        features=(LearnerFeature('verb ending', 'ending'),),
        dictionary_entries=whole_entries,
        lexical_components=(
            LexicalComponent(
                whole_surface,
                whole_surface + '\ub2e4',
                'action verb',
                whole_entries,
            ),
        ),
    )
    alternative = AnalysisCandidate(
        whole_surface + '\ub2e4',
        '\uac00\uc9c0\ub2e4',
        -1.5,
        features=(LearnerFeature('verb ending', 'ending'),),
        dictionary_entries=part_entries,
        lexical_components=(
            LexicalComponent(
                '\uac00\uc9c0',
                '\uac00\uc9c0\ub2e4',
                'action verb',
                part_entries,
            ),
            LexicalComponent(
                '\uc624',
                '\uc624\ub2e4',
                alternative_role,
                part_entries,
            ),
        ),
    )

    candidates = KoreanAnalyzer(
        FakeDictionary(),
        FakeKiwi([]),
    )._promote_close_complete_multi_component([first, alternative], set())

    assert candidates[0] is alternative


@pytest.mark.parametrize(
    ('sentence', 'surface', 'contextual', 'isolated', 'expected_lemma'),
    [
        (
            '해,',
            '해',
            [
                ([Token('해', 'NNG', 0, 1)], -1.0),
                ([Token('하', 'VV', 0, 1), Token('어', 'ETM', 0, 1)], -6.0),
            ],
            [([Token('하', 'VV', 0, 1), Token('어', 'ETM', 0, 1)], -1.0)],
            '해',
        ),
        (
            '그런, 상황',
            '그런',
            [
                ([Token('그런', 'MM', 0, 2)], -1.0),
                ([Token('그러', 'VV', 0, 2), Token('ㄴ', 'ETM', 1, 1)], -2.5),
            ],
            [([Token('그러', 'VV', 0, 2), Token('ㄴ', 'ETM', 1, 1)], -1.0)],
            '그런',
        ),
    ],
)
def test_boundary_inflected_evidence_preserves_ambiguous_nominals(
    sentence: str,
    surface: str,
    contextual: list[tuple[list[Token], float]],
    isolated: list[tuple[list[Token], float]],
    expected_lemma: str,
) -> None:
    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == surface else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (0, len(surface)))[0]

    assert candidate.lemma == expected_lemma


def test_complete_predicate_recovery_preserves_existing_tense_features() -> None:
    entry = ReviewedRoleDictionary().lookup('바뀌다', 'verb', 1)
    component = LexicalComponent('바뀌', '바뀌다', 'action verb', entry)
    current = AnalysisCandidate(
        '바뀌였다',
        '바뀌다',
        -1.0,
        (
            MorphemeExplanation('바뀌', '바뀌다', 'action verb'),
            MorphemeExplanation('이', '이', 'word part'),
        ),
        (
            LearnerFeature('past tense', 'past'),
            LearnerFeature('verb ending', 'ending'),
        ),
        entry,
        (component,),
    )
    alternative = AnalysisCandidate(
        '바뀌였다',
        '바뀌다',
        -1.5,
        (MorphemeExplanation('바뀌', '바뀌다', 'action verb'),),
        (LearnerFeature('verb ending', 'ending'),),
        entry,
        (component,),
    )

    promoted = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        FakeKiwi([]),
    )._promote_close_complete_inflected_word([current, alternative])

    assert promoted[0] is current

def test_wrapper_context_keeps_a_pronoun_over_dictionary_order() -> None:
    sentence = "'그' 돈"
    wrapped = [
        (
            [
                Token("'", 'SS', 0, 1),
                Token('그', 'NP', 1, 1),
                Token("'", 'SS', 2, 1),
                Token('돈', 'NNG', 4, 1),
            ],
            -1.0,
        ),
        (
            [
                Token("'", 'SS', 0, 1),
                Token('그', 'MM', 1, 1),
                Token("'", 'SS', 2, 1),
                Token('돈', 'NNG', 4, 1),
            ],
            -1.5,
        ),
    ]
    unwrapped = [
        ([Token('그', 'NP', 0, 1), Token('돈', 'NNG', 2, 1)], -1.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = unwrapped if text == '그 돈' else wrapped
            return values[:top_n]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary('determiner'),
        ContextKiwi([]),
    ).analyze(sentence, (1, 2))[0]

    assert candidate.lexical_components[0].learner_role == 'pronoun'


def test_isolated_action_role_does_not_override_wrapped_hayaman_auxiliary() -> None:
    sentence = '기울여야만 /할/'
    contextual = [
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('여야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VV', 7, 1),
                Token('ㄹ', 'ETM', 7, 1),
            ],
            -1.0,
        ),
        (
            [
                Token('기울이', 'VV', 0, 3),
                Token('여야', 'EC', 2, 2),
                Token('만', 'JX', 4, 1),
                Token('하', 'VX', 7, 1),
                Token('ㄹ', 'ETM', 7, 1),
            ],
            -1.8,
        ),
    ]
    isolated = [
        ([Token('하', 'VV', 0, 1), Token('ㄹ', 'ETM', 0, 1)], -1.0),
        ([Token('하', 'VX', 0, 1), Token('ㄹ', 'ETM', 0, 1)], -2.0),
    ]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            values = isolated if text == '할' else contextual
            return values[:top_n]

    candidate = KoreanAnalyzer(
        ReviewedRoleDictionary(),
        ContextKiwi([]),
    ).analyze(sentence, (7, 8))[0]

    assert candidate.lexical_components[0].learner_role == 'helping verb'


@pytest.mark.parametrize(
    ('isolated_tag', 'expected_role'),
    [('MAG', 'adverb'), ('NNG', 'noun')],
)
def test_wrapper_adverb_requires_isolated_and_unwrapped_support(
    isolated_tag: str,
    expected_role: str,
) -> None:
    sentence = "'깊이' 다룬다"
    wrapped = [
        (
            [
                Token("'", 'SS', 0, 1),
                Token('깊이', 'NNG', 1, 2),
                Token("'", 'SS', 3, 1),
                Token('다루', 'VV', 5, 2),
                Token('ㄴ다', 'EF', 6, 1),
            ],
            -1.0,
        ),
        (
            [
                Token("'", 'SS', 0, 1),
                Token('깊이', 'MAG', 1, 2),
                Token("'", 'SS', 3, 1),
                Token('다루', 'VV', 5, 2),
                Token('ㄴ다', 'EF', 6, 1),
            ],
            -5.0,
        ),
    ]
    unwrapped = [
        (
            [
                Token('깊이', 'MAG', 0, 2),
                Token('다루', 'VV', 3, 2),
                Token('ㄴ다', 'EF', 4, 1),
            ],
            -1.0,
        ),
    ]
    isolated = [([Token('깊이', isolated_tag, 0, 2)], -1.0)]

    class ContextKiwi(FakeKiwi):
        def analyze(self, text: str, top_n: int = 1):
            if text == '깊이':
                values = isolated
            elif text == '깊이 다룬다':
                values = unwrapped
            else:
                values = wrapped
            return values[:top_n]

    candidate = KoreanAnalyzer(
        AdverbNounRoleDictionary('noun'),
        ContextKiwi([]),
    ).analyze(sentence, (1, 3))[0]

    assert candidate.lexical_components[0].learner_role == expected_role
