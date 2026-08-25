from dataclasses import dataclass

import pytest

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import DictionaryEntry, DictionarySense


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


def test_dictionary_noun_entry_can_disambiguate_a_proper_noun_tag() -> None:
    analyses = [
        ([Token('그', 'NNP', 0, 1)], -1.0),
        ([Token('그', 'NNG', 0, 1)], -2.5),
    ]

    candidate = KoreanAnalyzer(
        NominalRoleDictionary('noun'),
        FakeKiwi(analyses),
    ).analyze('그', (0, 1))[0]

    assert candidate.lexical_components[0].learner_role == 'noun'


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
