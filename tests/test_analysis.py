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
