from dataclasses import dataclass

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
    assert analyzer.conservative_correction('갔다오다') == '갔다 오다'
    assert analyzer.conservative_correction('먹고싶어요') is None


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


def test_internal_kiwi_search_is_deeper_than_popup_candidate_limit() -> None:
    kiwi = RecordingKiwi(
        [([Token("먹", "VV", 0, 1), Token("어요", "EF", 1, 2)], -1.0)]
    )

    candidates = KoreanAnalyzer(FakeDictionary(), kiwi).analyze(
        "먹어요", (0, 3), max_candidates=5
    )

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

    candidate = KoreanAnalyzer(FakeDictionary(), SentenceKiwi([])).analyze(
        "오늘 어디에", (3, 6)
    )[0]

    assert candidate.lemma == "어디"
    assert {feature.label for feature in candidate.features} == {"particle"}
    assert len(candidate.lexical_components) == 1
