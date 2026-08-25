import pytest

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import DictionaryEntry, DictionarySense


class FakeKiwi:
    def analyze(self, text: str, top_n: int = 1):
        return [([(text, 'NNG', 0, len(text))], -1.0)]

    def space(self, text: str, reset_whitespace: bool = False) -> str:
        return text


class StemDictionary(DictionaryStore):
    stem = '\uc5b4\ub514'

    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma != self.stem:
            return ()
        return (
            DictionaryEntry(
                'where',
                lemma,
                'pronoun',
                None,
                None,
                (DictionarySense('where', 1),),
            ),
        )


class AmbiguousDictionary(StemDictionary):
    surface = '\uc5b4\ub514\uc5d0\uc11c'

    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        if lemma == self.surface:
            return (
                DictionaryEntry(
                    'whole',
                    lemma,
                    'noun',
                    None,
                    None,
                    (DictionarySense('whole surface', 1),),
                ),
            )
        return super().lookup(lemma, part_of_speech, limit)


def test_dictionary_backed_known_particle_is_promoted() -> None:
    surface = '\uc5b4\ub514\uc5d0\uc11c'
    candidate = KoreanAnalyzer(StemDictionary(), FakeKiwi()).analyze(
        surface, (0, len(surface))
    )[0]

    assert candidate.lemma == StemDictionary.stem
    assert candidate.dictionary_entries[0].entry_id == 'where'
    assert 'particle' in {feature.label for feature in candidate.features}


def test_known_particle_is_not_split_without_a_dictionary_stem() -> None:
    surface = '\ubbf8\ub4dc\uc5d0\uc11c'
    candidate = KoreanAnalyzer(StemDictionary(), FakeKiwi()).analyze(
        surface, (0, len(surface))
    )[0]

    assert candidate.lemma == surface
    assert not candidate.dictionary_entries


def test_existing_dictionary_analysis_is_not_replaced_by_suffix_fallback() -> None:
    surface = AmbiguousDictionary.surface
    candidate = KoreanAnalyzer(AmbiguousDictionary(), FakeKiwi()).analyze(
        surface, (0, len(surface))
    )[0]

    assert candidate.lemma == surface
    assert candidate.dictionary_entries[0].entry_id == 'whole'


class StandaloneParticleDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        entries = {
            ('를', 'particle'): ('object-particle', '를', 'particle'),
            ('르', 'noun'): ('defined-noun', '르', 'noun'),
            ('은', 'particle'): ('topic-particle', '은', 'particle'),
            ('은', 'noun'): ('silver', '은', 'noun'),
        }
        value = entries.get((lemma, part_of_speech))
        if value is None:
            return ()
        return (
            DictionaryEntry(
                value[0],
                value[1],
                value[2],
                None,
                None,
                (DictionarySense('definition', 1),),
            ),
        )


class StandaloneParticleKiwi:
    def analyze(self, text: str, top_n: int = 1):
        form = '르' if text == '를' else text
        return [([(form, 'NNG', 0, 1)], -1.0)]

    def space(self, text: str, reset_whitespace: bool = False) -> str:
        return text


def test_dictionary_backed_standalone_object_particle_is_promoted() -> None:
    candidate = KoreanAnalyzer(
        StandaloneParticleDictionary(),
        StandaloneParticleKiwi(),
    ).analyze('를', (0, 1))[0]

    assert candidate.lemma == '를'
    assert candidate.lexical_components[0].learner_role == 'particle'
    assert candidate.dictionary_entries[0].entry_id == 'object-particle'


def test_ambiguous_standalone_particle_surface_is_not_promoted() -> None:
    candidate = KoreanAnalyzer(
        StandaloneParticleDictionary(),
        StandaloneParticleKiwi(),
    ).analyze('은', (0, 1))[0]

    assert candidate.lemma == '은'
    assert candidate.lexical_components[0].learner_role == 'noun'
    assert candidate.dictionary_entries[0].entry_id == 'silver'


class ReviewedSuffixDictionary(DictionaryStore):
    def lookup(self, lemma: str, part_of_speech=None, limit: int = 10):
        role = {
            '번': '의존 명사',
            '필생': 'noun',
        }.get(lemma)
        if role is None or (part_of_speech is not None and part_of_speech != role):
            return ()
        return (
            DictionaryEntry(
                lemma,
                lemma,
                role,
                None,
                None,
                (DictionarySense('definition', 1),),
            ),
        )


class ReviewedSuffixKiwi(FakeKiwi):
    def analyze(self, text: str, top_n: int = 1):
        analyses = {
            '번쯤': [([('번쯤', 'NNG', 0, 2)], -1.0)],
            '필생토록': [
                (
                    [
                        ('필생', 'NNG', 0, 2),
                        ('토록', 'XSA', 2, 2),
                    ],
                    -1.0,
                )
            ],
        }
        return analyses[text][:top_n]


@pytest.mark.parametrize(
    ('surface', 'expected_lemma', 'expected_role'),
    [
        ('번쯤', '번', 'dependent noun'),
        ('필생토록', '필생', 'noun'),
    ],
)
def test_reviewed_particle_suffix_recovers_a_dictionary_stem(
    surface: str,
    expected_lemma: str,
    expected_role: str,
) -> None:
    candidate = KoreanAnalyzer(
        ReviewedSuffixDictionary(),
        ReviewedSuffixKiwi(),
    ).analyze(surface, (0, len(surface)))[0]

    assert candidate.lemma == expected_lemma
    assert candidate.lexical_components[0].learner_role == expected_role
    assert 'particle' in {feature.label for feature in candidate.features}
