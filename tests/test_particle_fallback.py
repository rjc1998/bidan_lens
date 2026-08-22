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
