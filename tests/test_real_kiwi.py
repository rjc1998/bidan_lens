import pytest

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import DictionaryEntry, DictionarySense

pytest.importorskip("kiwipiepy")


class LemmaDictionary(DictionaryStore):
    def lookup(self, lemma, part_of_speech=None, limit=10):
        return (
            DictionaryEntry(
                lemma,
                lemma,
                part_of_speech,
                None,
                None,
                (DictionarySense("fixture definition"),),
            ),
        )


@pytest.mark.parametrize(
    ("surface", "lemma", "labels"),
    [
        ("먹었습니다", "먹다", {"past tense", "formal polite style"}),
        ("먹으셨어요", "먹다", {"honorific", "past tense", "polite style"}),
        ("들었어요", "듣다", {"past tense", "polite style"}),
        ("공부해요", "공부하다", {"polite style"}),
        ("어디에서", "어디", {"particle"}),
    ],
)
def test_real_kiwi_canonical_forms(surface, lemma, labels) -> None:
    candidate = KoreanAnalyzer(LemmaDictionary()).analyze(surface, (0, len(surface)))[0]
    assert candidate.lemma == lemma
    assert labels <= {feature.label for feature in candidate.features}
