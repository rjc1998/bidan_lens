from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Protocol

from bidan_lens.analysis.grammar import (
    explain_morpheme,
    known_particle_suffixes,
    learner_features,
)
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import AnalysisCandidate


class KiwiLike(Protocol):
    def analyze(self, text: str, top_n: int = 1) -> list[Any]: ...

    def space(self, text: str, reset_whitespace: bool = False) -> str: ...


@dataclass(frozen=True, slots=True)
class _Token:
    form: str
    tag: str
    start: int
    length: int


_VERB_TAGS = {"VV", "VA", "VX", "XSV", "XSA"}


def _base_tag(tag: str) -> str:
    return tag.split("-", 1)[0]


def _to_token(value: Any) -> _Token:
    return _Token(
        str(getattr(value, "form", value[0] if isinstance(value, tuple) else "")),
        str(getattr(value, "tag", value[1] if isinstance(value, tuple) else "")),
        int(
            getattr(value, "start", value[2] if isinstance(value, tuple) and len(value) > 2 else 0)
        ),
        int(getattr(value, "len", value[3] if isinstance(value, tuple) and len(value) > 3 else 0)),
    )


class KoreanAnalyzer:
    def __init__(self, dictionary: DictionaryStore, kiwi: KiwiLike | None = None) -> None:
        self.dictionary = dictionary
        self._kiwi = kiwi

    @property
    def kiwi(self) -> KiwiLike:
        if self._kiwi is None:
            from kiwipiepy import Kiwi

            self._kiwi = Kiwi()
        return self._kiwi

    def analyze(
        self,
        sentence: str,
        target_span: tuple[int, int],
        max_candidates: int = 5,
    ) -> tuple[AnalysisCandidate, ...]:
        sentence = unicodedata.normalize("NFC", sentence)
        start, end = target_span
        surface = sentence[start:end]
        candidates = self._analyze_candidates(sentence, target_span, max_candidates)
        particle = self._particle_candidate(surface, candidates)
        if particle is not None:
            remaining = (candidate for candidate in candidates if candidate is not particle)
            candidates = (particle, *remaining)[:max_candidates]
        if any(candidate.dictionary_entries for candidate in candidates):
            return candidates
        correction = self.conservative_correction(surface)
        if not correction:
            return candidates
        corrected_sentence = sentence[:start] + correction + sentence[end:]
        corrected = self._analyze_candidates(
            corrected_sentence, (start, start + len(correction)), max_candidates
        )
        marked = tuple(
            replace(
                candidate,
                surface=surface,
                interpreted_surface=correction,
                score=candidate.score - 0.1,
                uncertain=True,
            )
            for candidate in corrected
            if candidate.dictionary_entries
        )
        # Original analyses stay first until a correction corpus calibrates safe promotion.
        return tuple((*candidates, *marked))[:max_candidates]

    def _particle_candidate(
        self, surface: str, candidates: tuple[AnalysisCandidate, ...]
    ) -> AnalysisCandidate | None:
        if candidates and candidates[0].dictionary_entries:
            return None
        for suffix in known_particle_suffixes():
            if not surface.endswith(suffix) or len(surface) <= len(suffix):
                continue
            lemma = surface[: -len(suffix)]
            entries = self.dictionary.lookup(lemma, 'noun', 10)
            if not entries:
                entries = self.dictionary.lookup(lemma, None, 10)
            if not entries:
                continue
            for candidate in candidates:
                labels = {feature.label for feature in candidate.features}
                labels.update(item.learner_label for item in candidate.morphemes)
                if candidate.lemma == lemma and 'particle' in labels:
                    return candidate
            score = candidates[0].score + 0.01 if candidates else 0.4
            return AnalysisCandidate(
                surface=surface,
                lemma=lemma,
                score=score,
                morphemes=(
                    explain_morpheme(lemma, lemma, 'NNG'),
                    explain_morpheme(suffix, suffix, 'JX'),
                ),
                features=learner_features([(suffix, 'JX')]),
                dictionary_entries=entries,
                uncertain=bool(candidates),
            )
        return None

    def _analyze_candidates(
        self,
        sentence: str,
        target_span: tuple[int, int],
        max_candidates: int,
    ) -> tuple[AnalysisCandidate, ...]:
        start, end = target_span
        surface = sentence[start:end]
        analyses = self.kiwi.analyze(sentence, top_n=max_candidates)
        candidates: list[AnalysisCandidate] = []
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for rank, analysis in enumerate(analyses):
            raw_tokens, kiwi_score = analysis
            tokens = [_to_token(token) for token in raw_tokens]
            target_tokens = [
                token
                for token in tokens
                if token.start < end and token.start + max(token.length, len(token.form)) > start
            ]
            if not target_tokens:
                continue
            lemma = self._recover_lemma(target_tokens, surface)
            key = (lemma, tuple((token.form, token.tag) for token in target_tokens))
            if key in seen:
                continue
            seen.add(key)
            entries = self.dictionary.lookup(lemma, self._dictionary_pos(target_tokens), 10)
            if not entries:
                # Context-free morphology can choose the wrong POS for a homograph.
                # Keep the lemma and recover its learner definition without hiding
                # the competing analysis.
                entries = self.dictionary.lookup(lemma, None, 10)
            score = float(kiwi_score) - rank * 0.05 + (0.4 if entries else 0)
            morphemes = tuple(
                explain_morpheme(token.form, self._lemma_for_token(token), token.tag)
                for token in target_tokens
            )
            features = learner_features([(token.form, token.tag) for token in target_tokens])
            candidates.append(
                AnalysisCandidate(
                    surface=surface,
                    lemma=lemma,
                    score=score,
                    morphemes=morphemes,
                    features=features,
                    dictionary_entries=entries,
                    uncertain=not bool(entries),
                )
            )
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        limited = candidates[:max_candidates]
        if len(limited) > 1:
            limited = [replace(candidate, uncertain=True) for candidate in limited]
        return tuple(limited)

    @staticmethod
    def _recover_lemma(tokens: list[_Token], surface: str) -> str:
        lexical = [
            token
            for token in tokens
            if _base_tag(token.tag) in _VERB_TAGS or token.tag.startswith("N")
        ]
        if not lexical:
            return surface
        for index, token in enumerate(lexical):
            if token.tag.startswith("N") and index + 1 < len(lexical):
                following = lexical[index + 1]
                if _base_tag(following.tag) == "XSV" and following.form in {"하", "되", "시키"}:
                    return token.form + following.form + "다"
        token = lexical[0]
        return KoreanAnalyzer._lemma_for_token(token)

    @staticmethod
    def _lemma_for_token(token: _Token) -> str:
        if _base_tag(token.tag) in _VERB_TAGS:
            return token.form if token.form.endswith("다") else token.form + "다"
        return token.form

    @staticmethod
    def _dictionary_pos(tokens: list[_Token]) -> str | None:
        if any(_base_tag(token.tag) == "VV" for token in tokens):
            return "verb"
        if any(_base_tag(token.tag) == "VA" for token in tokens):
            return "adjective"
        if any(token.tag.startswith("N") for token in tokens):
            return "noun"
        return None

    def conservative_correction(self, surface: str) -> str | None:
        """Offer one marked spacing correction; never silently alter OCR output."""
        normalized = unicodedata.normalize("NFC", surface)
        corrected = self.kiwi.space(normalized, reset_whitespace=False)
        if corrected == normalized:
            return None
        # A v1 correction may change whitespace only. Character edits are not trusted.
        if corrected.replace(" ", "") != normalized.replace(" ", ""):
            return None
        return corrected
