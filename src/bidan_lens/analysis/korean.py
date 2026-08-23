from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Protocol

from bidan_lens.analysis.grammar import (
    explain_morpheme,
    known_particle_suffixes,
    learner_features,
)
from bidan_lens.analysis.verified_spacing import VERIFIED_SPACING
from bidan_lens.dictionary.store import DictionaryStore
from bidan_lens.models import AnalysisCandidate, DictionaryEntry, LexicalComponent


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


_LEXICAL_TAGS = _VERB_TAGS | {'NNG', 'NNP', 'NNB', 'NP', 'NR'}
_KIWI_ANALYSIS_DEPTH = 10
_MULTI_COMPONENT_SCORE_MARGIN = 1.5
_AUXILIARY_EXPLANATIONS = {
    '버리다': 'indicates completion of the preceding action',
}


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
        candidates = self._augment_known_particle_features(surface, candidates)
        particle = self._particle_candidate(surface, candidates)
        if particle is not None:
            remaining = (candidate for candidate in candidates if candidate is not particle)
            candidates = (particle, *remaining)[:max_candidates]
        candidates = self._isolated_defined_component_fallback(
            sentence, surface, candidates, max_candidates
        )
        correction = self.conservative_correction(surface)
        if not correction:
            return candidates
        return tuple(
            replace(
                candidate,
                interpreted_surface=correction,
            )
            for candidate in candidates
        )

    def _isolated_defined_component_fallback(
        self,
        sentence: str,
        surface: str,
        candidates: tuple[AnalysisCandidate, ...],
        max_candidates: int,
    ) -> tuple[AnalysisCandidate, ...]:
        if sentence == surface or not candidates or candidates[0].dictionary_entries:
            return candidates
        first = candidates[0]
        isolated = self._analyze_candidates(surface, (0, len(surface)), max_candidates)
        richer = next(
            (
                candidate
                for candidate in isolated
                if len(candidate.lexical_components) > len(first.lexical_components)
                and len(candidate.lexical_components) >= 2
                and all(
                    component.dictionary_entries
                    for component in candidate.lexical_components
                )
            ),
            None,
        )
        if richer is None:
            return candidates
        signature = self._candidate_signature(richer)
        promoted = next(
            (
                candidate
                for candidate in candidates
                if self._candidate_signature(candidate) == signature
            ),
            richer,
        )
        remaining = (candidate for candidate in candidates if candidate is not promoted)
        return (promoted, *remaining)[:max_candidates]

    @staticmethod
    def _candidate_signature(
        candidate: AnalysisCandidate,
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        return (
            candidate.lemma,
            tuple(
                (component.surface, component.lemma, component.learner_role)
                for component in candidate.lexical_components
            ),
        )

    @staticmethod
    def _augment_known_particle_features(
        surface: str, candidates: tuple[AnalysisCandidate, ...]
    ) -> tuple[AnalysisCandidate, ...]:
        """Recover a particle label only from an already segmented noun base."""
        augmented: list[AnalysisCandidate] = []
        noun_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
        }
        for candidate in candidates:
            labels = {feature.label for feature in candidate.features}
            labels.update(item.learner_label for item in candidate.morphemes)
            if 'particle' in labels or len(candidate.lexical_components) != 1:
                augmented.append(candidate)
                continue
            component = candidate.lexical_components[0]
            suffix = next(
                (
                    item
                    for item in known_particle_suffixes()
                    if surface.endswith(item)
                    and len(surface) > len(item)
                    and component.surface == surface[: -len(item)]
                    and component.learner_role in noun_roles
                    and bool(component.dictionary_entries)
                ),
                None,
            )
            if suffix is None:
                augmented.append(candidate)
                continue
            features = (*candidate.features, *learner_features([(suffix, 'JX')]))
            augmented.append(replace(candidate, features=features))
        return tuple(augmented)

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
                lexical_components=(
                    LexicalComponent(lemma, lemma, 'noun', entries),
                ),
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
        analyses = self.kiwi.analyze(
            sentence, top_n=max(max_candidates, _KIWI_ANALYSIS_DEPTH)
        )
        candidates: list[AnalysisCandidate] = []
        contextual_auxiliary_ids: set[int] = set()
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for rank, analysis in enumerate(analyses):
            raw_tokens, kiwi_score = analysis
            tokens = [_to_token(token) for token in raw_tokens]
            target_pairs = [
                (index, token)
                for index, token in enumerate(tokens)
                if token.start < end and token.start + max(token.length, len(token.form)) > start
            ]
            target_tokens = [token for _, token in target_pairs]
            if not target_tokens:
                continue
            lemma = self._recover_lemma(target_tokens, surface)
            key = (lemma, tuple((token.form, token.tag) for token in target_tokens))
            if key in seen:
                continue
            seen.add(key)
            first_target_index = target_pairs[0][0]
            preceding_tag = (
                _base_tag(tokens[first_target_index - 1].tag)
                if first_target_index > 0
                else None
            )
            components = self._lexical_components(target_tokens, preceding_tag)
            if components:
                lemma = components[0].lemma
            entries = components[0].dictionary_entries if components else ()
            score = float(kiwi_score) - rank * 0.05 + (0.4 if entries else 0)
            morphemes = tuple(
                explain_morpheme(token.form, self._lemma_for_token(token), token.tag)
                for token in target_tokens
            )
            features = learner_features([(token.form, token.tag) for token in target_tokens])
            candidate = AnalysisCandidate(
                surface=surface,
                lemma=lemma,
                score=score,
                morphemes=morphemes,
                features=features,
                dictionary_entries=entries,
                lexical_components=components,
                uncertain=not bool(entries),
            )
            candidates.append(candidate)
            if preceding_tag == 'EC' and any(
                component.learner_role == 'helping verb' for component in components
            ):
                contextual_auxiliary_ids.add(id(candidate))
        candidates.sort(
            key=lambda candidate: (
                id(candidate) in contextual_auxiliary_ids,
                candidate.score,
            ),
            reverse=True,
        )
        candidates = self._promote_close_complete_multi_component(
            candidates, contextual_auxiliary_ids
        )
        limited = candidates[:max_candidates]
        if len(limited) > 1:
            limited = [replace(candidate, uncertain=True) for candidate in limited]
        return tuple(limited)

    @staticmethod
    def _promote_close_complete_multi_component(
        candidates: list[AnalysisCandidate], contextual_auxiliary_ids: set[int]
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in contextual_auxiliary_ids:
            return candidates
        first = candidates[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            components = candidate.lexical_components
            if (
                first.score - candidate.score <= _MULTI_COMPONENT_SCORE_MARGIN
                and len(components) > len(first.lexical_components)
                and len(components) >= 2
                and all(component.dictionary_entries for component in components)
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

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

    def _lexical_components(
        self, tokens: list[_Token], preceding_tag: str | None = None
    ) -> tuple[LexicalComponent, ...]:
        components: list[LexicalComponent] = []
        index = 0
        previous_tag = preceding_tag
        while index < len(tokens):
            token = tokens[index]
            tag = _base_tag(token.tag)
            if tag not in _LEXICAL_TAGS:
                previous_tag = tag
                index += 1
                continue
            surface = token.form
            lemma = self._lemma_for_token(token)
            role = explain_morpheme(surface, lemma, token.tag).learner_label
            lookup_tag = tag
            if tag.startswith('N') and index + 1 < len(tokens):
                following = tokens[index + 1]
                following_tag = _base_tag(following.tag)
                if following_tag in {'XSV', 'XSA'}:
                    surface += following.form
                    lemma = surface + '다'
                    lookup_tag = following_tag
                    role = 'action verb' if following_tag == 'XSV' else 'descriptive verb'
                    index += 1
            if (
                lookup_tag in {'VV', 'VA'}
                and previous_tag == 'EC'
                and self._has_auxiliary_entry(lemma)
            ):
                lookup_tag = 'VX'
                role = 'helping verb'
            entries = self._ordered_entries(lemma, lookup_tag)
            explanation = (
                _AUXILIARY_EXPLANATIONS.get(lemma)
                or 'functions as a helping verb in this sentence'
                if lookup_tag == 'VX'
                else None
            )
            components.append(
                LexicalComponent(surface, lemma, role, entries, explanation)
            )
            previous_tag = tag
            index += 1
        return tuple(components)

    def _has_auxiliary_entry(self, lemma: str) -> bool:
        return bool(
            self.dictionary.lookup(lemma, '보조 동사', 1)
            or self.dictionary.lookup(lemma, '보조 형용사', 1)
        )

    def _ordered_entries(
        self, lemma: str, tag: str
    ) -> tuple[DictionaryEntry, ...]:
        if tag == 'VX':
            roles = ('보조 동사', '보조 형용사')
        elif tag in {'VA', 'XSA'}:
            roles = ('adjective',)
        elif tag in {'VV', 'XSV'}:
            roles = ('verb',)
        elif tag.startswith('N'):
            roles = ('noun',)
        else:
            roles = ()
        ordered: list[DictionaryEntry] = []
        seen: set[str] = set()
        for dictionary_role in roles:
            for entry in self.dictionary.lookup(lemma, dictionary_role, 10):
                if entry.entry_id not in seen:
                    seen.add(entry.entry_id)
                    ordered.append(entry)
        for entry in self.dictionary.lookup(lemma, None, 10):
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                ordered.append(entry)
        return tuple(ordered[:10])

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
        return VERIFIED_SPACING.get(normalized)
