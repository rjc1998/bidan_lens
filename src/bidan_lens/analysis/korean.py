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


_VERB_TAGS = {"VV", "VA", "VX", "VCN", "XSV", "XSA"}


_LEXICAL_TAGS = _VERB_TAGS | {
    'NNG',
    'NNP',
    'NNB',
    'NP',
    'NR',
    'MAG',
    'MAJ',
    'MM',
}
_KIWI_ANALYSIS_DEPTH = 10
_MULTI_COMPONENT_SCORE_MARGIN = 2.0
_SAME_LEMMA_AUXILIARY_SCORE_MARGIN = 1.5
_GE_DOEDA_AUXILIARY_SCORE_MARGIN = 10.0
_WRAPPER_CONTEXT_SCORE_MARGIN = 1.0
_WRAPPER_CONTEXT_DEPENDENT_NOUN_SCORE_MARGIN = 3.1
_WRAPPER_CORROBORATED_ADVERB_SCORE_MARGIN = 6.0
_WRAPPER_CORROBORATED_INFLECTED_PREDICATE_SCORE_MARGIN = 4.9
_LOCAL_ITDA_NOUN_SCORE_MARGIN = 4.5
_ISOLATED_VERB_ROLE_SCORE_MARGIN = 2.0
_ISOLATED_COMPLETE_CANDIDATE_SCORE_MARGIN = 3.0
_ISOLATED_DESCRIPTIVE_ITDA_SCORE_MARGIN = 11.0
_ISOLATED_INFLECTED_PREDICATE_SCORE_MARGIN = 7.0
_ISOLATED_MULTI_COMPONENT_SCORE_MARGIN = 4.25
_CONTEXTUAL_MULTI_COMPONENT_SCORE_MARGIN = 6.5
_INFLECTED_VERB_SCORE_MARGIN = 0.75
_ADNOMINAL_DEPENDENT_NOUN_SCORE_MARGIN = 1.0
_ADNOMINAL_COPULAR_DEPENDENT_NOUN_SCORE_MARGIN = 4.3
_COMPLETE_INFLECTED_SCORE_MARGIN = 2.0
_COMPLETE_DERIVED_PREDICATE_SCORE_MARGIN = 5.0
_COMPLETE_LEXICAL_ADVERB_SCORE_MARGIN = 5.0
_TERMINAL_ADVERB_ENDING_SCORE_MARGIN = 3.25
_DICTIONARY_NOMINAL_ROLE_SCORE_MARGIN = 2.5
_DICTIONARY_PROPER_NOUN_ROLE_SCORE_MARGIN = 3.2
_DICTIONARY_DEPENDENT_ROLE_SCORE_MARGIN = 2.7
_UNSUPPORTED_DEPENDENT_ROLE_SCORE_MARGIN = 4.0
_CONTEXTUAL_SIK_NOUN_SCORE_MARGIN = 5.1
_COMPOUND_TERMINAL_NOUN_SCORE_MARGIN = 6.0
_DICTIONARY_PREDICATE_ROLE_SCORE_MARGIN = 1.0
_DICTIONARY_DESCRIPTIVE_ROLE_SCORE_MARGIN = 6.1
_PRENOMINAL_DETERMINER_SCORE_MARGIN = 4.0
_DICTIONARY_PRENOMINAL_DETERMINER_SCORE_MARGIN = 5.9
_LOCATIVE_ITDA_SCORE_MARGIN = 7.0
_NONCONTEXTUAL_AUXILIARY_SCORE_MARGIN = 7.1
_NOUN_PARTICLE_SCORE_MARGIN = 3.2
_AUXILIARY_EXPLANATIONS = {
    '버리다': 'indicates completion of the preceding action',
}
_REPORTED_SPEECH_CONNECTIVES = frozenset({'다고', '라고', '냐고', '자고'})
_NON_AUXILIARY_CONNECTIVES = frozenset({'라도'})
_STANDALONE_OBJECT_PARTICLES = frozenset({'을', '를'})
_DICTIONARY_POS_BY_LEARNER_ROLE = {
    'action verb': 'verb',
    'dependent noun': '의존 명사',
    'descriptive verb': 'adjective',
    'number': 'numeral',
}


def _base_tag(tag: str) -> str:
    return tag.split("-", 1)[0]


def _is_non_auxiliary_connective(form: str | None) -> bool:
    return bool(
        form
        and any(
            form.endswith(ending)
            for ending in (
                *_REPORTED_SPEECH_CONNECTIVES,
                *_NON_AUXILIARY_CONNECTIVES,
            )
        )
    )


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
        wrapper_candidates = self._promote_close_wrapper_context_candidate(
            sentence,
            target_span,
            candidates,
            max_candidates,
        )
        wrapper_synthesized = bool(wrapper_candidates) and all(
            wrapper_candidates[0] is not candidate for candidate in candidates
        )
        candidates = self._promote_isolated_verb_role_candidate(
            sentence,
            target_span,
            surface,
            wrapper_candidates,
            max_candidates,
        )
        if wrapper_synthesized:
            candidates = wrapper_candidates
        candidates = self._promote_local_itda_noun_candidate(
            sentence,
            target_span,
            candidates,
            max_candidates,
        )
        if (
            candidates
            and candidates[0].lexical_components
            and all(
                component.learner_role == 'name or proper noun'
                for component in candidates[0].lexical_components
            )
        ):
            candidates = tuple(
                self._promote_dictionary_preferred_nominal_role(list(candidates))
            )
        candidates = tuple(self._promote_contextual_sik_noun(list(candidates)))
        candidates = tuple(
            self._promote_compound_terminal_noun(list(candidates))
        )
        candidates = self._promote_gido_auxiliary_candidate(
            sentence,
            target_span,
            candidates,
        )
        candidates = self._promote_ge_doeda_auxiliary_candidate(
            sentence,
            target_span,
            candidates,
        )
        candidates = self._augment_verified_particle_features(surface, candidates)
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

    @staticmethod
    def _promote_gido_auxiliary_candidate(
        sentence: str,
        target_span: tuple[int, int],
        candidates: tuple[AnalysisCandidate, ...],
    ) -> tuple[AnalysisCandidate, ...]:
        start, _ = target_span
        prefix = sentence[:start].rstrip()
        while prefix and unicodedata.category(prefix[-1]).startswith('P'):
            prefix = prefix[:-1].rstrip()
        if len(candidates) < 2 or not prefix.endswith('기도'):
            return candidates
        first = candidates[0]
        if len(first.lexical_components) != 1:
            return candidates
        current = first.lexical_components[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score
                > _SAME_LEMMA_AUXILIARY_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or candidate.lemma != '하다'
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface == alternative.surface
                and current.lemma == alternative.lemma
                and current.learner_role != 'helping verb'
                and alternative.learner_role == 'helping verb'
                and alternative.dictionary_entries
            ):
                return (
                    candidate,
                    *candidates[:index],
                    *candidates[index + 1 :],
                )
        return candidates

    @staticmethod
    def _promote_ge_doeda_auxiliary_candidate(
        sentence: str,
        target_span: tuple[int, int],
        candidates: tuple[AnalysisCandidate, ...],
    ) -> tuple[AnalysisCandidate, ...]:
        start, _ = target_span
        prefix = sentence[:start].rstrip()
        while prefix and unicodedata.category(prefix[-1]).startswith('P'):
            prefix = prefix[:-1].rstrip()
        if len(candidates) < 2 or not prefix.endswith('\uac8c'):
            return candidates
        first = candidates[0]
        if len(first.lexical_components) != 1:
            return candidates
        current = first.lexical_components[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score > _GE_DOEDA_AUXILIARY_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or candidate.lemma != '\ub418\ub2e4'
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface == alternative.surface
                and current.lemma == alternative.lemma
                and current.learner_role != 'helping verb'
                and alternative.learner_role == 'helping verb'
                and alternative.dictionary_entries
            ):
                return (
                    candidate,
                    *candidates[:index],
                    *candidates[index + 1 :],
                )
        return candidates

    def _promote_close_wrapper_context_candidate(
        self,
        sentence: str,
        target_span: tuple[int, int],
        candidates: tuple[AnalysisCandidate, ...],
        max_candidates: int,
    ) -> tuple[AnalysisCandidate, ...]:
        if not candidates:
            return candidates
        start, end = target_span
        left = start
        while left > 0 and unicodedata.category(sentence[left - 1]).startswith('P'):
            left -= 1
        right = end
        while right < len(sentence) and unicodedata.category(sentence[right]).startswith('P'):
            right += 1
        if left == start or right == end:
            return candidates
        unwrapped = sentence[:left] + sentence[start:end] + sentence[right:]
        unwrapped_span = (left, left + end - start)
        contextual = self._analyze_candidates(
            unwrapped,
            unwrapped_span,
            max_candidates,
        )
        if not contextual:
            return candidates
        first = candidates[0]
        alternative = contextual[0]
        alternative_labels = {feature.label for feature in alternative.features}
        alternative_labels.update(
            morpheme.learner_label for morpheme in alternative.morphemes
        )
        surface = sentence[start:end]
        if (
            not alternative.lexical_components
            and 'particle' in alternative_labels
            and self._is_whole_surface_nominal_candidate(first)
            and self.dictionary.lookup(surface, 'particle', 1)
        ):
            particle_entries = self._ordered_entries(surface, 'JKG')
            promoted = replace(
                alternative,
                score=first.score,
                dictionary_entries=particle_entries,
                lexical_components=(
                    LexicalComponent(
                        surface,
                        surface,
                        'particle',
                        particle_entries,
                    ),
                ),
                uncertain=True,
            )
            return (promoted, *candidates)[:max_candidates]
        if not alternative.dictionary_entries:
            return candidates
        first_components = first.lexical_components
        alternative_components = alternative.lexical_components
        nominal_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
        }
        if (
            len(first_components) == len(alternative_components) + 1
            and alternative_components
            and first_components[:-1] == alternative_components
            and first_components[-1].surface == '일'
            and first_components[-1].lemma == '일'
            and first_components[-1].learner_role in nominal_roles
            and alternative.lemma == first.lemma
            and 'verb ending' in alternative_labels
            and self._has_only_unrepresented_copula(alternative)
        ):
            promoted = replace(alternative, score=first.score, uncertain=True)
            return (promoted, *candidates)[:max_candidates]
        signature = self._candidate_signature(alternative)
        if self._candidate_signature(candidates[0]) == signature:
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if self._candidate_signature(candidate) != signature:
                continue
            wrapper_margin = _WRAPPER_CONTEXT_SCORE_MARGIN
            if (
                self._is_whole_surface_nominal_candidate(first)
                and self._is_whole_surface_nominal_candidate(candidate)
                and first.lexical_components[0].learner_role == 'noun'
                and candidate.lexical_components[0].learner_role == 'dependent noun'
            ):
                wrapper_margin = _WRAPPER_CONTEXT_DEPENDENT_NOUN_SCORE_MARGIN
            if (
                candidate.dictionary_entries
                and candidates[0].score - candidate.score
                <= wrapper_margin
            ):
                return (candidate, *candidates[:index], *candidates[index + 1 :])
            if (
                first.score - candidate.score
                <= _WRAPPER_CORROBORATED_INFLECTED_PREDICATE_SCORE_MARGIN
                and first.lemma != candidate.lemma
                and self._is_whole_surface_nominal_candidate(first)
                and self._is_inflected_predicate_candidate(candidate)
            ):
                isolated = self._analyze_candidates(
                    surface,
                    (0, len(surface)),
                    max_candidates,
                )
                if isolated and self._candidate_signature(isolated[0]) == signature:
                    return (
                        candidate,
                        *candidates[:index],
                        *candidates[index + 1 :],
                    )
            first_component = first.lexical_components
            alternative_component = candidate.lexical_components
            if (
                first.score - candidate.score
                <= _WRAPPER_CORROBORATED_ADVERB_SCORE_MARGIN
                and len(first_component) == 1
                and len(alternative_component) == 1
                and first_component[0].learner_role == 'noun'
                and alternative_component[0].learner_role == 'adverb'
                and first_component[0].surface == alternative_component[0].surface
                and first_component[0].lemma == alternative_component[0].lemma
                and candidate.dictionary_entries
            ):
                isolated = self._analyze_candidates(
                    surface,
                    (0, len(surface)),
                    max_candidates,
                )
                if (
                    isolated
                    and self._candidate_signature(isolated[0]) == signature
                ):
                    return (
                        candidate,
                        *candidates[:index],
                        *candidates[index + 1 :],
                    )
            return candidates
        if (
            len(candidates) >= 2
            and len({candidate.lemma for candidate in candidates}) == 1
            and self._is_inflected_predicate_candidate(contextual[0])
            and all(
                self._is_whole_surface_nominal_candidate(candidate)
                for candidate in candidates
            )
        ):
            promoted = replace(contextual[0], score=candidates[0].score, uncertain=True)
            return (promoted, *candidates)[:max_candidates]
        first_signature = self._candidate_signature(candidates[0])
        if any(
            self._candidate_signature(candidate) != first_signature
            for candidate in candidates[1:]
        ):
            return candidates
        if (
            alternative.surface != first.surface
            or alternative.lemma != first.lemma
            or len(alternative.lexical_components)
            != len(first.lexical_components)
            or not all(
                component.dictionary_entries
                for component in alternative.lexical_components
            )
        ):
            return candidates
        role_pairs: list[tuple[str, str]] = []
        for current, contextual_component in zip(
            first.lexical_components,
            alternative.lexical_components,
            strict=True,
        ):
            if (
                current.surface != contextual_component.surface
                or current.lemma != contextual_component.lemma
            ):
                return candidates
            if current.learner_role != contextual_component.learner_role:
                role_pairs.append(
                    (current.learner_role, contextual_component.learner_role)
                )
        contextual_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
            'determiner',
            'adverb',
        }
        verb_roles = {'action verb', 'descriptive verb', 'helping verb'}
        if not role_pairs or not (
            all(
                current in contextual_roles and replacement in contextual_roles
                for current, replacement in role_pairs
            )
            or all(
                current in verb_roles and replacement in verb_roles
                for current, replacement in role_pairs
            )
        ):
            return candidates
        promoted = replace(alternative, score=first.score, uncertain=True)
        return (promoted, first)[:max_candidates]

    def _promote_local_itda_noun_candidate(
        self,
        sentence: str,
        target_span: tuple[int, int],
        candidates: tuple[AnalysisCandidate, ...],
        max_candidates: int,
    ) -> tuple[AnalysisCandidate, ...]:
        if len(candidates) < 2:
            return candidates
        first = candidates[0]
        if (
            len(first.lexical_components) != 1
            or first.lexical_components[0].learner_role != 'adverb'
        ):
            return candidates
        current = first.lexical_components[0]
        preferred = self.dictionary.lookup(first.lemma, None, 1)
        if not preferred or preferred[0].part_of_speech != 'noun':
            return candidates
        eligible: list[tuple[int, AnalysisCandidate]] = []
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score > _LOCAL_ITDA_NOUN_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface == alternative.surface
                and current.lemma == alternative.lemma
                and alternative.learner_role == 'noun'
                and alternative.dictionary_entries
            ):
                eligible.append((index, candidate))
        if not eligible:
            return candidates
        start, end = target_span
        separator_end = end
        while (
            separator_end < len(sentence)
            and unicodedata.category(sentence[separator_end]).startswith('P')
        ):
            separator_end += 1
        following_start = separator_end + 1
        if (
            separator_end >= len(sentence)
            or sentence[separator_end] != ' '
            or following_start >= len(sentence)
            or sentence[following_start].isspace()
        ):
            return candidates
        following_end = following_start
        while (
            following_end < len(sentence)
            and not sentence[following_end].isspace()
        ):
            following_end += 1
        following = sentence[following_start:following_end]
        if not following.startswith('있'):
            return candidates
        surface = sentence[start:end]
        local = self._analyze_candidates(
            surface + ' ' + following,
            (0, len(surface)),
            max_candidates,
        )
        if not local or not local[0].dictionary_entries:
            return candidates
        local_signature = self._candidate_signature(local[0])
        for index, candidate in eligible:
            if self._candidate_signature(candidate) == local_signature:
                return (
                    candidate,
                    *candidates[:index],
                    *candidates[index + 1 :],
                )
        return candidates

    def _promote_isolated_verb_role_candidate(
        self,
        sentence: str,
        target_span: tuple[int, int],
        surface: str,
        candidates: tuple[AnalysisCandidate, ...],
        max_candidates: int,
    ) -> tuple[AnalysisCandidate, ...]:
        if sentence == surface or len(candidates) < 2:
            return candidates
        start, end = target_span
        punctuation_or_fragment_boundary = (
            start == 0
            or end == len(sentence)
            or (
                start > 0
                and unicodedata.category(sentence[start - 1]).startswith('P')
            )
            or (
                end < len(sentence)
                and unicodedata.category(sentence[end]).startswith('P')
            )
        )
        paired_wrapper_boundary = (
            start > 0
            and end < len(sentence)
            and unicodedata.category(sentence[start - 1]).startswith('P')
            and unicodedata.category(sentence[end]).startswith('P')
        )
        isolated = self._analyze_candidates(
            surface,
            (0, len(surface)),
            max_candidates,
        )
        if not isolated or not isolated[0].dictionary_entries:
            return candidates
        signature = self._candidate_signature(isolated[0])
        verb_roles = {'action verb', 'descriptive verb', 'helping verb'}
        first = candidates[0]
        prefix = sentence[:start].rstrip()
        while prefix and unicodedata.category(prefix[-1]).startswith('P'):
            prefix = prefix[:-1].rstrip()
        if (
            prefix.endswith(('아야만', '어야만', '여야만'))
            and any(
                component.learner_role == 'helping verb'
                for component in first.lexical_components
            )
        ):
            return candidates
        if punctuation_or_fragment_boundary:
            isolated_leader = isolated[0]
            isolated_signature = self._candidate_signature(isolated_leader)
            first_lexical_surface = ''.join(
                component.surface for component in first.lexical_components
            )
            for index, candidate in enumerate(candidates[1:], start=1):
                components = candidate.lexical_components
                candidate_lexical_surface = ''.join(
                    component.surface for component in components
                )
                complete_predicate = (
                    len(candidate_lexical_surface) > len(first_lexical_surface)
                    and self._is_inflected_predicate_candidate(candidate)
                )
                complete_number = (
                    len(first.lexical_components) > 1
                    and len(components) == 1
                    and components[0].learner_role == 'number'
                    and components[0].surface == surface
                )
                if (
                    candidate.score <= first.score
                    and first.score - candidate.score
                    <= _ISOLATED_COMPLETE_CANDIDATE_SCORE_MARGIN
                    and candidate.lemma != first.lemma
                    and self._candidate_signature(candidate) == isolated_signature
                    and components
                    and all(component.dictionary_entries for component in components)
                    and not self._has_unrepresented_word_part(candidate)
                    and (complete_predicate or complete_number)
                ):
                    return (
                        candidate,
                        *candidates[:index],
                        *candidates[index + 1 :],
                    )
        for index, candidate in enumerate(candidates[1:], start=1):
            score_margin = _ISOLATED_VERB_ROLE_SCORE_MARGIN
            if (
                not paired_wrapper_boundary
                and len(first.lexical_components) == 1
                and len(candidate.lexical_components) == 1
                and len(isolated[0].lexical_components) == 1
                and first.lexical_components[0].lemma == '있다'
                and first.lexical_components[0].learner_role == 'action verb'
                and candidate.lexical_components[0].lemma == '있다'
                and candidate.lexical_components[0].learner_role
                == 'descriptive verb'
                and isolated[0].lexical_components[0].lemma == '있다'
                and isolated[0].lexical_components[0].learner_role
                == 'descriptive verb'
            ):
                score_margin = _ISOLATED_DESCRIPTIVE_ITDA_SCORE_MARGIN
            if (
                candidate.score > first.score
                or self._candidate_signature(candidate) != signature
                or first.score - candidate.score > score_margin
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components)
                != len(first.lexical_components)
            ):
                continue
            differing_roles = []
            for current, alternative in zip(
                first.lexical_components,
                candidate.lexical_components,
                strict=True,
            ):
                if (
                    current.surface != alternative.surface
                    or current.lemma != alternative.lemma
                ):
                    differing_roles = []
                    break
                if current.learner_role != alternative.learner_role:
                    differing_roles.append(
                        (current.learner_role, alternative.learner_role)
                    )
            if punctuation_or_fragment_boundary and differing_roles and all(
                current in verb_roles and alternative in verb_roles
                for current, alternative in differing_roles
            ):
                return (candidate, *candidates[:index], *candidates[index + 1 :])
        if (
            punctuation_or_fragment_boundary
            and self._is_whole_surface_nominal_candidate(first)
            and len(first.surface) >= 2
            and first.lexical_components[0].learner_role
            in {'noun', 'name or proper noun'}
            and self._is_inflected_predicate_candidate(isolated[0])
        ):
            for index, candidate in enumerate(candidates[1:], start=1):
                if (
                    self._candidate_signature(candidate) == signature
                    and first.score - candidate.score
                    <= _ISOLATED_INFLECTED_PREDICATE_SCORE_MARGIN
                ):
                    return (
                        candidate,
                        *candidates[:index],
                        *candidates[index + 1 :],
                    )
        if self._is_dictionary_backed_adverb_hada_derivation(first):
            return candidates
        if (
            surface == first.lemma
            or not first.lexical_components
            or not all(
                component.dictionary_entries for component in first.lexical_components
            )
        ):
            return candidates
        first_labels = {feature.label for feature in first.features}
        first_labels.update(item.learner_label for item in first.morphemes)
        for evidence in isolated:
            components = evidence.lexical_components
            if (
                isolated[0].score - evidence.score
                > _ISOLATED_MULTI_COMPONENT_SCORE_MARGIN
                or len(components) <= len(first.lexical_components)
                or len(components) < 2
                or not all(component.dictionary_entries for component in components)
                or components[0].learner_role == 'number'
                or not any(
                    component.learner_role == 'helping verb'
                    for component in components
                )
                or not self._has_connective_before_helping_verb(evidence)
                or self._has_unrepresented_word_part(evidence)
                or self._has_duplicate_lexical_component(evidence)
            ):
                continue
            evidence_labels = {feature.label for feature in evidence.features}
            evidence_labels.update(item.learner_label for item in evidence.morphemes)
            if 'particle' in first_labels and 'particle' not in evidence_labels:
                continue
            signature = self._candidate_signature(evidence)
            for index, candidate in enumerate(candidates[1:], start=1):
                if (
                    self._candidate_signature(candidate) == signature
                    and first.score - candidate.score
                    <= _CONTEXTUAL_MULTI_COMPONENT_SCORE_MARGIN
                ):
                    return (
                        candidate,
                        *candidates[:index],
                        *candidates[index + 1 :],
                    )
        return candidates

    @staticmethod
    def _has_connective_before_helping_verb(candidate: AnalysisCandidate) -> bool:
        saw_ending = False
        for morpheme in candidate.morphemes:
            if morpheme.learner_label == 'verb ending':
                saw_ending = True
            elif morpheme.learner_label == 'helping verb' and saw_ending:
                return True
        return False

    @staticmethod
    def _is_dictionary_backed_adverb_hada_derivation(
        candidate: AnalysisCandidate,
    ) -> bool:
        if len(candidate.lexical_components) != 1 or len(candidate.morphemes) < 2:
            return False
        component = candidate.lexical_components[0]
        return bool(
            component.dictionary_entries
            and candidate.lemma == component.lemma == component.surface + '다'
            and candidate.morphemes[0].learner_label == 'adverb'
            and candidate.morphemes[1].surface == '하'
            and candidate.morphemes[1].learner_label == 'word part'
            and component.surface
            == candidate.morphemes[0].surface + candidate.morphemes[1].surface
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
                if candidate.lexical_components
                and len(candidate.lexical_components) >= len(first.lexical_components)
                and (
                    len(candidate.lexical_components)
                    > len(first.lexical_components)
                    or candidate.lemma != first.lemma
                )
                and all(
                    component.dictionary_entries
                    for component in candidate.lexical_components
                )
                and (
                    not self._has_unrepresented_word_part(candidate)
                    or self._has_only_unrepresented_copula(candidate)
                )
                and any(
                    feature.label in {'verb ending', 'particle'}
                    for feature in candidate.features
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
    def _has_unrepresented_word_part(candidate: AnalysisCandidate) -> bool:
        lexical_surface = ''.join(
            component.surface for component in candidate.lexical_components
        )
        return any(
            morpheme.learner_label == 'word part'
            and morpheme.surface not in lexical_surface
            for morpheme in candidate.morphemes
        )

    @staticmethod
    def _has_duplicate_lexical_component(candidate: AnalysisCandidate) -> bool:
        components = tuple(
            (component.surface, component.lemma, component.learner_role)
            for component in candidate.lexical_components
        )
        return len(set(components)) != len(components)

    @staticmethod
    def _has_only_unrepresented_copula(candidate: AnalysisCandidate) -> bool:
        lexical_surface = ''.join(
            component.surface for component in candidate.lexical_components
        )
        unrepresented = tuple(
            morpheme
            for morpheme in candidate.morphemes
            if morpheme.learner_label == 'word part'
            and morpheme.surface not in lexical_surface
        )
        return bool(unrepresented) and all(
            morpheme.surface == '이' and morpheme.lemma == '이'
            for morpheme in unrepresented
        )

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

    def _is_inflected_predicate_candidate(
        self,
        candidate: AnalysisCandidate,
    ) -> bool:
        labels = {feature.label for feature in candidate.features}
        labels.update(item.learner_label for item in candidate.morphemes)
        components = candidate.lexical_components
        return bool(
            len(components) == 1
            and components[0].learner_role
            in {'action verb', 'descriptive verb', 'helping verb'}
            and components[0].dictionary_entries
            and 'verb ending' in labels
            and not self._has_unrepresented_word_part(candidate)
        )

    @staticmethod
    def _is_whole_surface_nominal_candidate(
        candidate: AnalysisCandidate,
    ) -> bool:
        components = candidate.lexical_components
        return bool(
            len(components) == 1
            and components[0].surface == candidate.surface
            and components[0].learner_role
            in {
                'noun',
                'name or proper noun',
                'pronoun',
                'number',
                'dependent noun',
                'determiner',
            }
            and components[0].dictionary_entries
        )

    def _augment_verified_particle_features(
        self, surface: str, candidates: tuple[AnalysisCandidate, ...]
    ) -> tuple[AnalysisCandidate, ...]:
        """Recover a particle label only from a verified segmented noun prefix."""
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
            components = candidate.lexical_components
            if (
                'particle' in labels
                or not components
                or not all(
                    component.learner_role in noun_roles
                    and bool(component.dictionary_entries)
                    for component in components
                )
            ):
                augmented.append(candidate)
                continue
            lexical_surface = ''.join(component.surface for component in components)
            if not surface.startswith(lexical_surface) or len(lexical_surface) >= len(surface):
                augmented.append(candidate)
                continue
            suffix = surface[len(lexical_surface) :]
            if suffix not in known_particle_suffixes() and not self.dictionary.lookup(
                suffix, 'particle', 1
            ):
                augmented.append(candidate)
                continue
            features = (*candidate.features, *learner_features([(suffix, 'JX')]))
            augmented.append(replace(candidate, features=features))
        return tuple(augmented)

    def _particle_candidate(
        self, surface: str, candidates: tuple[AnalysisCandidate, ...]
    ) -> AnalysisCandidate | None:
        if surface in _STANDALONE_OBJECT_PARTICLES:
            entries = self.dictionary.lookup(surface, 'particle', 10)
            if entries:
                score = candidates[0].score + 0.01 if candidates else 0.4
                return AnalysisCandidate(
                    surface=surface,
                    lemma=surface,
                    score=score,
                    morphemes=(explain_morpheme(surface, surface, 'JKO'),),
                    features=learner_features([(surface, 'JKO')]),
                    dictionary_entries=entries,
                    lexical_components=(
                        LexicalComponent(surface, surface, 'particle', entries),
                    ),
                    uncertain=bool(candidates),
                )
        if (
            candidates
            and candidates[0].dictionary_entries
            and not self._has_unrepresented_word_part(candidates[0])
            and not any(
                component.learner_role == 'word part'
                for component in candidates[0].lexical_components
            )
        ):
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
            dependent = entries[0].part_of_speech == '의존 명사'
            role = 'dependent noun' if dependent else 'noun'
            tag = 'NNB' if dependent else 'NNG'
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
                    explain_morpheme(lemma, lemma, tag),
                    explain_morpheme(suffix, suffix, 'JX'),
                ),
                features=learner_features([(suffix, 'JX')]),
                dictionary_entries=entries,
                lexical_components=(
                    LexicalComponent(lemma, lemma, role, entries),
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
        prenominal_determiner_ids: set[int] = set()
        locative_itda_ids: set[int] = set()
        post_particle_inflected_verb_ids: set[int] = set()
        adnominal_dependent_noun_ids: set[int] = set()
        adnominal_copular_dependent_noun_ids: set[int] = set()
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
            preceding_context_tokens = [
                token
                for token in tokens[:first_target_index]
                if not _base_tag(token.tag).startswith('S')
            ]
            preceding_context_token = (
                preceding_context_tokens[-1] if preceding_context_tokens else None
            )
            preceding_context_tag = (
                _base_tag(preceding_context_token.tag)
                if preceding_context_token is not None
                else None
            )
            preceding_context_form = (
                preceding_context_token.form
                if preceding_context_token is not None
                else None
            )
            following_context_tokens = [
                token
                for token in tokens[target_pairs[-1][0] + 1 :]
                if not _base_tag(token.tag).startswith('S')
            ]
            following_context_tag = (
                _base_tag(following_context_tokens[0].tag)
                if following_context_tokens
                else None
            )
            nominalized_gido_context = bool(
                len(preceding_context_tokens) >= 2
                and preceding_context_tokens[-2].form == '기'
                and _base_tag(preceding_context_tokens[-2].tag) == 'ETN'
                and preceding_context_tokens[-1].form == '도'
                and _base_tag(preceding_context_tokens[-1].tag) == 'JX'
            )
            obligative_hayaman_context = bool(
                len(preceding_context_tokens) >= 2
                and preceding_context_tokens[-2].form.endswith(('아야', '어야', '여야'))
                and _base_tag(preceding_context_tokens[-2].tag) == 'EC'
                and preceding_context_tokens[-1].form == '만'
                and _base_tag(preceding_context_tokens[-1].tag) == 'JX'
            )
            lexical_hada_context = bool(
                preceding_context_token is not None
                and preceding_context_tag == 'EC'
                and (
                    preceding_context_form in {'\uc544', '\uc5b4'}
                    or (
                        preceding_context_form == '\uac8c'
                        and len(preceding_context_tokens) >= 2
                        and preceding_context_tokens[-2].form == '\uadf8\ub807'
                        and _base_tag(preceding_context_tokens[-2].tag) == 'VA'
                    )
                )
            )
            components = self._lexical_components(
                target_tokens,
                preceding_context_tag,
                preceding_context_form,
                lexical_hada_context=lexical_hada_context,
            )
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
            if (
                end < len(sentence)
                and unicodedata.category(sentence[end]).startswith('P')
                and following_context_tag is not None
                and following_context_tag.startswith('N')
                and len(components) == 1
                and components[0].learner_role == 'determiner'
            ):
                prenominal_determiner_ids.add(id(candidate))
            if (
                preceding_context_tag == 'EC'
                or nominalized_gido_context
                or obligative_hayaman_context
            ) and any(
                component.learner_role == 'helping verb' for component in components
            ):
                contextual_auxiliary_ids.add(id(candidate))
            if (
                preceding_context_tag == 'JKB'
                and len(components) == 1
                and components[0].lemma == '있다'
                and components[0].learner_role == 'descriptive verb'
            ):
                locative_itda_ids.add(id(candidate))
            if (
                preceding_context_tag is not None
                and preceding_context_tag.startswith('J')
                and any(
                    component.learner_role in {'action verb', 'descriptive verb'}
                    for component in components
                )
                and any(feature.label == 'verb ending' for feature in features)
            ):
                post_particle_inflected_verb_ids.add(id(candidate))
            if (
                following_context_tag == 'NNB'
                and any(_base_tag(token.tag) == 'ETM' for token in target_tokens)
                and any(
                    component.learner_role
                    in {'action verb', 'descriptive verb', 'helping verb'}
                    for component in components
                )
            ):
                adnominal_dependent_noun_ids.add(id(candidate))
            preceding_context_end = (
                preceding_context_token.start
                + max(
                    preceding_context_token.length,
                    len(preceding_context_token.form),
                )
                if preceding_context_token is not None
                else -1
            )
            if (
                preceding_context_tag == 'ETM'
                and preceding_context_end < start
                and sentence[preceding_context_end:start] == ' '
                and len(components) == 1
                and components[0].learner_role == 'dependent noun'
                and self.dictionary.lookup(components[0].lemma, '의존 명사', 1)
                and any(_base_tag(token.tag) == 'VCP' for token in target_tokens)
                and any(_base_tag(token.tag) == 'EF' for token in target_tokens)
            ):
                adnominal_copular_dependent_noun_ids.add(id(candidate))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = self._promote_dictionary_preferred_nominal_role(candidates)
        candidates = self._promote_prenominal_determiner(
            candidates,
            prenominal_determiner_ids,
        )
        candidates = self._promote_dictionary_preferred_predicate_role(candidates)
        candidates = self._promote_contextual_auxiliary(
            candidates, contextual_auxiliary_ids
        )
        candidates = self._promote_locative_itda(candidates, locative_itda_ids)
        candidates = self._promote_noncontextual_lexical_verb(
            candidates, contextual_auxiliary_ids
        )
        candidates = self._promote_close_inflected_verb_after_particle(
            candidates, post_particle_inflected_verb_ids
        )
        candidates = self._promote_adnominal_before_dependent_noun(
            candidates,
            adnominal_dependent_noun_ids,
        )
        candidates = self._promote_adnominal_copular_dependent_noun(
            candidates,
            adnominal_copular_dependent_noun_ids,
        )
        candidates = self._promote_close_noun_particle_candidate(candidates)
        candidates = self._promote_close_complete_multi_component(
            candidates, contextual_auxiliary_ids
        )
        candidates = self._promote_close_complete_inflected_word(candidates)
        candidates = self._promote_complete_lexical_adverb(candidates)
        candidates = self._promote_terminal_adverb_ending_candidate(candidates)
        limited = candidates[:max_candidates]
        if len(limited) > 1:
            limited = [replace(candidate, uncertain=True) for candidate in limited]
        return tuple(limited)

    def _promote_dictionary_preferred_nominal_role(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or not candidates[0].lexical_components:
            return candidates
        first = candidates[0]
        nominal_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
            'determiner',
            'adverb',
        }
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score
                > _UNSUPPORTED_DEPENDENT_ROLE_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components)
                != len(first.lexical_components)
            ):
                continue
            differing = []
            same_boundaries = True
            for current, alternative in zip(
                first.lexical_components,
                candidate.lexical_components,
                strict=True,
            ):
                if (
                    current.surface != alternative.surface
                    or current.lemma != alternative.lemma
                ):
                    same_boundaries = False
                    break
                if current.learner_role != alternative.learner_role:
                    differing.append((current, alternative))
            if not same_boundaries or not differing:
                continue
            if (
                first.score - candidate.score
                <= _UNSUPPORTED_DEPENDENT_ROLE_SCORE_MARGIN
                and all(
                    current.learner_role == 'dependent noun'
                    and alternative.learner_role == 'noun'
                    and self.dictionary.lookup(alternative.lemma, 'noun', 1)
                    and not self.dictionary.lookup(
                        alternative.lemma,
                        '의존 명사',
                        1,
                    )
                    for current, alternative in differing
                )
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
            if (
                first.score - candidate.score
                > _DICTIONARY_DEPENDENT_ROLE_SCORE_MARGIN
                and not all(
                    current.learner_role == 'name or proper noun'
                    and alternative.learner_role == 'noun'
                    for current, alternative in differing
                )
            ):
                continue
            if (
                first.score - candidate.score
                > (
                    _DICTIONARY_PROPER_NOUN_ROLE_SCORE_MARGIN
                    if all(
                        current.learner_role == 'name or proper noun'
                        and alternative.learner_role == 'noun'
                        for current, alternative in differing
                    )
                    else _DICTIONARY_NOMINAL_ROLE_SCORE_MARGIN
                )
                and not all(
                    alternative.learner_role == 'dependent noun'
                    for _, alternative in differing
                )
            ):
                continue
            if all(
                current.learner_role in nominal_roles
                and alternative.learner_role in nominal_roles
                and current.learner_role != 'dependent noun'
                and current.learner_role not in {'adverb', 'determiner'}
                and (preferred := self.dictionary.lookup(alternative.lemma, None, 1))
                and preferred[0].part_of_speech
                == _DICTIONARY_POS_BY_LEARNER_ROLE.get(
                    alternative.learner_role,
                    alternative.learner_role,
                )
                for current, alternative in differing
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    @staticmethod
    def _promote_prenominal_determiner(
        candidates: list[AnalysisCandidate],
        prenominal_determiner_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in prenominal_determiner_ids:
            return candidates
        first = candidates[0]
        if len(first.lexical_components) != 1:
            return candidates
        current = first.lexical_components[0]
        if current.learner_role not in {'noun', 'pronoun', 'number'}:
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if id(candidate) not in prenominal_determiner_ids:
                continue
            score_margin = _PRENOMINAL_DETERMINER_SCORE_MARGIN
            if (
                len(candidate.lexical_components) == 1
                and candidate.surface == candidate.lexical_components[0].surface
                and candidate.dictionary_entries
                and candidate.dictionary_entries[0].part_of_speech == 'determiner'
            ):
                score_margin = _DICTIONARY_PRENOMINAL_DETERMINER_SCORE_MARGIN
            if (
                first.score - candidate.score > score_margin
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface == alternative.surface
                and current.lemma == alternative.lemma
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_terminal_adverb_ending_candidate(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) < 2:
            return candidates
        first = candidates[0]
        terminal = first.lexical_components[-1]
        prefix = first.lexical_components[:-1]
        if (
            terminal.learner_role != 'adverb'
            or len(terminal.surface) != 1
            or not first.surface.endswith(terminal.surface)
            or not terminal.dictionary_entries
            or not any(
                component.learner_role
                in {'action verb', 'descriptive verb', 'helping verb'}
                for component in prefix
            )
            or not first.morphemes
            or first.morphemes[-1].surface != terminal.surface
            or first.morphemes[-1].learner_label != 'adverb'
            or not any(
                morpheme.learner_label == 'verb ending'
                for morpheme in first.morphemes[:-1]
            )
        ):
            return candidates
        prefix_signature = tuple(
            (component.surface, component.lemma, component.learner_role)
            for component in prefix
        )
        for index, candidate in enumerate(candidates[1:], start=1):
            candidate_signature = tuple(
                (component.surface, component.lemma, component.learner_role)
                for component in candidate.lexical_components
            )
            if (
                first.score - candidate.score
                <= _TERMINAL_ADVERB_ENDING_SCORE_MARGIN
                and candidate.surface == first.surface
                and candidate.lemma == first.lemma
                and candidate_signature == prefix_signature
                and candidate.morphemes
                and candidate.morphemes[-1].surface == terminal.surface
                and candidate.morphemes[-1].learner_label == 'verb ending'
                and self._is_inflected_predicate_candidate(candidate)
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_contextual_sik_noun(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) != 1:
            return candidates
        first = candidates[0]
        current = first.lexical_components[0]
        suffix = first.surface[len(current.surface) :]
        labels = {feature.label for feature in first.features}
        labels.update(item.learner_label for item in first.morphemes)
        contextual_suffix = suffix in {'로', '으로'} or (
            suffix.startswith('이') and 'verb ending' in labels
        )
        if (
            current.lemma != '식'
            or current.learner_role != 'dependent noun'
            or not contextual_suffix
        ):
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score > _CONTEXTUAL_SIK_NOUN_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                alternative.surface == current.surface
                and alternative.lemma == current.lemma
                and alternative.learner_role == 'noun'
                and alternative.dictionary_entries
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_compound_terminal_noun(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) < 2:
            return candidates
        first = candidates[0]
        current_components = first.lexical_components
        current_terminal = current_components[-1]
        labels = {feature.label for feature in first.features}
        labels.update(item.learner_label for item in first.morphemes)
        if (
            current_terminal.learner_role != 'dependent noun'
            or 'particle' not in labels
            or not first.surface.startswith(
                ''.join(component.surface for component in current_components)
            )
        ):
            return candidates
        preferred = self.dictionary.lookup(current_terminal.lemma, None, 1)
        if not preferred or preferred[0].part_of_speech != 'noun':
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score > _COMPOUND_TERMINAL_NOUN_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components) != len(current_components)
            ):
                continue
            alternative_components = candidate.lexical_components
            if any(
                current != alternative
                for current, alternative in zip(
                    current_components[:-1],
                    alternative_components[:-1],
                    strict=True,
                )
            ):
                continue
            alternative_terminal = alternative_components[-1]
            if (
                alternative_terminal.surface == current_terminal.surface
                and alternative_terminal.lemma == current_terminal.lemma
                and alternative_terminal.learner_role == 'noun'
                and alternative_terminal.dictionary_entries
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_dictionary_preferred_predicate_role(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) != 1:
            return candidates
        first = candidates[0]
        current = first.lexical_components[0]
        predicate_roles = {'action verb', 'descriptive verb'}
        if current.learner_role not in predicate_roles:
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                candidate.lemma != first.lemma
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface != alternative.surface
                or current.lemma != alternative.lemma
                or alternative.learner_role not in predicate_roles
            ):
                continue
            score_margin = (
                _DICTIONARY_DESCRIPTIVE_ROLE_SCORE_MARGIN
                if current.learner_role == 'action verb'
                and alternative.learner_role == 'descriptive verb'
                else _DICTIONARY_PREDICATE_ROLE_SCORE_MARGIN
            )
            if first.score - candidate.score > score_margin:
                continue
            preferred = self.dictionary.lookup(alternative.lemma, None, 1)
            if preferred and preferred[0].part_of_speech == (
                _DICTIONARY_POS_BY_LEARNER_ROLE[alternative.learner_role]
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    @staticmethod
    def _promote_contextual_auxiliary(
        candidates: list[AnalysisCandidate],
        contextual_auxiliary_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in contextual_auxiliary_ids:
            return candidates
        first = candidates[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if id(candidate) not in contextual_auxiliary_ids:
                continue
            if (
                first.lemma != candidate.lemma
                or first.score - candidate.score
                <= _SAME_LEMMA_AUXILIARY_SCORE_MARGIN
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    @staticmethod
    def _promote_locative_itda(
        candidates: list[AnalysisCandidate],
        locative_itda_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in locative_itda_ids:
            return candidates
        first = candidates[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if id(candidate) not in locative_itda_ids:
                continue
            if (
                first.score - candidate.score <= _LOCATIVE_ITDA_SCORE_MARGIN
                and first.lemma == candidate.lemma == '있다'
                and len(first.lexical_components)
                == len(candidate.lexical_components)
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_noncontextual_lexical_verb(
        self,
        candidates: list[AnalysisCandidate],
        contextual_auxiliary_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if len(candidates) < 2 or id(candidates[0]) in contextual_auxiliary_ids:
            return candidates
        first = candidates[0]
        if (
            len(first.lexical_components) != 1
            or first.lexical_components[0].learner_role != 'helping verb'
            or self._has_auxiliary_entry(first.lemma)
        ):
            return candidates
        candidate = candidates[1]
        if (
            first.score - candidate.score > _NONCONTEXTUAL_AUXILIARY_SCORE_MARGIN
            or candidate.lemma != first.lemma
            or len(candidate.lexical_components) != 1
        ):
            return candidates
        current = first.lexical_components[0]
        alternative = candidate.lexical_components[0]
        if (
            current.surface != alternative.surface
            or current.lemma != alternative.lemma
            or alternative.learner_role not in {'action verb', 'descriptive verb'}
            or not alternative.dictionary_entries
        ):
            return candidates
        return [candidate, first, *candidates[2:]]

    @staticmethod
    def _promote_close_inflected_verb_after_particle(
        candidates: list[AnalysisCandidate], inflected_verb_ids: set[int]
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in inflected_verb_ids:
            return candidates
        first = candidates[0]
        noun_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
        }
        if not first.lexical_components or not all(
            component.learner_role in noun_roles
            for component in first.lexical_components
        ):
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                id(candidate) in inflected_verb_ids
                and first.score - candidate.score <= _INFLECTED_VERB_SCORE_MARGIN
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_adnominal_before_dependent_noun(
        self,
        candidates: list[AnalysisCandidate],
        adnominal_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if (
            not candidates
            or id(candidates[0]) in adnominal_ids
            or not self._is_whole_surface_nominal_candidate(candidates[0])
        ):
            return candidates
        first = candidates[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                id(candidate) in adnominal_ids
                and first.score - candidate.score
                <= _ADNOMINAL_DEPENDENT_NOUN_SCORE_MARGIN
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    @staticmethod
    def _promote_adnominal_copular_dependent_noun(
        candidates: list[AnalysisCandidate],
        adnominal_ids: set[int],
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in adnominal_ids:
            return candidates
        first = candidates[0]
        if (
            len(first.lexical_components) != 1
            or first.lexical_components[0].learner_role != 'noun'
        ):
            return candidates
        current = first.lexical_components[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                id(candidate) not in adnominal_ids
                or first.score - candidate.score
                > _ADNOMINAL_COPULAR_DEPENDENT_NOUN_SCORE_MARGIN
                or candidate.lemma != first.lemma
                or len(candidate.lexical_components) != 1
            ):
                continue
            alternative = candidate.lexical_components[0]
            if (
                current.surface == alternative.surface
                and current.lemma == alternative.lemma
                and alternative.learner_role == 'dependent noun'
                and alternative.dictionary_entries
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_close_complete_inflected_word(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates:
            return candidates
        first = candidates[0]
        first_feature_labels = {feature.label for feature in first.features}
        incomplete = self._has_unrepresented_word_part(first) or any(
            component.learner_role == 'word part'
            for component in first.lexical_components
        )
        if not incomplete:
            return candidates
        verb_roles = {'action verb', 'descriptive verb'}
        for index, candidate in enumerate(candidates[1:], start=1):
            labels = {feature.label for feature in candidate.features}
            feature_labels = set(labels)
            labels.update(item.learner_label for item in candidate.morphemes)
            components = candidate.lexical_components
            margin = _COMPLETE_INFLECTED_SCORE_MARGIN
            if len(first.lexical_components) > 1 and len(components) == 1:
                margin = _COMPLETE_DERIVED_PREDICATE_SCORE_MARGIN
            if (
                first.score - candidate.score <= margin
                and components
                and any(component.learner_role in verb_roles for component in components)
                and all(component.dictionary_entries for component in components)
                and first_feature_labels <= feature_labels
                and 'verb ending' in labels
                and not self._has_unrepresented_word_part(candidate)
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    @staticmethod
    def _promote_complete_lexical_adverb(
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) < 2:
            return candidates
        first = candidates[0]
        nominal_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
            'determiner',
        }
        if not all(
            component.learner_role in nominal_roles
            for component in first.lexical_components
        ):
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            components = candidate.lexical_components
            if (
                first.score - candidate.score
                <= _COMPLETE_LEXICAL_ADVERB_SCORE_MARGIN
                and len(components) == 1
                and components[0].surface == candidate.surface
                and components[0].lemma == candidate.lemma
                and components[0].learner_role == 'adverb'
                and components[0].dictionary_entries
            ):
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_close_noun_particle_candidate(
        self,
        candidates: list[AnalysisCandidate],
    ) -> list[AnalysisCandidate]:
        if not candidates or len(candidates[0].lexical_components) != 1:
            return candidates
        first = candidates[0]
        first_component = first.lexical_components[0]
        if (
            len(first.surface) != 2
            or first.lemma != first.surface
            or first_component.surface != first.surface
            or first_component.lemma != first.lemma
            or first_component.learner_role != 'name or proper noun'
            or not first_component.dictionary_entries
        ):
            return candidates
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                first.score - candidate.score > _NOUN_PARTICLE_SCORE_MARGIN
                or candidate.surface != first.surface
                or len(candidate.lexical_components) != 1
            ):
                continue
            component = candidate.lexical_components[0]
            if (
                component.learner_role != 'noun'
                or len(component.surface) != 1
                or component.lemma != candidate.lemma
                or not component.dictionary_entries
                or not candidate.dictionary_entries
                or not first.surface.startswith(component.surface)
            ):
                continue
            suffix = first.surface[len(component.surface) :]
            particle_surface = ''.join(
                morpheme.surface
                for morpheme in candidate.morphemes
                if morpheme.learner_label == 'particle'
            )
            known_particle = (
                suffix in known_particle_suffixes()
                or bool(self.dictionary.lookup(suffix, 'particle', 1))
            )
            if known_particle and particle_surface == suffix:
                return [candidate, *candidates[:index], *candidates[index + 1 :]]
        return candidates

    def _promote_close_complete_multi_component(
        self,
        candidates: list[AnalysisCandidate], contextual_auxiliary_ids: set[int]
    ) -> list[AnalysisCandidate]:
        if not candidates or id(candidates[0]) in contextual_auxiliary_ids:
            return candidates
        first = candidates[0]
        protected_complete_predicate = bool(
            self._is_inflected_predicate_candidate(first)
            and len(first.lexical_components[0].surface) >= 2
        )
        noun_roles = {
            'noun',
            'name or proper noun',
            'pronoun',
            'number',
            'dependent noun',
            'adverb',
        }
        if (
            len(first.lexical_components) >= 2
            and all(
                component.dictionary_entries
                for component in first.lexical_components
            )
        ):
            return candidates
        if (
            len(first.lexical_components) == 1
            and first.lexical_components[0].learner_role in noun_roles
            and first.lexical_components[0].dictionary_entries
        ):
            return candidates
        first_labels = {feature.label for feature in first.features}
        first_labels.update(item.learner_label for item in first.morphemes)
        for index, candidate in enumerate(candidates[1:], start=1):
            components = candidate.lexical_components
            candidate_labels = {feature.label for feature in candidate.features}
            candidate_labels.update(item.learner_label for item in candidate.morphemes)
            if (
                first.score - candidate.score <= _MULTI_COMPONENT_SCORE_MARGIN
                and len(components) > len(first.lexical_components)
                and len(components) >= 2
                and components[0].learner_role != 'number'
                and all(component.dictionary_entries for component in components)
                and not (
                    protected_complete_predicate
                    and not any(
                        component.learner_role == 'helping verb'
                        for component in components
                    )
                )
                and not (
                    'particle' in first_labels and 'particle' not in candidate_labels
                )
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
        self,
        tokens: list[_Token],
        preceding_tag: str | None = None,
        preceding_form: str | None = None,
        *,
        lexical_hada_context: bool = False,
    ) -> tuple[LexicalComponent, ...]:
        components: list[LexicalComponent] = []
        index = 0
        previous_tag = preceding_tag
        previous_form = preceding_form
        while index < len(tokens):
            component_index = index
            token = tokens[index]
            tag = _base_tag(token.tag)
            if (
                token.length <= 0
                and components
                and components[-1].surface.endswith(token.form)
            ):
                index += 1
                continue
            if tag == 'XR' and index + 1 < len(tokens):
                suffix = tokens[index + 1]
                if _base_tag(suffix.tag) == 'XSA':
                    surface = token.form + suffix.form
                    lemma = surface + '다'
                    entries = self._ordered_entries(lemma, 'XSA')
                    if entries:
                        components.append(
                            LexicalComponent(
                                surface,
                                lemma,
                                'descriptive verb',
                                entries,
                            )
                        )
                        previous_tag = 'XSA'
                        previous_form = suffix.form
                        index += 2
                        continue
            if tag not in _LEXICAL_TAGS:
                previous_tag = tag
                previous_form = token.form
                index += 1
                continue
            surface = token.form
            lemma = self._lemma_for_token(token)
            role = explain_morpheme(surface, lemma, token.tag).learner_label
            lookup_tag = tag
            if tag in {'MAG', 'MAJ'} and index + 1 < len(tokens):
                verb_suffix = tokens[index + 1]
                verb_suffix_tag = _base_tag(verb_suffix.tag)
                if (
                    verb_suffix_tag in {'XSV', 'XSA'}
                    and verb_suffix.form == '하'
                ):
                    derived_surface = token.form + verb_suffix.form
                    derived_lemma = derived_surface + '다'
                    if self._ordered_entries(derived_lemma, verb_suffix_tag):
                        surface = derived_surface
                        lemma = derived_lemma
                        lookup_tag = verb_suffix_tag
                        role = (
                            'action verb'
                            if verb_suffix_tag == 'XSV'
                            else 'descriptive verb'
                        )
                        index += 1
            if tag.startswith('N') and index + 2 < len(tokens):
                noun_suffix = tokens[index + 1]
                verb_suffix = tokens[index + 2]
                if (
                    _base_tag(noun_suffix.tag) == 'XSN'
                    and noun_suffix.form == '화'
                    and _base_tag(verb_suffix.tag) == 'XSV'
                    and verb_suffix.form in {'하', '되'}
                ):
                    derived_surface = token.form + noun_suffix.form + verb_suffix.form
                    derived_lemma = derived_surface + '다'
                    if self._ordered_entries(derived_lemma, 'XSV'):
                        surface = derived_surface
                        lemma = derived_lemma
                        lookup_tag = 'XSV'
                        role = 'action verb'
                        index += 2
            if tag.startswith('N') and lookup_tag == tag and index + 1 < len(tokens):
                following = tokens[index + 1]
                following_tag = _base_tag(following.tag)
                if following_tag in {'XSV', 'XSA'}:
                    surface += following.form
                    lemma = surface + '다'
                    lookup_tag = following_tag
                    role = 'action verb' if following_tag == 'XSV' else 'descriptive verb'
                    index += 1
            if tag.startswith('N') and index > 0:
                prefix = tokens[component_index - 1]
                if _base_tag(prefix.tag) == 'XPN':
                    prefixed = prefix.form + lemma
                    if self._ordered_entries(prefixed, lookup_tag):
                        surface = prefix.form + surface
                        lemma = prefixed
            if tag.startswith('N') and index + 1 < len(tokens):
                suffix = tokens[index + 1]
                following_tag = _base_tag(suffix.tag)
                tag_after_suffix = (
                    None
                    if index + 2 >= len(tokens)
                    else _base_tag(tokens[index + 2].tag)
                )
                suffix_closes_noun = (
                    tag_after_suffix is None or tag_after_suffix.startswith('J')
                )
                dictionary_backed_suffix = bool(
                    following_tag == 'XSN'
                    and suffix.form == '화'
                    and tag_after_suffix not in {'VCP', 'VCN'}
                    and self._ordered_entries(lemma + suffix.form, 'NNG')
                )
                if (
                    following_tag == 'XSN'
                    and suffix.form != '들'
                    and (suffix_closes_noun or dictionary_backed_suffix)
                ):
                    surface += suffix.form
                    lemma += suffix.form
                    index += 1
            if lookup_tag in {'VV', 'VA', 'XSV', 'XSA'} and (
                self._has_only_auxiliary_entries(lemma)
                or (
                    previous_tag == 'EC'
                    and not _is_non_auxiliary_connective(previous_form)
                    and self._has_auxiliary_entry(lemma)
                    and not (lexical_hada_context and lemma == '\ud558\ub2e4')
                )
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
            previous_form = token.form
            index += 1
        return tuple(components)

    def _has_auxiliary_entry(self, lemma: str) -> bool:
        return bool(
            self.dictionary.lookup(lemma, '보조 동사', 1)
            or self.dictionary.lookup(lemma, '보조 형용사', 1)
        )

    def _has_only_auxiliary_entries(self, lemma: str) -> bool:
        return bool(
            self._has_auxiliary_entry(lemma)
            and not self.dictionary.lookup(lemma, 'verb', 1)
            and not self.dictionary.lookup(lemma, 'adjective', 1)
        )

    def _ordered_entries(
        self, lemma: str, tag: str
    ) -> tuple[DictionaryEntry, ...]:
        if tag == 'VX':
            roles = ('보조 동사', '보조 형용사')
        elif tag in {'VA', 'VCN', 'XSA'}:
            roles = ('adjective',)
        elif tag in {'VV', 'XSV'}:
            roles = ('verb',)
        elif tag in {'MAG', 'MAJ'}:
            roles = ('adverb',)
        elif tag == 'MM':
            roles = ('determiner',)
        elif tag.startswith('J'):
            roles = ('particle',)
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
