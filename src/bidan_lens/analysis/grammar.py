from __future__ import annotations

from bidan_lens.models import LearnerFeature, MorphemeExplanation

_PARTICLE_EXPLANATIONS = {
    "은": "topic particle; marks what the sentence is about",
    "는": "topic particle; marks what the sentence is about",
    "이": "subject particle; marks who or what performs/is described",
    "가": "subject particle; marks who or what performs/is described",
    "을": "object particle; marks what receives the action",
    "를": "object particle; marks what receives the action",
    "에": "location/time particle; often means at, in, on, or to",
    "에서": "location particle; where an action happens or starts",
    "에게": "recipient particle; often means to someone",
    "도": "additive particle; means also or even",
    "만": "limiting particle; means only",
    "와": "joining particle; means and or with",
    "과": "joining particle; means and or with",
    "쯤": "approximation particle; means about or approximately",
    "토록": "extent particle; means throughout or to the extent of",
}

_ENDING_EXPLANATIONS = {
    "었": ("past tense", "places the event or state in the past"),
    "았": ("past tense", "places the event or state in the past"),
    "겠": ("future or intention", "often expresses intention, prediction, or conjecture"),
    "시": ("honorific", "shows respect toward the person performing the action"),
    "세요": ("polite request", "a polite ending often used for requests or instructions"),
    "어요": ("polite style", "a polite, everyday sentence ending"),
    "아요": ("polite style", "a polite, everyday sentence ending"),
    "습니다": ("formal polite style", "a formal, polite statement ending"),
    "ㅂ니다": ("formal polite style", "a formal, polite statement ending"),
    "고": ("connecting ending", "connects this action or state to what follows"),
    "지만": ("contrast ending", "connects clauses with a meaning like but or although"),
    "면": ("conditional ending", "connects clauses with a meaning like if or when"),
}

_TAG_LABELS = {
    "NNG": "noun",
    "NNP": "name or proper noun",
    "NNB": "dependent noun",
    "NP": "pronoun",
    "NR": "number",
    "VV": "action verb",
    "VA": "descriptive verb",
    "VX": "helping verb",
    "MAG": "adverb",
    "MAJ": "adverb",
    "MM": "determiner",
    "VCN": "descriptive verb",
}


def known_particle_suffixes() -> tuple[str, ...]:
    return tuple(sorted(_PARTICLE_EXPLANATIONS, key=len, reverse=True))


def explain_morpheme(form: str, lemma: str, tag: str) -> MorphemeExplanation:
    base_tag = tag.split("-", 1)[0]
    if tag.startswith("J"):
        label = "particle"
    elif base_tag in {"EP", "EF", "EC", "ETN", "ETM"}:
        label = "verb ending"
    else:
        label = _TAG_LABELS.get(base_tag, "word part")
    return MorphemeExplanation(form, lemma, label)


def learner_features(morphemes: list[tuple[str, str]]) -> tuple[LearnerFeature, ...]:
    features: list[LearnerFeature] = []
    for form, tag in morphemes:
        if tag.startswith("J"):
            explanation = _PARTICLE_EXPLANATIONS.get(
                form, "particle attached to the preceding word"
            )
            features.append(LearnerFeature("particle", explanation, form))
            continue
        if tag not in {"EP", "EF", "EC", "ETN", "ETM"}:
            continue
        matched = False
        for ending, (label, explanation) in _ENDING_EXPLANATIONS.items():
            if form == ending or form.endswith(ending):
                feature = LearnerFeature(label, explanation, ending)
                if feature not in features:
                    features.append(feature)
                matched = True
        if not matched:
            features.append(
                LearnerFeature("verb ending", "an ending that shapes how the verb is used", form)
            )
    return tuple(features)
