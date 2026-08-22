from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("invalid bounding box")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def contains(self, x: float, y: float, padding: float = 0) -> bool:
        return (
            self.left - padding <= x <= self.right + padding
            and self.top - padding <= y <= self.bottom + padding
        )

    def translated(self, x: float, y: float) -> BoundingBox:
        return BoundingBox(self.left + x, self.top + y, self.right + x, self.bottom + y)

    @classmethod
    def union(cls, boxes: Sequence[BoundingBox]) -> BoundingBox:
        if not boxes:
            raise ValueError("cannot union an empty sequence")
        return cls(
            min(box.left for box in boxes),
            min(box.top for box in boxes),
            max(box.right for box in boxes),
            max(box.bottom for box in boxes),
        )


@dataclass(frozen=True, slots=True)
class OcrGlyph:
    text: str
    box: BoundingBox
    confidence: float
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class OcrEojeol:
    text: str
    box: BoundingBox
    confidence: float
    sentence_start: int
    sentence_end: int
    glyphs: tuple[OcrGlyph, ...] = ()


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    box: BoundingBox
    confidence: float
    eojeols: tuple[OcrEojeol, ...]


@dataclass(frozen=True, slots=True)
class OcrDocument:
    lines: tuple[OcrLine, ...]
    captured_at: float
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True, slots=True)
class HoverTarget:
    surface: str
    sentence: str
    sentence_start: int
    sentence_end: int
    box: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class DictionarySense:
    definition: str
    order: int = 0
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    entry_id: str
    headword: str
    part_of_speech: str | None
    homograph_number: str | None
    vocabulary_level: str | None
    senses: tuple[DictionarySense, ...]
    source: str = "KRDict"


@dataclass(frozen=True, slots=True)
class LearnerFeature:
    label: str
    explanation: str
    surface: str = ""


@dataclass(frozen=True, slots=True)
class MorphemeExplanation:
    surface: str
    lemma: str
    learner_label: str


@dataclass(frozen=True, slots=True)
class AnalysisCandidate:
    surface: str
    lemma: str
    score: float
    morphemes: tuple[MorphemeExplanation, ...] = ()
    features: tuple[LearnerFeature, ...] = ()
    dictionary_entries: tuple[DictionaryEntry, ...] = ()
    interpreted_surface: str | None = None
    uncertain: bool = False


@dataclass(frozen=True, slots=True)
class PopupResult:
    target: HoverTarget
    candidates: tuple[AnalysisCandidate, ...]
    selected_index: int = 0
    requested_at: float | None = None

    @property
    def selected(self) -> AnalysisCandidate | None:
        return self.candidates[self.selected_index] if self.candidates else None

    def with_index(self, index: int) -> PopupResult:
        if not self.candidates:
            return self
        return PopupResult(
            self.target,
            self.candidates,
            index % len(self.candidates),
            self.requested_at,
        )


def immutable_metadata(values: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return MappingProxyType(dict(values or {}))
