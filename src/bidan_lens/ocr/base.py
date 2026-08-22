from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from bidan_lens.models import BoundingBox, OcrDocument


@dataclass(frozen=True, slots=True)
class DetectedRegion:
    box: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class RecognizedText:
    text: str
    confidence: float


class OcrEngine(ABC):
    @abstractmethod
    def recognize(self, image: Image.Image, *, origin: tuple[int, int] = (0, 0)) -> OcrDocument: ...
