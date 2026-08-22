from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from bidan_lens.models import BoundingBox, OcrDocument
from bidan_lens.ocr.base import DetectedRegion, OcrEngine, RecognizedText
from bidan_lens.ocr.hangul import contains_hangul, make_line


def _session(model: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, (os_cpu_count() or 2) - 1))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _input_name(session: Any) -> str:
    return session.get_inputs()[0].name


def _resize_for_detection(image: Image.Image, limit: int = 960) -> tuple[Image.Image, float, float]:
    width, height = image.size
    scale = min(1.0, limit / max(width, height))
    resized_width = max(32, int(math.ceil(width * scale / 32) * 32))
    resized_height = max(32, int(math.ceil(height * scale / 32) * 32))
    resized = image.convert("RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    return resized, width / resized_width, height / resized_height


def _normalize(image: Image.Image) -> np.ndarray:
    # PaddleOCR's exported inference transforms operate on OpenCV BGR images.
    pixels = np.asarray(image, dtype=np.float32)[..., ::-1] / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)


def _components(mask: np.ndarray, scores: np.ndarray, minimum_pixels: int = 6) -> list[tuple]:
    """Small dependency-free connected-component pass for DB detector probability maps."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    found: list[tuple[int, int, int, int, float]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            values: list[float] = []
            while queue:
                current_x, current_y = queue.popleft()
                values.append(float(scores[current_y, current_x]))
                min_x, max_x = min(min_x, current_x), max(max_x, current_x)
                min_y, max_y = min(min_y, current_y), max(max_y, current_y)
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_x, next_y))
            if len(values) >= minimum_pixels:
                found.append((min_x, min_y, max_x + 1, max_y + 1, sum(values) / len(values)))
    return found


class PaddleDetector:
    def __init__(
        self, model_path: Path, session: Any | None = None, threshold: float = 0.3
    ) -> None:
        self.session = session or _session(model_path)
        self.threshold = threshold

    def detect(self, image: Image.Image) -> tuple[DetectedRegion, ...]:
        resized, scale_x, scale_y = _resize_for_detection(image)
        output = self.session.run(None, {_input_name(self.session): _normalize(resized)})[0]
        probability = np.asarray(output).squeeze()
        if probability.ndim != 2:
            raise RuntimeError(f"unexpected Paddle detector output shape: {output.shape}")
        components = _components(probability >= self.threshold, probability)
        map_height, map_width = probability.shape
        image_width, image_height = resized.size
        map_x, map_y = image_width / map_width, image_height / map_height
        regions: list[DetectedRegion] = []
        for left, top, right, bottom, confidence in components:
            # Expand small DB-map regions enough to capture ascenders and punctuation.
            padding_x = max(1, (right - left) * 0.08)
            padding_y = max(1, (bottom - top) * 0.40)
            box = BoundingBox(
                max(0, (left - padding_x) * map_x * scale_x),
                max(0, (top - padding_y) * map_y * scale_y),
                min(image.size[0], (right + padding_x) * map_x * scale_x),
                min(image.size[1], (bottom + padding_y) * map_y * scale_y),
            )
            if box.width >= 4 and box.height >= 4:
                regions.append(DetectedRegion(box, confidence))
        return tuple(sorted(regions, key=lambda region: (region.box.top, region.box.left)))


class PaddleRecognizer:
    def __init__(
        self,
        model_path: Path,
        characters_path: Path,
        session: Any | None = None,
        input_height: int = 48,
        maximum_width: int = 320,
    ) -> None:
        self.session = session or _session(model_path)
        raw = characters_path.read_text(encoding="utf-8").splitlines()
        self.characters = ["<blank>", *raw, " "]
        self.input_height = input_height
        self.maximum_width = maximum_width

    def recognize(self, image: Image.Image) -> RecognizedText:
        rgb = image.convert("RGB")
        target_width = min(
            self.maximum_width,
            max(16, int(round(rgb.width * self.input_height / max(1, rgb.height)))),
        )
        resized = rgb.resize((target_width, self.input_height), Image.Resampling.BICUBIC)
        pixels = np.asarray(resized, dtype=np.float32)[..., ::-1] / 255.0
        normalized = np.transpose((pixels - 0.5) / 0.5, (2, 0, 1))
        # RecResizeImg pads the already-normalized tensor with zeroes. Keep the
        # configured width fixed so this is byte-for-byte compatible with the
        # Korean PP-OCRv5 inference transform.
        tensor = np.zeros((1, 3, self.input_height, self.maximum_width), dtype=np.float32)
        tensor[0, :, :, :target_width] = normalized
        output = np.asarray(self.session.run(None, {_input_name(self.session): tensor})[0])
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise RuntimeError(f"unexpected Paddle recognizer output shape: {output.shape}")
        # Some exports use [classes, timesteps]. The character dimension is the larger match.
        if output.shape[0] == len(self.characters) and output.shape[1] != len(self.characters):
            output = output.T
        row_sums = np.sum(output, axis=1)
        if np.all(output >= 0) and np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-4):
            probabilities = output
        else:
            shifted = output - np.max(output, axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        indices = probabilities.argmax(axis=1)
        text: list[str] = []
        confidences: list[float] = []
        previous = -1
        for timestep, index in enumerate(indices):
            index = int(index)
            if index != 0 and index != previous and index < len(self.characters):
                text.append(self.characters[index])
                confidences.append(float(probabilities[timestep, index]))
            previous = index
        return RecognizedText("".join(text).strip(), min(confidences, default=0.0))


class PaddleOcrEngine(OcrEngine):
    def __init__(
        self,
        detector: PaddleDetector,
        recognizer: PaddleRecognizer,
        retry_threshold: float = 0.72,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.retry_threshold = retry_threshold

    @classmethod
    def from_asset_directory(cls, directory: Path) -> PaddleOcrEngine:
        configuration = json.loads((directory / "ocr.json").read_text(encoding="utf-8"))
        return cls(
            PaddleDetector(directory / configuration["detection_model"]),
            PaddleRecognizer(
                directory / configuration["recognition_model"],
                directory / configuration["character_dictionary"],
            ),
        )

    def recognize(self, image: Image.Image, *, origin: tuple[int, int] = (0, 0)) -> OcrDocument:
        lines = []
        for region in self.detector.detect(image):
            crop = image.crop(
                (
                    math.floor(region.box.left),
                    math.floor(region.box.top),
                    math.ceil(region.box.right),
                    math.ceil(region.box.bottom),
                )
            )
            recognized = self.recognizer.recognize(crop)
            if recognized.confidence < self.retry_threshold:
                retry_image = ImageOps.autocontrast(crop.convert("L")).resize(
                    (crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC
                )
                retry_image = ImageEnhance.Contrast(retry_image).enhance(1.2)
                retry = self.recognizer.recognize(retry_image.convert("RGB"))
                if retry.confidence > recognized.confidence:
                    recognized = retry
            if recognized.text and contains_hangul(recognized.text):
                lines.append(make_line(recognized.text, region.box, recognized.confidence))
        return OcrDocument(tuple(lines), time.monotonic(), origin[0], origin[1])
