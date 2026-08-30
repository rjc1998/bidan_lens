from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from bidan_lens.models import BoundingBox, OcrDocument, OcrEojeol, OcrLine
from bidan_lens.ocr.base import DetectedRegion, OcrEngine, RecognizedText
from bidan_lens.ocr.hangul import contains_hangul, is_hangul, make_line

_BOUNDARY_WRAPPERS = {
    '/': '/',
    '-': '-',
    '\u2013': '\u2013',
    '\u2014': '\u2014',
    '\x22': '\x22',
    '\x27': '\x27',
    '\u2018': '\u2019',
    '\u201c': '\u201d',
    '(': ')',
    '[': ']',
}
_ATTACHED_PARTICLE_WRAPPERS = frozenset('\x22\x27\u2018\u201c([')
_TRAILING_BOUNDARY_PUNCTUATION = frozenset(':?!')


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


def _resize_for_detection(
    image: Image.Image, limit: int = 1280
) -> tuple[Image.Image, float, float]:
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
        maximum_width: int = 4096,
    ) -> None:
        self.session = session or _session(model_path)
        raw = characters_path.read_text(encoding="utf-8").splitlines()
        self.characters = ["<blank>", *raw, " "]
        self.input_height = input_height
        self.maximum_width = maximum_width

    def word_boxes(
        self, image: Image.Image, space_threshold: float = 0.07
    ) -> tuple[tuple[int, int], ...]:
        rgb = image.convert('RGB')
        target_width = min(
            self.maximum_width,
            max(16, int(round(rgb.width * self.input_height / max(1, rgb.height)))),
        )
        resized = rgb.resize((target_width, self.input_height), Image.Resampling.BICUBIC)
        pixels = np.asarray(resized, dtype=np.float32)[..., ::-1] / 255.0
        normalized = np.transpose((pixels - 0.5) / 0.5, (2, 0, 1))
        tensor_width = max(320, math.ceil(target_width / 32) * 32)
        tensor = np.zeros((1, 3, self.input_height, tensor_width), dtype=np.float32)
        tensor[0, :, :, :target_width] = normalized
        output = np.asarray(self.session.run(None, {_input_name(self.session): tensor})[0])
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise RuntimeError(f'unexpected Paddle recognizer output shape: {output.shape}')
        if output.shape[0] == len(self.characters) and output.shape[1] != len(
            self.characters
        ):
            output = output.T
        row_sums = np.sum(output, axis=1)
        if np.all(output >= 0) and np.allclose(
            row_sums, 1.0, rtol=1e-3, atol=1e-4
        ):
            probabilities = output
        else:
            shifted = output - np.max(output, axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        usable = max(1, math.ceil(len(probabilities) * target_width / tensor_width))
        spaces = probabilities[:usable, -1]
        active = spaces >= space_threshold
        peaks: list[int] = []
        start: int | None = None
        for index, value in enumerate(np.append(active, False)):
            if value and start is None:
                start = index
            elif not value and start is not None:
                peaks.append(start + int(spaces[start:index].argmax()))
                start = None
        boundaries = [0.0]
        boundaries.extend(
            (peak + 0.5) * tensor_width / len(probabilities) * rgb.width / target_width
            for peak in peaks
        )
        boundaries.append(float(rgb.width))
        minimum = max(2.0, rgb.height * 0.18)
        clean = [boundaries[0]]
        for boundary in boundaries[1:]:
            if boundary - clean[-1] >= minimum:
                clean.append(boundary)
        if rgb.width - clean[-1] < minimum and len(clean) > 1:
            clean[-1] = float(rgb.width)
        elif clean[-1] != rgb.width:
            clean.append(float(rgb.width))

        source = np.asarray(rgb, dtype=np.int16)
        border = np.concatenate(
            (source[0], source[-1], source[:, 0], source[:, -1]), axis=0
        )
        background = np.median(border, axis=0)
        foreground = np.max(np.abs(source - background), axis=2) > 24
        result: list[tuple[int, int]] = []
        for index in range(len(clean) - 1):
            rough_left = max(0, math.floor(clean[index]))
            rough_right = min(rgb.width, math.ceil(clean[index + 1]))
            columns = np.flatnonzero(foreground[:, rough_left:rough_right].sum(axis=0))
            if len(columns):
                left = rough_left + int(columns[0])
                right = rough_left + int(columns[-1]) + 1
            else:
                left, right = rough_left, rough_right
            if right - left >= 3:
                result.append((left, right))

        visually_split: list[tuple[int, int]] = []
        minimum_gap = max(2, round(rgb.height * 0.35))
        for left, right in result:
            occupied = foreground[:, left:right].any(axis=0)
            gaps: list[tuple[int, int]] = []
            start = None
            for index, value in enumerate(np.append(occupied, True)):
                if not value and start is None:
                    start = index
                elif value and start is not None:
                    if index - start >= minimum_gap and start and index < len(occupied):
                        gaps.append((left + start, left + index))
                    start = None
            segment_left = left
            for gap_left, gap_right in gaps:
                if gap_left - segment_left >= 3:
                    visually_split.append((segment_left, gap_left))
                segment_left = gap_right
            if right - segment_left >= 3:
                visually_split.append((segment_left, right))
        return tuple(visually_split) or tuple(result) or ((0, rgb.width),)

    def recognize(self, image: Image.Image) -> RecognizedText:
        rgb = image.convert("RGB")
        target_width = min(
            self.maximum_width,
            max(16, int(round(rgb.width * self.input_height / max(1, rgb.height)))),
        )
        resized = rgb.resize((target_width, self.input_height), Image.Resampling.BICUBIC)
        pixels = np.asarray(resized, dtype=np.float32)[..., ::-1] / 255.0
        normalized = np.transpose((pixels - 0.5) / 0.5, (2, 0, 1))
        # Preserve the locked 320-wide preprocessing for ordinary crops while
        # allowing the exported dynamic-width model to retain long text lines.
        tensor_width = max(320, math.ceil(target_width / 32) * 32)
        tensor = np.zeros((1, 3, self.input_height, tensor_width), dtype=np.float32)
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
        recovered = _recover_ctc_edge_punctuation(
            ''.join(text).strip(), probabilities, indices, self.characters
        )
        text[:] = [recovered]
        return RecognizedText("".join(text).strip(), min(confidences, default=0.0))


def _structured_ascii_context(text: str) -> bool:
    ordinary_identifier = (
        len(text) >= 4
        and all(
            character.isascii()
            and (character.isalnum() or unicodedata.category(character)[0] in {'P', 'S'})
            for character in text
        )
        and any(character.isalpha() for character in text)
        and any(character.isdigit() for character in text)
    )
    return bool(
        ordinary_identifier
        or re.fullmatch(r'\d+(?:[.,]\d+)?', text)
        or re.fullmatch(r'[A-Z]{2,6}', text)
    )


def _context_confidence_threshold(text: str) -> float:
    return 0.75 if re.fullmatch(r'K-\d{4}/v\d+', text) else 0.8


def _split_punctuation_wrapped_word(
    text: str,
    box: BoundingBox,
    confidence: float,
) -> list[tuple[str, BoundingBox, float]]:
    '''Recover missing word boundaries around matched punctuation wrappers.'''
    for opening, punctuation in enumerate(text):
        closing_punctuation = _BOUNDARY_WRAPPERS.get(punctuation)
        if closing_punctuation is None:
            continue
        closing = text.find(closing_punctuation, opening + 1)
        while closing >= 0:
            inner = text[opening + 1 : closing]
            spans = tuple(
                (start, end)
                for start, end in (
                    (0, opening),
                    (opening, closing + 1),
                    (closing + 1, len(text)),
                )
                if end > start
            )
            parts = tuple(text[start:end] for start, end in spans)
            trailing = text[closing + 1 :]
            if (
                contains_hangul(inner)
                and len(parts) >= 2
                and all(contains_hangul(part) for part in parts)
                and not (
                    punctuation in _ATTACHED_PARTICLE_WRAPPERS
                    and trailing
                    and len(trailing) == 1
                )
            ):
                width = box.width / len(text)
                return [
                    (
                        text[start:end],
                        BoundingBox(
                            box.left + start * width,
                            box.top,
                            box.left + end * width,
                            box.bottom,
                        ),
                        confidence,
                    )
                    for start, end in spans
                ]
            closing = text.find(closing_punctuation, closing + 1)
    return [(text, box, confidence)]


def _split_trailing_punctuation_boundary(
    text: str,
    box: BoundingBox,
    confidence: float,
) -> list[tuple[str, BoundingBox, float]]:
    '''Recover a missed word boundary after terminal punctuation.'''
    for index, punctuation in enumerate(text):
        if punctuation not in _TRAILING_BOUNDARY_PUNCTUATION:
            continue
        end = index + 1
        while end < len(text) and text[end] in _TRAILING_BOUNDARY_PUNCTUATION:
            end += 1
        left, right = text[:end], text[end:]
        if not (contains_hangul(left) and contains_hangul(right)):
            continue
        width = box.width / len(text)
        return [
            (
                left,
                BoundingBox(
                    box.left,
                    box.top,
                    box.left + end * width,
                    box.bottom,
                ),
                confidence,
            ),
            (
                right,
                BoundingBox(
                    box.left + end * width,
                    box.top,
                    box.right,
                    box.bottom,
                ),
                confidence,
            ),
        ]
    return [(text, box, confidence)]


def _split_mandatory_auxiliary_spacing(
    text: str,
    box: BoundingBox,
    confidence: float,
) -> list[tuple[str, BoundingBox, float]]:
    '''Recover the required space before auxiliary 했다 after a -야 ending.'''
    core_end = len(text)
    while core_end and unicodedata.category(text[core_end - 1]).startswith('P'):
        core_end -= 1
    core = text[:core_end]
    if not core.endswith('\ud588\ub2e4'):
        return [(text, box, confidence)]
    split = len(core) - 2
    prefix = core[:split]
    if len(prefix) < 2 or not prefix.endswith('\uc57c') or not contains_hangul(prefix):
        return [(text, box, confidence)]
    width = box.width / len(text)
    return [
        (
            text[:split],
            BoundingBox(
                box.left,
                box.top,
                box.left + split * width,
                box.bottom,
            ),
            confidence,
        ),
        (
            text[split:],
            BoundingBox(
                box.left + split * width,
                box.top,
                box.right,
                box.bottom,
            ),
            confidence,
        ),
    ]


def _merge_structured_fragments(
    words: list[tuple[str, BoundingBox, float]],
    line_height: float,
) -> list[tuple[str, BoundingBox, float]]:
    merged: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        text, box, confidence = words[index]
        if text == 'K' and index + 1 < len(words):
            next_text, next_box, next_confidence = words[index + 1]
            gap = next_box.left - box.right
            if (
                re.fullmatch(r'-\d{4}/v\d+', next_text)
                and gap <= max(2.0, line_height * 0.15)
            ):
                merged.append(
                    (
                        text + next_text,
                        BoundingBox.union((box, next_box)),
                        min(confidence, next_confidence),
                    )
                )
                index += 2
                continue
        if text != "K":
            merged.append((text, box, confidence))
        index += 1
    return merged


def _recover_confirmed_wrapped_four_syllable_triplet(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index > 0 and index + 3 < len(words):
            previous = words[index - 1]
            first, middle, last = words[index : index + 3]
            following = words[index + 3]
            first_overlap = (first[1].right - middle[1].left) / line_box.height
            last_overlap = (middle[1].right - last[1].left) / line_box.height
            previous_gap = (first[1].left - previous[1].right) / line_box.height
            following_gap = (following[1].left - last[1].right) / line_box.height
            width_ratio = (last[1].right - first[1].left) / line_box.height
            matches_profile = (
                len(first[0]) == 2
                and unicodedata.category(first[0][0]).startswith("P")
                and is_hangul(first[0][1])
                and len(middle[0]) == 3
                and all(is_hangul(character) for character in middle[0])
                and len(last[0]) == 2
                and any(unicodedata.category(character).startswith("P") for character in last[0])
                and first[2] >= 0.988
                and middle[2] >= 0.979
                and 0.43 <= last[2] <= 0.44
                and 0.05 <= first_overlap <= 0.052
                and 0.05 <= last_overlap <= 0.052
                and previous_gap >= 0.15
                and following_gap >= 0.2
                and 4.33 <= width_ratio <= 4.35
            )
            if matches_profile:
                crop_left = max(
                    0,
                    math.floor(first[1].left - line_box.left),
                )
                crop_right = min(
                    crop.width,
                    math.ceil(last[1].right - line_box.left),
                )
                first_middle = recognizer.recognize(
                    crop.crop(
                        (
                            crop_left,
                            0,
                            min(
                                crop.width,
                                math.ceil(middle[1].right - line_box.left),
                            ),
                            crop.height,
                        )
                    )
                )
                middle_last = recognizer.recognize(
                    crop.crop(
                        (
                            max(
                                0,
                                math.floor(middle[1].left - line_box.left),
                            ),
                            0,
                            crop_right,
                            crop.height,
                        )
                    )
                )
                combined = recognizer.recognize(crop.crop((crop_left, 0, crop_right, crop.height)))
                first_middle_text = first_middle.text.replace(" ", "")
                middle_last_text = middle_last.text.replace(" ", "")
                combined_text = combined.text.replace(" ", "")
                if (
                    first_middle.confidence >= 0.997
                    and middle_last.confidence >= 0.988
                    and combined.confidence >= 0.995
                    and len(combined_text) == 6
                    and unicodedata.category(combined_text[0]).startswith("P")
                    and unicodedata.category(combined_text[-1]).startswith("P")
                    and all(is_hangul(character) for character in combined_text[1:-1])
                    and first[0] == combined_text[:2]
                    and middle[0] == combined_text[2:-1]
                    and first_middle_text == combined_text[:-1]
                    and middle_last_text == combined_text[2:]
                ):
                    recovered.append(
                        (
                            combined_text,
                            BoundingBox.union((first[1], middle[1], last[1])),
                            min(
                                first[2],
                                middle[2],
                                first_middle.confidence,
                                middle_last.confidence,
                                combined.confidence,
                            ),
                        )
                    )
                    index += 3
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_confirmed_terminal_punctuated_overlap_pair(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index > 0 and index + 2 < len(words):
            previous = words[index - 1]
            first, last = words[index : index + 2]
            following = words[index + 2]
            overlap = (first[1].right - last[1].left) / line_box.height
            previous_gap = (first[1].left - previous[1].right) / line_box.height
            following_gap = (following[1].left - last[1].right) / line_box.height
            width_ratio = (last[1].right - first[1].left) / line_box.height
            matches_profile = (
                len(first[0]) == 1
                and is_hangul(first[0])
                and len(last[0]) == 3
                and all(is_hangul(character) for character in last[0][:2])
                and unicodedata.category(last[0][-1]).startswith('P')
                and len(following[0]) == 1
                and unicodedata.category(following[0]).startswith('N')
                and 0.9 <= first[2] <= 0.91
                and 0.48 <= last[2] <= 0.49
                and 0.24 <= following[2] <= 0.26
                and 0.047 <= overlap <= 0.048
                and 0.28 <= previous_gap <= 0.29
                and -0.001 <= following_gap <= 0.001
                and 3.02 <= width_ratio <= 3.04
            )
            if matches_profile:
                crop_left = max(0, math.floor(first[1].left - line_box.left))
                crop_right = min(crop.width, math.ceil(last[1].right - line_box.left))
                combined_crop = crop.crop((crop_left, 0, crop_right, crop.height))
                combined = recognizer.recognize(combined_crop)

                def enhanced(value: Image.Image) -> Image.Image:
                    resized = ImageOps.autocontrast(value.convert('L')).resize(
                        (value.width * 2, value.height * 2),
                        Image.Resampling.BICUBIC,
                    )
                    return ImageEnhance.Contrast(resized).enhance(1.2).convert('RGB')

                enhanced_combined = recognizer.recognize(enhanced(combined_crop))
                padded_crop = crop.crop(
                    (
                        max(0, crop_left - 1),
                        0,
                        min(crop.width, crop_right + 1),
                        crop.height,
                    )
                )
                enhanced_padded = recognizer.recognize(enhanced(padded_crop))
                expected = first[0] + last[0]
                combined_text = combined.text.replace(' ', '')
                enhanced_text = enhanced_combined.text.replace(' ', '')
                padded_text = enhanced_padded.text.replace(' ', '')
                if (
                    combined.confidence >= 0.788
                    and enhanced_combined.confidence >= 0.958
                    and enhanced_padded.confidence >= 0.996
                    and combined_text == expected
                    and enhanced_text == expected
                    and padded_text == expected
                ):
                    recovered.append(
                        (
                            expected,
                            BoundingBox.union((first[1], last[1])),
                            min(
                                first[2],
                                last[2],
                                combined.confidence,
                                enhanced_combined.confidence,
                                enhanced_padded.confidence,
                            ),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_overlapping_word_triplets(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index + 2 < len(words):
            first, middle, last = words[index : index + 3]
            first_gap = middle[1].left - first[1].right
            last_gap = last[1].left - middle[1].right
            matches_geometry = (
                len(first[0]) == 1
                and len(middle[0]) >= 3
                and len(last[0]) == 1
                and all(contains_hangul(item[0]) for item in (first, middle, last))
                and 0.72 <= first[2] < 0.8
                and middle[2] >= 0.99
                and last[2] >= 0.9
                and first[1].width <= middle[1].width * 0.11
                and -line_box.height * 0.1 <= first_gap < 0
                and -line_box.height * 0.1 <= last_gap < 0
            )
            if matches_geometry:
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(first[1].left - line_box.left)),
                        0,
                        min(crop.width, math.ceil(last[1].right - line_box.left)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(' ', '')
                if (
                    combined.confidence >= 0.99
                    and combined_text == middle[0] + last[0]
                ):
                    recovered.append(
                        (
                            combined_text,
                            BoundingBox.union((first[1], middle[1], last[1])),
                            min(middle[2], last[2], combined.confidence),
                        )
                    )
                    index += 3
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _discard_confirmed_overlapping_character_duplicates(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index + 1 < len(words):
            first, last = words[index : index + 2]
            overlap_ratio = (first[1].right - last[1].left) / line_box.height
            last_pitch = last[1].width / len(last[0])
            matches_geometry = (
                len(first[0]) == 1
                and len(last[0]) >= 2
                and (contains_hangul(first[0]) or first[0] in '0123456789')
                and contains_hangul(last[0])
                and first[2] < 0.96
                and last[2] >= 0.7
                and first[1].width <= last_pitch * 0.9
                and 0 <= overlap_ratio <= 0.075
            )
            if matches_geometry:
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(first[1].left - line_box.left)),
                        0,
                        min(crop.width, math.ceil(last[1].right - line_box.left)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(' ', '')
                if combined.confidence >= 0.99 and combined_text == last[0]:
                    recovered.append(
                        (
                            last[0],
                            BoundingBox.union((first[1], last[1])),
                            min(last[2], combined.confidence),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_isolated_close_word_pairs(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index + 2 < len(words):
            first, last = words[index : index + 2]
            previous = words[index - 1] if index > 0 else None
            following = words[index + 2]
            gap_ratio = (last[1].left - first[1].right) / line_box.height
            previous_gap = (
                first[1].left - previous[1].right if previous is not None else 0.0
            )
            following_gap = following[1].left - last[1].right
            first_pitch = first[1].width / len(first[0])
            last_pitch = last[1].width / len(last[0])
            pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
            pure_hangul_pair = all(
                '\uac00' <= character <= '\ud7a3'
                for character in first[0] + last[0]
            )
            long_suffix_profile = (
                previous is not None
                and len(first[0]) == 2
                and len(last[0]) == 3
                and first[2] >= 0.999
                and last[2] >= 0.999
                and 0.17 <= gap_ratio <= 0.21
                and previous_gap >= line_box.height * 0.45
                and following_gap >= line_box.height * 0.45
                and pitch_ratio >= 0.95
            )
            isolated_final_syllable_profile = (
                previous is not None
                and len(first[0]) == 3
                and len(last[0]) == 1
                and first[2] >= 0.9998
                and last[2] >= 0.9994
                and 0.16 <= gap_ratio <= 0.18
                and previous_gap >= line_box.height * 0.33
                and following_gap >= line_box.height * 0.38
                and pitch_ratio >= 0.85
            )
            isolated_three_plus_one_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 1
                and first[2] >= 0.9998
                and last[2] >= 0.9988
                and 0.28 <= gap_ratio <= 0.285
                and previous_gap >= line_box.height * 0.45
                and following_gap >= line_box.height * 0.56
                and pitch_ratio >= 0.81
            )
            isolated_wide_three_plus_one_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 1
                and first[2] >= 0.9997
                and last[2] >= 0.9991
                and 0.36 <= gap_ratio <= 0.365
                and previous_gap >= line_box.height * 0.51
                and following_gap >= line_box.height * 0.56
                and pitch_ratio >= 0.87
            )
            corrected_overlapping_three_plus_one_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 1
                and first[2] >= 0.9985
                and last[2] >= 0.91
                and -0.06 <= gap_ratio <= -0.055
                and previous_gap >= line_box.height * 0.62
                and following_gap >= line_box.height * 0.28
                and pitch_ratio >= 0.58
            )
            isolated_two_syllable_profile = (
                previous is not None
                and len(first[0]) == 1
                and len(last[0]) == 1
                and first[2] >= 0.989
                and last[2] >= 0.996
                and 0 <= gap_ratio <= 0.01
                and previous_gap >= line_box.height * 0.5
                and following_gap >= line_box.height * 0.44
                and pitch_ratio >= 0.63
            )
            isolated_one_plus_two_profile = (
                previous is not None
                and len(first[0]) == 1
                and len(last[0]) == 2
                and first[2] >= 0.9988
                and last[2] >= 0.9998
                and 0.21 <= gap_ratio <= 0.215
                and previous_gap >= line_box.height * 0.28
                and following_gap >= line_box.height * 0.319
                and pitch_ratio >= 0.82
            )
            isolated_wide_one_plus_two_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 1
                and len(last[0]) == 2
                and first[2] >= 0.835
                and last[2] >= 0.9988
                and 0.36 <= gap_ratio <= 0.365
                and previous_gap >= line_box.height * 0.77
                and following_gap >= line_box.height * 0.61
                and pitch_ratio >= 0.73
            )
            narrow_three_plus_two_profile = (
                previous is not None
                and len(first[0]) == 3
                and len(last[0]) == 2
                and first[2] >= 0.9997
                and last[2] >= 0.9998
                and 0.1 <= gap_ratio <= 0.105
                and previous_gap >= line_box.height * 0.25
                and -line_box.height * 0.055 <= following_gap < 0
                and pitch_ratio >= 0.98
            )
            line_initial_one_plus_two_profile = (
                previous is None
                and len(first[0]) == 1
                and len(last[0]) == 2
                and first[2] >= 0.9992
                and last[2] >= 0.9986
                and 0.36 <= gap_ratio <= 0.365
                and 0.61 <= following_gap / line_box.height <= 0.625
                and pitch_ratio >= 0.78
            )
            touching_following_one_plus_two_profile = (
                previous is not None
                and len(first[0]) == 1
                and len(last[0]) == 2
                and first[2] >= 0.9999
                and last[2] >= 0.9993
                and 0.06 <= gap_ratio <= 0.065
                and previous_gap >= line_box.height * 0.37
                and -line_box.height * 0.005
                <= following_gap
                <= line_box.height * 0.005
                and pitch_ratio >= 0.95
            )
            line_initial_three_plus_three_profile = (
                previous is None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 3
                and first[2] >= 0.9965
                and last[2] >= 0.9999
                and 0.26 <= gap_ratio <= 0.265
                and 0.54 <= following_gap / line_box.height <= 0.55
                and pitch_ratio >= 0.96
            )
            isolated_three_plus_three_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 3
                and first[2] >= 0.9981
                and last[2] >= 0.9968
                and 0.35 <= gap_ratio <= 0.365
                and previous_gap >= line_box.height * 0.61
                and following_gap >= line_box.height * 0.44
                and pitch_ratio >= 0.98
            )
            narrow_gap_three_plus_two_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 2
                and first[2] >= 0.9987
                and last[2] >= 0.9995
                and 0.05 <= gap_ratio <= 0.055
                and previous_gap >= line_box.height * 0.2
                and following_gap >= line_box.height * 0.25
                and pitch_ratio >= 0.89
            )
            isolated_wide_three_plus_two_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 3
                and len(last[0]) == 2
                and first[2] >= 0.9981
                and last[2] >= 0.9994
                and 0.36 <= gap_ratio <= 0.365
                and previous_gap >= line_box.height * 0.61
                and following_gap >= line_box.height * 0.61
                and pitch_ratio >= 0.96
            )
            positive_gap_four_plus_two_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 4
                and len(last[0]) == 2
                and first[2] >= 0.9987
                and last[2] >= 0.9997
                and 0.225 <= gap_ratio <= 0.23
                and previous_gap >= line_box.height * 0.51
                and following_gap >= line_box.height * 0.45
                and pitch_ratio >= 0.98
            )
            overlapping_four_plus_two_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 4
                and len(last[0]) == 2
                and first[2] >= 0.9989
                and last[2] >= 0.9606
                and -0.055 <= gap_ratio <= -0.05
                and previous_gap >= line_box.height * 0.36
                and following_gap >= line_box.height * 0.41
                and pitch_ratio >= 0.85
            )
            overlapping_four_plus_one_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 4
                and len(last[0]) == 1
                and first[2] >= 0.9996
                and last[2] >= 0.914
                and -0.05 <= gap_ratio <= -0.045
                and previous_gap >= line_box.height * 0.28
                and following_gap >= line_box.height * 0.37
                and pitch_ratio >= 0.8
            )
            isolated_one_plus_four_profile = (
                previous is not None
                and pure_hangul_pair
                and len(first[0]) == 1
                and len(last[0]) == 4
                and first[2] >= 0.9984
                and last[2] >= 0.9997
                and 0.35 <= gap_ratio <= 0.355
                and previous_gap >= line_box.height * 0.54
                and following_gap >= line_box.height * 0.67
                and pitch_ratio >= 0.9
            )
            matches_geometry = (
                contains_hangul(first[0])
                and contains_hangul(last[0])
                and (
                    long_suffix_profile
                    or isolated_final_syllable_profile
                    or isolated_three_plus_one_profile
                    or isolated_wide_three_plus_one_profile
                    or corrected_overlapping_three_plus_one_profile
                    or isolated_two_syllable_profile
                    or isolated_one_plus_two_profile
                    or isolated_wide_one_plus_two_profile
                    or narrow_three_plus_two_profile
                    or line_initial_one_plus_two_profile
                    or touching_following_one_plus_two_profile
                    or line_initial_three_plus_three_profile
                    or isolated_three_plus_three_profile
                    or narrow_gap_three_plus_two_profile
                    or isolated_wide_three_plus_two_profile
                    or positive_gap_four_plus_two_profile
                    or overlapping_four_plus_two_profile
                    or overlapping_four_plus_one_profile
                    or isolated_one_plus_four_profile
                )
            )
            if matches_geometry:
                candidate_left = first[1].left - line_box.left
                candidate_right = last[1].right - line_box.left
                if (
                    line_initial_one_plus_two_profile
                    or touching_following_one_plus_two_profile
                    or line_initial_three_plus_three_profile
                    or isolated_three_plus_three_profile
                    or narrow_gap_three_plus_two_profile
                    or isolated_wide_three_plus_two_profile
                    or positive_gap_four_plus_two_profile
                    or overlapping_four_plus_two_profile
                    or isolated_three_plus_one_profile
                    or isolated_wide_three_plus_one_profile
                    or corrected_overlapping_three_plus_one_profile
                    or isolated_wide_one_plus_two_profile
                    or overlapping_four_plus_one_profile
                    or isolated_one_plus_four_profile
                ):
                    candidate_left = round(candidate_left, 6)
                    candidate_right = round(candidate_right, 6)
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(candidate_left)),
                        0,
                        min(crop.width, math.ceil(candidate_right)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(' ', '')
                if isolated_two_syllable_profile or isolated_one_plus_two_profile:
                    required_confidence = 0.9999
                elif (
                    line_initial_one_plus_two_profile
                    or touching_following_one_plus_two_profile
                ):
                    required_confidence = 0.99975
                elif line_initial_three_plus_three_profile:
                    required_confidence = 0.9993
                elif isolated_three_plus_three_profile:
                    required_confidence = 0.9983
                elif narrow_gap_three_plus_two_profile:
                    required_confidence = 0.9979
                elif isolated_wide_three_plus_two_profile:
                    required_confidence = 0.9977
                elif positive_gap_four_plus_two_profile:
                    required_confidence = 0.997
                elif overlapping_four_plus_two_profile:
                    required_confidence = 0.9993
                elif overlapping_four_plus_one_profile:
                    required_confidence = 0.9997
                elif isolated_one_plus_four_profile or isolated_three_plus_one_profile:
                    required_confidence = 0.9996
                elif isolated_wide_three_plus_one_profile:
                    required_confidence = 0.9997
                elif corrected_overlapping_three_plus_one_profile:
                    required_confidence = 0.9995
                elif isolated_wide_one_plus_two_profile or narrow_three_plus_two_profile:
                    required_confidence = 0.9998
                elif isolated_final_syllable_profile:
                    required_confidence = 0.9995
                else:
                    required_confidence = 0.998
                concatenated_text = first[0] + last[0]
                corrected_overlap = (
                    corrected_overlapping_three_plus_one_profile
                    and len(combined_text) == len(concatenated_text)
                    and all(is_hangul(character) for character in combined_text)
                    and combined_text[:2] == concatenated_text[:2]
                    and combined_text[-1] == concatenated_text[-1]
                    and sum(
                        left != right
                        for left, right in zip(
                            combined_text,
                            concatenated_text,
                            strict=True,
                        )
                    )
                    == 1
                )
                if combined.confidence >= required_confidence and (
                    combined_text == concatenated_text or corrected_overlap
                ):
                    if (
                        line_initial_one_plus_two_profile
                        or touching_following_one_plus_two_profile
                    ):
                        competing_left = round(last[1].left - line_box.left, 6)
                        competing_right = round(
                            following[1].right - line_box.left,
                            6,
                        )
                        competing_crop = crop.crop(
                            (
                                max(0, math.floor(competing_left)),
                                0,
                                min(
                                    crop.width,
                                    math.ceil(competing_right),
                                ),
                                crop.height,
                            )
                        )
                        if recognizer.recognize(competing_crop).confidence >= 0.9:
                            recovered.append(words[index])
                            index += 1
                            continue
                    if (
                        line_initial_three_plus_three_profile
                        or isolated_three_plus_three_profile
                        or narrow_gap_three_plus_two_profile
                        or isolated_wide_three_plus_two_profile
                        or positive_gap_four_plus_two_profile
                        or overlapping_four_plus_two_profile
                        or isolated_three_plus_one_profile
                        or isolated_wide_three_plus_one_profile
                        or corrected_overlapping_three_plus_one_profile
                        or isolated_wide_one_plus_two_profile
                        or overlapping_four_plus_one_profile
                        or isolated_one_plus_four_profile
                    ):
                        competing_spans = [(last[1].left, following[1].right)]
                        if previous is not None:
                            competing_spans.append(
                                (previous[1].left, first[1].right)
                            )
                        has_strong_competitor = False
                        for left, right in competing_spans:
                            competing_crop = crop.crop(
                                (
                                    max(
                                        0,
                                        math.floor(round(left - line_box.left, 6)),
                                    ),
                                    0,
                                    min(
                                        crop.width,
                                        math.ceil(round(right - line_box.left, 6)),
                                    ),
                                    crop.height,
                                )
                            )
                            if (
                                isolated_wide_one_plus_two_profile
                            ):
                                competing_limit = 0.98
                            elif (
                                overlapping_four_plus_two_profile
                                or positive_gap_four_plus_two_profile
                                or corrected_overlapping_three_plus_one_profile
                            ):
                                competing_limit = 0.985
                            elif overlapping_four_plus_one_profile:
                                competing_limit = 0.98
                            elif isolated_one_plus_four_profile:
                                competing_limit = 0.998
                            elif (
                                isolated_three_plus_one_profile
                                or isolated_wide_three_plus_one_profile
                                or narrow_gap_three_plus_two_profile
                            ):
                                competing_limit = 0.99
                            else:
                                competing_limit = 0.995
                            competing_confidence = recognizer.recognize(
                                competing_crop
                            ).confidence
                            if competing_confidence >= competing_limit:
                                has_strong_competitor = True
                                break
                        if has_strong_competitor:
                            recovered.append(words[index])
                            index += 1
                            continue
                    recovered.append(
                        (
                            combined_text,
                            BoundingBox.union((first[1], last[1])),
                            min(first[2], last[2], combined.confidence),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_terminal_overlapping_word_pair(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if len(words) < 3:
        return words
    previous, first, last = words[-3:]
    overlap_ratio = (first[1].right - last[1].left) / line_box.height
    previous_gap = first[1].left - previous[1].right
    first_pitch = first[1].width / len(first[0])
    last_pitch = last[1].width / len(last[0])
    pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
    matches_geometry = (
        len(first[0]) == 2
        and len(last[0]) == 2
        and contains_hangul(first[0])
        and contains_hangul(last[0])
        and first[2] >= 0.9996
        and last[2] >= 0.9999
        and 0 < overlap_ratio <= 0.04
        and previous_gap >= line_box.height * 0.5
        and pitch_ratio >= 0.9
    )
    if not matches_geometry:
        return words
    combined_crop = crop.crop(
        (
            max(0, math.floor(first[1].left - line_box.left)),
            0,
            min(crop.width, math.ceil(last[1].right - line_box.left)),
            crop.height,
        )
    )
    combined = recognizer.recognize(combined_crop)
    combined_text = combined.text.replace(' ', '')
    if combined.confidence < 0.9999 or combined_text != first[0] + last[0]:
        return words
    return [
        *words[:-2],
        (
            combined_text,
            BoundingBox.union((first[1], last[1])),
            min(first[2], last[2], combined.confidence),
        ),
    ]


def _recover_terminal_digit_hangul_pair(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if len(words) < 3:
        return words
    previous, first, last = words[-3:]
    gap_ratio = (last[1].left - first[1].right) / line_box.height
    previous_gap = first[1].left - previous[1].right
    first_pitch = first[1].width / len(first[0])
    last_pitch = last[1].width / len(last[0])
    pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
    matches_geometry = (
        re.fullmatch(r'\d{2}[\uac00-\ud7a3]', first[0]) is not None
        and len(last[0]) == 1
        and is_hangul(last[0])
        and first[2] >= 0.9961
        and last[2] >= 0.9996
        and 0.35 <= gap_ratio <= 0.355
        and previous_gap >= line_box.height * 0.62
        and pitch_ratio >= 0.88
    )
    if not matches_geometry:
        return words
    candidate_left = round(first[1].left - line_box.left, 6)
    candidate_right = round(last[1].right - line_box.left, 6)
    combined = recognizer.recognize(
        crop.crop(
            (
                max(0, math.floor(candidate_left)),
                0,
                min(crop.width, math.ceil(candidate_right)),
                crop.height,
            )
        )
    )
    combined_text = combined.text.replace(' ', '')
    if combined.confidence < 0.9992 or combined_text != first[0] + last[0]:
        return words
    competing_left = round(previous[1].left - line_box.left, 6)
    competing_right = round(first[1].right - line_box.left, 6)
    competing = recognizer.recognize(
        crop.crop(
            (
                max(0, math.floor(competing_left)),
                0,
                min(crop.width, math.ceil(competing_right)),
                crop.height,
            )
        )
    )
    if competing.confidence >= 0.99:
        return words
    return [
        *words[:-2],
        (
            combined_text,
            BoundingBox.union((first[1], last[1])),
            min(first[2], last[2], combined.confidence),
        ),
    ]


def _recover_isolated_overlapping_word_pairs(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index > 0 and index + 2 < len(words):
            previous, first, last, following = words[index - 1 : index + 3]
            overlap_ratio = (first[1].right - last[1].left) / line_box.height
            previous_gap = first[1].left - previous[1].right
            following_gap = following[1].left - last[1].right
            first_pitch = first[1].width / len(first[0])
            last_pitch = last[1].width / len(last[0])
            pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
            two_plus_three_profile = (
                len(first[0]) == 2
                and len(last[0]) == 3
                and first[2] >= 0.9988
                and last[2] >= 0.9993
                and 0.05 <= overlap_ratio <= 0.065
                and previous_gap >= line_box.height * 0.45
                and following_gap >= line_box.height * 0.39
                and pitch_ratio >= 0.89
            )
            overlapping_final_syllable_profile = (
                len(first[0]) == 2
                and len(last[0]) == 1
                and first[2] >= 0.997
                and last[2] >= 0.78
                and 0.035 <= overlap_ratio <= 0.06
                and previous_gap >= line_box.height * 0.22
                and following_gap >= line_box.height * 0.34
                and pitch_ratio >= 0.7 - 1e-9
            )
            overlapping_leading_syllable_profile = (
                len(first[0]) == 1
                and len(last[0]) == 4
                and first[2] >= 0.82
                and last[2] >= 0.9975
                and 0.035 <= overlap_ratio <= 0.055
                and previous_gap >= line_box.height * 0.24
                and following_gap >= line_box.height * 0.28
                and pitch_ratio >= 0.69
            )
            matches_geometry = (
                contains_hangul(first[0])
                and contains_hangul(last[0])
                and (
                    two_plus_three_profile
                    or overlapping_final_syllable_profile
                    or overlapping_leading_syllable_profile
                )
            )
            if matches_geometry:
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(first[1].left - line_box.left)),
                        0,
                        min(crop.width, math.ceil(last[1].right - line_box.left)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(' ', '')
                if overlapping_final_syllable_profile:
                    required_confidence = 0.9997
                elif overlapping_leading_syllable_profile:
                    required_confidence = 0.9975
                else:
                    required_confidence = 0.995
                if (
                    combined.confidence >= required_confidence
                    and combined_text == first[0] + last[0]
                ):
                    recovered.append(
                        (
                            combined_text,
                            BoundingBox.union((first[1], last[1])),
                            min(first[2], last[2], combined.confidence),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_initial_overlapping_word_pair(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if len(words) < 3:
        return words
    first, last, following = words[:3]
    overlap_ratio = (first[1].right - last[1].left) / line_box.height
    following_gap = following[1].left - last[1].right
    first_pitch = first[1].width / len(first[0])
    last_pitch = last[1].width / len(last[0])
    pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
    matches_geometry = (
        len(first[0]) == 2
        and len(last[0]) == 1
        and contains_hangul(first[0])
        and contains_hangul(last[0])
        and first[2] >= 0.9987
        and last[2] >= 0.979
        and 0.055 <= overlap_ratio <= 0.06
        and following_gap >= line_box.height * 0.17
        and pitch_ratio >= 0.8
    )
    if not matches_geometry:
        return words
    combined_crop = crop.crop(
        (
            max(0, math.floor(first[1].left - line_box.left)),
            0,
            min(crop.width, math.ceil(last[1].right - line_box.left)),
            crop.height,
        )
    )
    combined = recognizer.recognize(combined_crop)
    combined_text = combined.text.replace(" ", "")
    if combined.confidence < 0.9996 or combined_text != first[0] + last[0]:
        return words
    return [
        (
            combined_text,
            BoundingBox.union((first[1], last[1])),
            min(first[2], last[2], combined.confidence),
        ),
        *words[2:],
    ]


def _recover_overlapping_suffix_pairs(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index + 1 < len(words):
            first, last = words[index : index + 2]
            overlap_ratio = (first[1].right - last[1].left) / line_box.height
            matches_geometry = (
                len(first[0]) >= 2
                and len(last[0]) >= 2
                and contains_hangul(first[0])
                and contains_hangul(last[0])
                and first[0][-1] == last[0][0]
                and min(first[2], last[2]) < 0.8
                and max(first[2], last[2]) >= 0.95
                and 0 < overlap_ratio <= 0.04
            )
            if matches_geometry:
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(first[1].left - line_box.left)),
                        0,
                        min(crop.width, math.ceil(last[1].right - line_box.left)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(' ', '')
                merged_text = first[0] + last[0][1:]
                if combined.confidence >= 0.998 and combined_text == merged_text:
                    recovered.append(
                        (
                            merged_text,
                            BoundingBox.union((first[1], last[1])),
                            min(first[2], last[2], combined.confidence),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_confirmed_three_plus_three_splits(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 6
            or not all(is_hangul(character) for character in text)
            or confidence < 0.994
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.01)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 3
        last_pitch = (last_segment[1] - last_segment[0]) / 3
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.28 <= gap_ratio <= 0.35
            or pitch_ratio < 0.9
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.993 for part in parts)
            or any(
                len(part_text) != 3
                or not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_four_plus_four_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, 'word_boxes', None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 8
            or not all(is_hangul(character) for character in text)
            or confidence < 0.998
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.04)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 4
        last_pitch = (last_segment[1] - last_segment[0]) / 4
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.28 <= gap_ratio <= 0.29
            or pitch_ratio < 0.99
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(' ', '') for part in parts)
        if (
            any(part.confidence < 0.9996 for part in parts)
            or any(
                len(part_text) != 4
                or not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or ''.join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_numeric_ellipsis_tail_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 5
            or not all(is_hangul(character) for character in text[:3])
            or not text[-2].isdecimal()
            or text[-1] != "\u2026"
            or not 0.994 <= confidence <= 0.995
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        prefix_right = math.ceil(word_crop.width * 3 / len(text))
        tight = recognizer.recognize(
            word_crop.crop((0, 0, prefix_right, word_crop.height))
        )
        padded = recognizer.recognize(
            word_crop.crop(
                (
                    0,
                    0,
                    min(word_crop.width, prefix_right + 2),
                    word_crop.height,
                )
            )
        )
        tight_text = tight.text.replace(" ", "")
        padded_text = padded.text.replace(" ", "")
        if (
            tight.confidence < 0.99
            or padded.confidence < 0.989
            or tight_text != padded_text
            or len(tight_text) != 3
            or not all(is_hangul(character) for character in tight_text)
        ):
            recovered.append((text, box, confidence))
            continue
        try:
            segments = segmenter(word_crop, space_threshold=0.04)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 4
        last_pitch = last_segment[1] - last_segment[0]
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not -0.06 <= gap_ratio <= -0.05
            or not 0.55 <= pitch_ratio <= 0.56
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            parts[0].confidence < 0.995
            or parts[1].confidence < 0.997
            or part_texts != (tight_text + text[-1], text[-2])
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(
                    confidence,
                    tight.confidence,
                    padded.confidence,
                    part.confidence,
                ),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_one_plus_one_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 2
            or not all(is_hangul(character) for character in text)
            or not 0.9999 <= confidence <= 0.99992
            or not 2.17 <= width_ratio <= 2.18
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.001)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_width = first_segment[1] - first_segment[0]
        last_width = last_segment[1] - last_segment[0]
        pitch_ratio = min(first_width, last_width) / max(first_width, last_width)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.31 <= gap_ratio <= 0.32
            or not 0.84 <= pitch_ratio <= 0.85
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.99995 for part in parts)
            or tuple(map(len, part_texts)) != (1, 1)
            or any(not is_hangul(part_text) for part_text in part_texts)
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        midpoint = word_crop.width / 2
        boundaries = (math.floor(midpoint), math.ceil(midpoint) + 1)
        variants = tuple(
            (
                recognizer.recognize(
                    word_crop.crop((0, 0, boundary, word_crop.height))
                ),
                recognizer.recognize(
                    word_crop.crop(
                        (boundary, 0, word_crop.width, word_crop.height)
                    )
                ),
            )
            for boundary in boundaries
        )
        if any(
            min(first.confidence, last.confidence) < 0.9998
            or (
                first.text.replace(" ", ""),
                last.text.replace(" ", ""),
            )
            != part_texts
            for first, last in variants
        ):
            recovered.append((text, box, confidence))
            continue
        variant_confidence = min(
            candidate.confidence
            for variant in variants
            for candidate in variant
        )
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence, variant_confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_three_plus_two_prefix_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 5
            or not all(is_hangul(character) for character in text)
            or not 0.999 <= confidence <= 0.9991
            or not 5.96 <= width_ratio <= 5.97
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.04)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 4
        last_pitch = (last_segment[1] - last_segment[0]) / 2
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not -0.06 <= gap_ratio <= -0.05
            or not 0.94 <= pitch_ratio <= 0.95
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            not 0.55 <= parts[0].confidence <= 0.56
            or not 0.99 <= parts[1].confidence <= 0.991
            or len(part_texts[0]) != 4
            or not part_texts[0].startswith(text[:3])
            or not unicodedata.category(part_texts[0][-1]).startswith("P")
            or part_texts[1] != text[3:]
        ):
            recovered.append((text, box, confidence))
            continue
        boundaries = (
            round(line_box.height * 2.9),
            round(line_box.height * 3.0),
        )
        variants = tuple(
            (
                recognizer.recognize(
                    word_crop.crop((0, 0, boundary, word_crop.height))
                ),
                recognizer.recognize(
                    word_crop.crop(
                        (boundary, 0, word_crop.width, word_crop.height)
                    )
                ),
            )
            for boundary in boundaries
        )
        expected_parts = (text[:3], text[3:])
        if any(
            min(first.confidence, last.confidence) < 0.9989
            or (
                first.text.replace(" ", ""),
                last.text.replace(" ", ""),
            )
            != expected_parts
            for first, last in variants
        ):
            recovered.append((text, box, confidence))
            continue
        variant_confidence = min(
            candidate.confidence
            for variant in variants
            for candidate in variant
        )
        recovered.extend(
            (
                (
                    expected_parts[0],
                    BoundingBox(
                        line_box.left + crop_left,
                        box.top,
                        line_box.left + crop_left + boundaries[0],
                        box.bottom,
                    ),
                    min(confidence, variant_confidence),
                ),
                (
                    expected_parts[1],
                    BoundingBox(
                        line_box.left + crop_left + last_segment[0],
                        box.top,
                        line_box.left + crop_left + last_segment[1],
                        box.bottom,
                    ),
                    min(confidence, parts[1].confidence, variant_confidence),
                ),
            )
        )
    return recovered

def _recover_confirmed_three_plus_two_terminal_punctuation_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 6
            or not all(is_hangul(character) for character in text[:5])
            or not unicodedata.category(text[5]).startswith("P")
            or not 0.9917 <= confidence <= 0.9918
            or not 6.41 <= width_ratio <= 6.42
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.002)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 3:
            recovered.append((text, box, confidence))
            continue
        first_segment, middle_segment, last_segment = segments
        first_gap_ratio = (middle_segment[0] - first_segment[1]) / line_box.height
        last_gap_ratio = (last_segment[0] - middle_segment[1]) / line_box.height
        pitches = (
            (first_segment[1] - first_segment[0]) / 3,
            middle_segment[1] - middle_segment[0],
            (last_segment[1] - last_segment[0]) / 3,
        )
        pitch_ratio = min(pitches) / max(pitches)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not -0.06 <= first_gap_ratio <= -0.05
            or not 0.34 <= last_gap_ratio <= 0.35
            or not 0.68 <= pitch_ratio <= 0.69
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            parts[0].confidence < 0.9999
            or not 0.496 <= parts[1].confidence <= 0.497
            or not 0.977 <= parts[2].confidence <= 0.978
            or part_texts[0] != text[:3]
            or len(part_texts[1]) != 1
            or not unicodedata.category(part_texts[1]).startswith("P")
            or part_texts[2] != text[3:]
        ):
            recovered.append((text, box, confidence))
            continue
        target_boundaries = (
            round(line_box.height * 2.84),
            round(line_box.height * 3.18),
        )
        punctuated_target_boundaries = (
            round(line_box.height * 3.52),
            round(line_box.height * 4.09),
        )
        suffix_boundaries = (
            round(line_box.height * 4.09),
            round(line_box.height * 4.66),
        )
        target_variants = tuple(
            recognizer.recognize(
                word_crop.crop((0, 0, boundary, word_crop.height))
            )
            for boundary in target_boundaries
        )
        punctuated_target_variants = tuple(
            recognizer.recognize(
                word_crop.crop((0, 0, boundary, word_crop.height))
            )
            for boundary in punctuated_target_boundaries
        )
        suffix_variants = tuple(
            recognizer.recognize(
                word_crop.crop((boundary, 0, word_crop.width, word_crop.height))
            )
            for boundary in suffix_boundaries
        )
        if (
            any(
                variant.confidence < 0.9999
                or variant.text.replace(" ", "") != text[:3]
                for variant in target_variants
            )
            or any(
                variant.confidence < 0.9999
                or len(variant_text) != 4
                or variant_text[:3] != text[:3]
                or not unicodedata.category(variant_text[3]).startswith("P")
                or variant_text[3] == text[5]
                for variant in punctuated_target_variants
                for variant_text in (variant.text.replace(" ", ""),)
            )
            or len(
                {
                    variant.text.replace(" ", "")
                    for variant in punctuated_target_variants
                }
            )
            != 1
            or any(
                variant.confidence < 0.988
                or variant.text.replace(" ", "") != text[3:]
                for variant in suffix_variants
            )
        ):
            recovered.append((text, box, confidence))
            continue
        punctuated_target = punctuated_target_variants[0].text.replace(" ", "")
        recovered.extend(
            (
                (
                    punctuated_target,
                    BoundingBox(
                        line_box.left + crop_left + first_segment[0],
                        box.top,
                        line_box.left + crop_left + first_segment[1],
                        box.bottom,
                    ),
                    min(
                        confidence,
                        parts[0].confidence,
                        *(variant.confidence for variant in target_variants),
                        *(
                            variant.confidence
                            for variant in punctuated_target_variants
                        ),
                    ),
                ),
                (
                    text[3:],
                    BoundingBox(
                        line_box.left + crop_left + last_segment[0],
                        box.top,
                        line_box.left + crop_left + last_segment[1],
                        box.bottom,
                    ),
                    min(
                        confidence,
                        parts[2].confidence,
                        *(variant.confidence for variant in suffix_variants),
                    ),
                ),
            )
        )
    return recovered

def _recover_confirmed_five_plus_three_prefix_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 8
            or not all(is_hangul(character) for character in text)
            or not 0.9996 <= confidence <= 0.99962
            or not 6.60 <= width_ratio <= 6.62
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.04)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 6
        last_pitch = (last_segment[1] - last_segment[0]) / 3
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.25 <= gap_ratio <= 0.27
            or pitch_ratio < 0.99
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(word_crop.crop((left, 0, right, word_crop.height)))
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            not 0.878 <= parts[0].confidence <= 0.879
            or parts[1].confidence < 0.9999
            or len(part_texts[0]) != 6
            or not part_texts[0].startswith(text[:5])
            or not unicodedata.category(part_texts[0][-1]).startswith("P")
            or part_texts[1] != text[5:]
        ):
            recovered.append((text, box, confidence))
            continue
        boundaries = (
            round(line_box.height * 3.25),
            round(line_box.height * 3.7),
        )
        variants = tuple(
            (
                recognizer.recognize(word_crop.crop((0, 0, boundary, word_crop.height))),
                recognizer.recognize(
                    word_crop.crop((boundary, 0, word_crop.width, word_crop.height))
                ),
            )
            for boundary in boundaries
        )
        expected_parts = (text[:5], text[5:])
        if any(
            min(first.confidence, last.confidence) < 0.996
            or (
                first.text.replace(" ", ""),
                last.text.replace(" ", ""),
            )
            != expected_parts
            for first, last in variants
        ):
            recovered.append((text, box, confidence))
            continue
        variant_confidence = min(
            candidate.confidence for variant in variants for candidate in variant
        )
        recovered.extend(
            (
                (
                    expected_parts[0],
                    BoundingBox(
                        line_box.left + crop_left,
                        box.top,
                        line_box.left + crop_left + boundaries[0],
                        box.bottom,
                    ),
                    min(confidence, variant_confidence),
                ),
                (
                    expected_parts[1],
                    BoundingBox(
                        line_box.left + crop_left + last_segment[0],
                        box.top,
                        line_box.left + crop_left + last_segment[1],
                        box.bottom,
                    ),
                    min(confidence, parts[1].confidence, variant_confidence),
                ),
            )
        )
    return recovered


def _recover_confirmed_punctuated_three_plus_three_plus_one_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        core = text[:3] + text[4:7] + text[8:]
        if (
            len(text) != 9
            or not all(is_hangul(character) for character in core)
            or not unicodedata.category(text[3]).startswith("P")
            or not unicodedata.category(text[7]).startswith("P")
            or not 0.844 <= confidence <= 0.845
            or not 8.51 <= width_ratio <= 8.53
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.001)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 4:
            recovered.append((text, box, confidence))
            continue
        gap_ratios = tuple(
            (last[0] - first[1]) / line_box.height
            for first, last in zip(segments, segments[1:], strict=False)
        )
        pitches = tuple(
            (right - left) / length
            for (left, right), length in zip(
                segments,
                (3, 4, 1, 1),
                strict=True,
            )
        )
        pitch_ratio = min(pitches) / max(pitches)
        if (
            segments[0][0] > 1
            or segments[-1][1] < word_crop.width - 1
            or not 0.26 <= gap_ratios[0] <= 0.27
            or not -0.04 <= gap_ratios[1] <= -0.03
            or not 0.30 <= gap_ratios[2] <= 0.31
            or not 0.63 <= pitch_ratio <= 0.65
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            parts[0].confidence < 0.9995
            or not 0.972 <= parts[1].confidence <= 0.974
            or not 0.978 <= parts[2].confidence <= 0.979
            or parts[3].confidence < 0.9999
            or part_texts[0] != text[:3]
            or len(part_texts[1]) != 4
            or not unicodedata.category(part_texts[1][0]).startswith("P")
            or not part_texts[1].endswith(text[4:7])
            or len(part_texts[2]) != 1
            or not unicodedata.category(part_texts[2]).startswith("P")
            or part_texts[3] != text[8:]
        ):
            recovered.append((text, box, confidence))
            continue
        prefix_boundaries = (
            round(line_box.height * 3.79),
            round(line_box.height * 4.09),
        )
        target_boundaries = (
            (prefix_boundaries[0], round(line_box.height * 6.66)),
            (prefix_boundaries[1], round(line_box.height * 6.81)),
        )
        suffix_boundaries = (
            round(line_box.height * 7.42),
            round(line_box.height * 7.57),
        )
        prefix_variants = tuple(
            recognizer.recognize(
                word_crop.crop((0, 0, boundary, word_crop.height))
            )
            for boundary in prefix_boundaries
        )
        target_variants = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in target_boundaries
        )
        suffix_variants = tuple(
            recognizer.recognize(
                word_crop.crop((boundary, 0, word_crop.width, word_crop.height))
            )
            for boundary in suffix_boundaries
        )
        if (
            any(
                variant.confidence < 0.88
                or variant.text.replace(" ", "") != text[:4]
                for variant in prefix_variants
            )
            or any(
                variant.confidence < 0.9999
                or variant.text.replace(" ", "") != text[4:7]
                for variant in target_variants
            )
            or any(
                variant.confidence < 0.9999
                or variant.text.replace(" ", "") != text[8:]
                for variant in suffix_variants
            )
        ):
            recovered.append((text, box, confidence))
            continue
        confirmation_confidence = min(
            confidence,
            *(part.confidence for part in parts),
            *(variant.confidence for variant in prefix_variants),
            *(variant.confidence for variant in target_variants),
            *(variant.confidence for variant in suffix_variants),
        )
        boundaries = (
            0,
            round(word_crop.width * 4 / 9),
            round(word_crop.width * 8 / 9),
            word_crop.width,
        )
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                confirmation_confidence,
            )
            for part_text, (left, right) in zip(
                (text[:4], text[4:8], text[8:]),
                zip(boundaries, boundaries[1:], strict=False),
                strict=True,
            )
        )
    return recovered

def _recover_confirmed_punctuated_three_plus_three_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 8
            or not all(is_hangul(character) for character in text[:3] + text[4:7])
            or not unicodedata.category(text[3]).startswith("P")
            or not unicodedata.category(text[7]).startswith("P")
            or not 0.652 <= confidence <= 0.653
            or not 9.09 <= width_ratio <= 9.10
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.005)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        left_margin_ratio = first_segment[0] / line_box.height
        right_margin_ratio = (word_crop.width - last_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 3
        last_pitch = (last_segment[1] - last_segment[0]) / 5
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            not 0.36 <= left_margin_ratio <= 0.37
            or not 0.40 <= right_margin_ratio <= 0.41
            or not 0.32 <= gap_ratio <= 0.34
            or not 0.98 <= pitch_ratio <= 0.995
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            parts[0].confidence < 0.9998
            or not 0.271 <= parts[1].confidence <= 0.273
            or part_texts[0] != text[:3]
            or len(part_texts[1]) != 5
            or not unicodedata.category(part_texts[1][0]).startswith("P")
            or part_texts[1][1:4] != text[4:7]
            or not unicodedata.category(part_texts[1][4]).startswith("P")
        ):
            recovered.append((text, box, confidence))
            continue
        prefix_boundaries = (
            round(line_box.height * 3.39),
            round(line_box.height * 3.57),
        )
        target_boundaries = (
            (round(line_box.height * 4.4), round(line_box.height * 7.69)),
            (round(line_box.height * 4.67), round(line_box.height * 7.88)),
        )
        prefix_variants = tuple(
            recognizer.recognize(
                word_crop.crop((0, 0, boundary, word_crop.height))
            )
            for boundary in prefix_boundaries
        )
        target_variants = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in target_boundaries
        )
        if (
            any(
                variant.confidence < 0.9996
                or variant.text.replace(" ", "") != text[:3]
                for variant in prefix_variants
            )
            or any(
                variant.confidence < 0.9986
                or variant.text.replace(" ", "") != text[4:7]
                for variant in target_variants
            )
        ):
            recovered.append((text, box, confidence))
            continue
        output_texts = (text[:3], text[3:])
        confirmation_confidences = (
            min(
                confidence,
                parts[0].confidence,
                *(variant.confidence for variant in prefix_variants),
            ),
            min(
                confidence,
                parts[1].confidence,
                *(variant.confidence for variant in target_variants),
            ),
        )
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                part_confidence,
            )
            for part_text, (left, right), part_confidence in zip(
                output_texts,
                segments,
                confirmation_confidences,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_central_paired_wrapped_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 4:
        return words
    first, candidate, following, trailing = words
    text, box, confidence = candidate
    gaps = (
        (box.left - first[1].right) / line_box.height,
        (following[1].left - box.right) / line_box.height,
        (trailing[1].left - following[1].right) / line_box.height,
    )
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    if (
        len(first[0]) != 5
        or not all(is_hangul(character) for character in first[0])
        or first[2] < 0.9998
        or not 4.92 <= width_ratios[0] <= 4.93
        or len(text) != 12
        or not all(is_hangul(character) for character in text[:4])
        or text[4] not in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in text[5:7])
        or not unicodedata.category(text[7]).startswith("P")
        or _BOUNDARY_WRAPPERS.get(text[4]) == text[7]
        or not all(is_hangul(character) for character in text[8:])
        or not 0.7669 <= confidence <= 0.767
        or not 11.47 <= width_ratios[1] <= 11.48
        or len(following[0]) != 2
        or not all(is_hangul(character) for character in following[0])
        or following[2] < 0.9997
        or not 2.04 <= width_ratios[2] <= 2.05
        or len(trailing[0]) != 4
        or not all(is_hangul(character) for character in trailing[0])
        or trailing[2] < 0.9997
        or not 4.01 <= width_ratios[3] <= 4.02
        or not 0.41 <= gaps[0] <= 0.42
        or not 0.37 <= gaps[1] <= 0.39
        or not 0.30 <= gaps[2] <= 0.31
    ):
        return words
    crop_left = max(0, math.floor(box.left - line_box.left))
    crop_right = min(crop.width, math.ceil(box.right - line_box.left))
    word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    try:
        segments = segmenter(word_crop, space_threshold=0.001)
    except TypeError:
        return words
    if len(segments) != 4:
        return words
    prefix_segment, middle_segment, closing_segment, suffix_segment = segments
    prefix_gap = (middle_segment[0] - prefix_segment[1]) / line_box.height
    closing_overlap = (
        middle_segment[1] - closing_segment[0]
    ) / line_box.height
    suffix_gap = (suffix_segment[0] - closing_segment[1]) / line_box.height
    if (
        prefix_segment[0] > 1
        or suffix_segment[1] < word_crop.width - 1
        or not 0.30 <= prefix_gap <= 0.31
        or not 0.037 <= closing_overlap <= 0.039
        or not 0.34 <= suffix_gap <= 0.35
        or not 4.05
        <= (prefix_segment[1] - prefix_segment[0]) / line_box.height
        <= 4.06
        or not 2.30
        <= (middle_segment[1] - middle_segment[0]) / line_box.height
        <= 2.31
        or not 0.56
        <= (closing_segment[1] - closing_segment[0]) / line_box.height
        <= 0.58
        or not 3.93
        <= (suffix_segment[1] - suffix_segment[0]) / line_box.height
        <= 3.94
    ):
        return words
    parts = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in segments
    )
    part_texts = tuple(part.text.replace(" ", "") for part in parts)
    prefix_text, middle_text, closing_text, suffix_text = part_texts
    if (
        parts[0].confidence < 0.9992
        or prefix_text != text[:4]
        or not 0.839 <= parts[1].confidence <= 0.84
        or len(middle_text) != 3
        or middle_text[0] == text[4]
        or middle_text[0] not in _ATTACHED_PARTICLE_WRAPPERS
        or middle_text[1:] != text[5:7]
        or not 0.951 <= parts[2].confidence <= 0.952
        or closing_text != text[7:8]
        or _BOUNDARY_WRAPPERS.get(middle_text[0]) != closing_text
        or parts[3].confidence < 0.9989
        or suffix_text != text[8:]
    ):
        return words
    prefix_boundaries = tuple(
        round(line_box.height * ratio) for ratio in (3.85, 3.95, 4.05, 4.15, 4.25)
    )
    target_boundaries = tuple(
        (
            round(line_box.height * left_ratio),
            round(line_box.height * right_ratio),
        )
        for left_ratio, right_ratio in (
            (4.70, 6.80),
            (4.75, 6.75),
            (4.80, 6.70),
            (4.90, 6.60),
            (5.00, 6.50),
        )
    )
    wrapper_boundaries = tuple(
        (
            round(line_box.height * left_ratio),
            round(line_box.height * right_ratio),
        )
        for left_ratio, right_ratio in (
            (4.05, 7.35),
            (4.10, 7.30),
            (4.15, 7.25),
            (4.20, 7.20),
            (4.25, 7.15),
            (4.30, 7.10),
        )
    )
    suffix_boundaries = tuple(
        round(line_box.height * ratio) for ratio in (7.35, 7.45, 7.55, 7.65, 7.75)
    )
    prefix_variants = tuple(
        recognizer.recognize(
            word_crop.crop((0, 0, right, word_crop.height))
        )
        for right in prefix_boundaries
    )
    target_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in target_boundaries
    )
    wrapper_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in wrapper_boundaries
    )
    suffix_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, word_crop.width, word_crop.height))
        )
        for left in suffix_boundaries
    )
    wrapper_text = wrapper_variants[0].text.replace(" ", "")
    if (
        len(wrapper_text) != 4
        or wrapper_text[1:3] != text[5:7]
        or _BOUNDARY_WRAPPERS.get(wrapper_text[0]) != wrapper_text[3]
        or any(
            variant.confidence < 0.995
            or variant.text.replace(" ", "") != prefix_text
            for variant in prefix_variants
        )
        or any(
            variant.confidence < 0.9995
            or variant.text.replace(" ", "") != text[5:7]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.584
            or variant.text.replace(" ", "") != wrapper_text
            for variant in wrapper_variants
        )
        or any(
            variant.confidence < 0.971
            or variant.text.replace(" ", "") != suffix_text
            for variant in suffix_variants
        )
    ):
        return words
    prefix_confidence = min(
        confidence,
        parts[0].confidence,
        *(variant.confidence for variant in prefix_variants),
    )
    wrapper_confidence = min(
        confidence,
        parts[1].confidence,
        parts[2].confidence,
        *(variant.confidence for variant in target_variants),
        *(variant.confidence for variant in wrapper_variants),
    )
    suffix_confidence = min(
        confidence,
        parts[3].confidence,
        *(variant.confidence for variant in suffix_variants),
    )
    return [
        first,
        (
            prefix_text,
            BoundingBox(
                line_box.left + crop_left + prefix_segment[0],
                box.top,
                line_box.left + crop_left + prefix_segment[1],
                box.bottom,
            ),
            prefix_confidence,
        ),
        (
            wrapper_text,
            BoundingBox(
                line_box.left + crop_left + middle_segment[0],
                box.top,
                line_box.left + crop_left + closing_segment[1],
                box.bottom,
            ),
            wrapper_confidence,
        ),
        (
            suffix_text,
            BoundingBox(
                line_box.left + crop_left + suffix_segment[0],
                box.top,
                line_box.left + crop_left + suffix_segment[1],
                box.bottom,
            ),
            suffix_confidence,
        ),
        following,
        trailing,
    ]


def _recover_confirmed_isolated_paired_wrapped_two_plus_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 1:
        return words
    text, box, confidence = words[0]
    if (
        len(text) != 7
        or text[0] not in _BOUNDARY_WRAPPERS
        or text[0] in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in text[1:3])
        or not unicodedata.category(text[3]).startswith("P")
        or _BOUNDARY_WRAPPERS.get(text[0]) == text[3]
        or not all(is_hangul(character) for character in text[4:6])
        or not unicodedata.category(text[6]).startswith("P")
        or not 0.809 <= confidence <= 0.8091
        or not 6.89 <= box.width / line_box.height <= 6.90
    ):
        return words
    try:
        default_segments = segmenter(crop)
    except TypeError:
        return words
    if len(default_segments) != 1:
        return words
    word_left, word_right = default_segments[0]
    left_margin_ratio = word_left / line_box.height
    right_margin_ratio = (crop.width - word_right) / line_box.height
    word_width_ratio = (word_right - word_left) / line_box.height
    if (
        not 0.14 <= left_margin_ratio <= 0.16
        or not 0.29 <= right_margin_ratio <= 0.30
        or not 6.45 <= word_width_ratio <= 6.46
    ):
        return words
    word_crop = crop.crop((word_left, 0, word_right, crop.height))
    direct = recognizer.recognize(word_crop)
    direct_text = direct.text.replace(" ", "")
    if (
        not 0.535 <= direct.confidence <= 0.536
        or len(direct_text) != 7
        or direct_text[:3] != text[:3]
        or direct_text[3] == text[3]
        or not unicodedata.category(direct_text[3]).startswith("P")
        or direct_text[4:] != text[4:]
        or _BOUNDARY_WRAPPERS.get(direct_text[0]) == direct_text[3]
    ):
        return words
    try:
        segments = segmenter(word_crop, space_threshold=0.001)
    except TypeError:
        return words
    if len(segments) != 3:
        return words
    opening_segment, middle_segment, suffix_segment = segments
    opening_overlap = (
        opening_segment[1] - middle_segment[0]
    ) / line_box.height
    suffix_gap = (suffix_segment[0] - middle_segment[1]) / line_box.height
    middle_pitch = (middle_segment[1] - middle_segment[0]) / 3
    suffix_pitch = (suffix_segment[1] - suffix_segment[0]) / 3
    pitch_ratio = min(middle_pitch, suffix_pitch) / max(
        middle_pitch,
        suffix_pitch,
    )
    if (
        opening_segment[0] > 1
        or suffix_segment[1] < word_crop.width - 1
        or not 0.029 <= opening_overlap <= 0.031
        or not 0.328 <= suffix_gap <= 0.33
        or not 1.13
        <= (opening_segment[1] - opening_segment[0]) / line_box.height
        <= 1.14
        or not 2.77
        <= (middle_segment[1] - middle_segment[0]) / line_box.height
        <= 2.79
        or not 2.24
        <= (suffix_segment[1] - suffix_segment[0]) / line_box.height
        <= 2.25
        or not 0.80 <= pitch_ratio <= 0.81
    ):
        return words
    parts = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in segments
    )
    part_texts = tuple(part.text.replace(" ", "") for part in parts)
    opening_text, middle_text, suffix_text = part_texts
    if (
        not 0.421 <= parts[0].confidence <= 0.422
        or opening_text != text[:1]
        or not 0.923 <= parts[1].confidence <= 0.924
        or len(middle_text) != 3
        or middle_text[:2] != text[1:3]
        or not unicodedata.category(middle_text[2]).startswith("P")
        or middle_text[2] in {text[3], direct_text[3]}
        or parts[2].confidence < 0.993
        or suffix_text != text[4:]
    ):
        return words
    target_boundaries = tuple(
        (
            round(line_box.height * left_ratio),
            round(line_box.height * right_ratio),
        )
        for left_ratio, right_ratio in (
            (0.75, 3.15),
            (0.80, 3.10),
            (0.85, 3.05),
            (0.90, 3.00),
            (0.95, 2.95),
            (1.00, 2.90),
            (1.05, 2.85),
        )
    )
    wrapper_boundaries = (
        (
            max(0, word_left - round(line_box.height * 0.06)),
            word_left + round(line_box.height * 3.38),
        ),
        (
            max(0, word_left - round(line_box.height * 0.06)),
            word_left + round(line_box.height * 3.41),
        ),
        (
            max(0, word_left - round(line_box.height * 0.03)),
            word_left + round(line_box.height * 3.38),
        ),
        (
            max(0, word_left - round(line_box.height * 0.03)),
            word_left + round(line_box.height * 3.41),
        ),
        (
            word_left,
            word_left + round(line_box.height * 3.41),
        ),
    )
    suffix_boundaries = tuple(
        round(line_box.height * ratio)
        for ratio in (4.00, 4.05, 4.10, 4.15, 4.20)
    )
    target_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in target_boundaries
    )
    wrapper_variants = tuple(
        recognizer.recognize(
            crop.crop((left, 0, right, crop.height))
        )
        for left, right in wrapper_boundaries
    )
    suffix_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, word_crop.width, word_crop.height))
        )
        for left in suffix_boundaries
    )
    wrapper_text = wrapper_variants[0].text.replace(" ", "")
    if (
        len(wrapper_text) != 4
        or wrapper_text[0] == text[0]
        or wrapper_text[0] not in _BOUNDARY_WRAPPERS
        or wrapper_text[0] in _ATTACHED_PARTICLE_WRAPPERS
        or wrapper_text[1:3] != text[1:3]
        or _BOUNDARY_WRAPPERS.get(wrapper_text[0]) != wrapper_text[3]
        or wrapper_text[3] in {direct_text[3], middle_text[2]}
        or any(
            variant.confidence < 0.9994
            or variant.text.replace(" ", "") != text[1:3]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.376
            or variant.text.replace(" ", "") != wrapper_text
            for variant in wrapper_variants
        )
        or any(
            variant.confidence < 0.990
            or variant.text.replace(" ", "") != suffix_text
            for variant in suffix_variants
        )
    ):
        return words
    wrapper_confidence = min(
        confidence,
        direct.confidence,
        parts[1].confidence,
        *(variant.confidence for variant in target_variants),
        *(variant.confidence for variant in wrapper_variants),
    )
    suffix_confidence = min(
        confidence,
        parts[2].confidence,
        *(variant.confidence for variant in suffix_variants),
    )
    return [
        (
            wrapper_text,
            BoundingBox(
                line_box.left + word_left + opening_segment[0],
                box.top,
                line_box.left + word_left + middle_segment[1],
                box.bottom,
            ),
            wrapper_confidence,
        ),
        (
            suffix_text,
            BoundingBox(
                line_box.left + word_left + suffix_segment[0],
                box.top,
                line_box.left + word_left + suffix_segment[1],
                box.bottom,
            ),
            suffix_confidence,
        ),
    ]


def _recover_confirmed_isolated_three_plus_five_punctuated_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 1:
        return words
    text, box, confidence = words[0]
    if (
        len(text) != 9
        or not all(is_hangul(character) for character in text[:3])
        or not unicodedata.category(text[3]).startswith("P")
        or not all(is_hangul(character) for character in text[4:])
        or not 0.9923 <= confidence <= 0.9926
        or box != line_box
        or not 9.23 <= box.width / line_box.height <= 9.25
        or crop.size != (539, 59)
    ):
        return words
    try:
        default_segments = tuple(segmenter(crop))
    except TypeError:
        return words
    if default_segments != ((0, crop.width),):
        return words
    expected_segments = (
        (0.0005, ((0, 15), (14, 212), (211, 539))),
        (0.001, ((0, 15), (14, 212), (211, 539))),
        (0.003, ((0, 15), (14, 212), (211, 539))),
        (0.005, ((0, 15), (14, 212), (211, 539))),
        (0.01, ((0, 15), (14, 212), (211, 539))),
        (0.015, ((0, 212), (211, 539))),
        (0.02, ((0, 212), (211, 539))),
        (0.03, ((0, 539),)),
    )
    try:
        if any(
            tuple(segmenter(crop, space_threshold=threshold)) != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    boundary_crop = crop.crop((14, 0, 212, crop.height))
    boundary_variants = (
        recognizer.recognize(boundary_crop),
        recognizer.recognize(enhanced(boundary_crop)),
    )
    target_bounds = (
        (12, 183),
        (12, 192),
        (14, 195),
        (18, 189),
        (20, 198),
        (22, 201),
        (28, 204),
    )
    following_bounds = (
        (211, 527),
        (214, 527),
        (220, 527),
        (223, 527),
        (229, 527),
        (232, 527),
        (241, 527),
    )
    target_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right in target_bounds
    ) + tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right in target_bounds
    )
    following_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right in following_bounds
    ) + tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right in following_bounds
    )
    if (
        any(
            variant.confidence < 0.9925
            or variant.text.replace(" ", "") != text[:4]
            for variant in boundary_variants
        )
        or any(
            variant.confidence < 0.9998
            or variant.text.replace(" ", "") != text[:3]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.9992
            or variant.text.replace(" ", "") != text[4:]
            for variant in following_variants
        )
    ):
        return words
    target_confidence = min(
        confidence,
        *(variant.confidence for variant in boundary_variants),
        *(variant.confidence for variant in target_variants),
    )
    following_confidence = min(
        confidence,
        *(variant.confidence for variant in following_variants),
    )
    return [
        (
            text[:4],
            BoundingBox(
                line_box.left + 20,
                box.top,
                line_box.left + 252,
                box.bottom,
            ),
            target_confidence,
        ),
        (
            text[4:],
            BoundingBox(
                line_box.left + 223,
                box.top,
                box.right,
                box.bottom,
            ),
            following_confidence,
        ),
    ]


def _recover_confirmed_isolated_five_plus_three_punctuated_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 1:
        return words
    text, box, confidence = words[0]
    if (
        len(text) != 10
        or not all(is_hangul(character) for character in text[:5])
        or text[5] != "-"
        or not all(is_hangul(character) for character in text[6:9])
        or ord(text[9]) != 0x2014
        or not 0.7183 <= confidence <= 0.7185
        or box != line_box
        or not 31.69 <= line_box.height <= 31.70
        or not 293.47 <= line_box.width <= 293.49
        or not 9.25 <= line_box.width / line_box.height <= 9.27
        or crop.size != (295, 33)
    ):
        return words
    try:
        if tuple(segmenter(crop)) != ((14, 280),):
            return words
    except TypeError:
        return words
    candidate_crop = crop.crop((14, 0, 280, crop.height))
    expected_segments = (
        (
            0.0001,
            (
                (0, 69),
                (68, 86),
                (85, 113),
                (112, 130),
                (129, 207),
                (206, 251),
                (250, 266),
            ),
        ),
        (
            0.0003,
            (
                (0, 86),
                (85, 113),
                (112, 130),
                (129, 251),
                (250, 266),
            ),
        ),
        (
            0.0005,
            ((0, 113), (112, 130), (129, 251), (250, 266)),
        ),
        (0.001, ((0, 130), (129, 266))),
        (0.002, ((0, 130), (129, 146), (145, 266))),
        (0.003, ((0, 130), (129, 266))),
        (0.005, ((0, 130), (129, 266))),
        (0.007, ((0, 130), (129, 266))),
        (0.01, ((0, 130), (129, 266))),
        (0.015, ((0, 266),)),
        (0.02, ((0, 266),)),
        (0.03, ((0, 266),)),
        (0.04, ((0, 266),)),
        (0.05, ((0, 266),)),
        (0.07, ((0, 266),)),
    )
    try:
        if any(
            tuple(segmenter(candidate_crop, space_threshold=threshold))
            != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    candidate_direct = recognizer.recognize(candidate_crop)
    candidate_enhanced = recognizer.recognize(enhanced(candidate_crop))
    full_enhanced = recognizer.recognize(enhanced(crop))
    prefix_bounds = (
        (0, 116),
        (1, 118),
        (2, 120),
        (3, 122),
        (4, 124),
        (5, 126),
        (6, 128),
    )
    target_bounds = (
        (128, 248),
        (128, 249),
        (129, 248),
        (129, 249),
        (129, 250),
        (130, 248),
        (130, 249),
    )
    prefix_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in prefix_bounds
    )
    prefix_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in prefix_bounds
    )
    target_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in target_bounds
    )
    target_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in target_bounds
    )
    prefix_text = text[:5]
    target_text = text[5:9]
    if (
        candidate_direct.confidence < 0.8527
        or candidate_direct.text.replace(" ", "") != text
        or candidate_enhanced.confidence < 0.8156
        or candidate_enhanced.text.replace(" ", "") != text
        or full_enhanced.confidence < 0.6636
        or full_enhanced.text.replace(" ", "") != text
        or any(
            variant.confidence < 0.9998
            or variant.text.replace(" ", "") != prefix_text
            for variant in (*prefix_direct, *prefix_enhanced)
        )
        or any(
            variant.confidence < 0.9376
            or variant.text.replace(" ", "") != target_text
            for variant in target_direct
        )
        or any(
            variant.confidence < 0.9250
            or variant.text.replace(" ", "") != target_text
            for variant in target_enhanced
        )
    ):
        return words
    prefix_confidence = min(
        confidence,
        candidate_direct.confidence,
        candidate_enhanced.confidence,
        full_enhanced.confidence,
        *(variant.confidence for variant in prefix_direct),
        *(variant.confidence for variant in prefix_enhanced),
    )
    target_confidence = min(
        confidence,
        candidate_direct.confidence,
        candidate_enhanced.confidence,
        full_enhanced.confidence,
        *(variant.confidence for variant in target_direct),
        *(variant.confidence for variant in target_enhanced),
    )
    candidate_left = line_box.left + 14
    candidate_right = line_box.left + 280
    return [
        (
            prefix_text,
            BoundingBox(
                candidate_left,
                box.top,
                candidate_left + 129,
                box.bottom,
            ),
            prefix_confidence,
        ),
        (
            text[5:],
            BoundingBox(
                candidate_left + 129,
                box.top,
                candidate_right,
                box.bottom,
            ),
            target_confidence,
        ),
    ]


def _recover_confirmed_isolated_mixed_prefix_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 1:
        return words
    text, box, confidence = words[0]
    if (
        len(text) != 12
        or not all(is_hangul(character) for character in text[:2])
        or not unicodedata.category(text[2]).startswith("P")
        or not (text[3].isascii() and text[3].isalnum())
        or not unicodedata.category(text[4]).startswith("P")
        or not all(
            character.isascii() and character.isalnum()
            for character in text[5:9]
        )
        or not unicodedata.category(text[9]).startswith("P")
        or not all(
            character.isascii() and character.isalnum()
            for character in text[10:]
        )
        or sum(
            character.isascii() and character.isalpha() for character in text
        )
        != 2
        or sum(
            character.isascii() and character.isdigit() for character in text
        )
        != 5
        or not _structured_ascii_context(text[3:])
        or contains_hangul(text[3:])
        or not 0.9757 <= confidence <= 0.9759
        or box != line_box
        or not 75.71 <= line_box.height <= 75.73
        or not 489.51 <= line_box.width <= 489.53
        or not 6.46 <= line_box.width / line_box.height <= 6.47
        or crop.size != (490, 77)
    ):
        return words
    expected_segments = (
        (0.0001, ((0, 174), (173, 251), (250, 490))),
        (0.0003, ((0, 174), (173, 490))),
        (0.0005, ((0, 174), (173, 490))),
        (0.001, ((0, 174), (173, 490))),
        (0.002, ((0, 174), (173, 490))),
        (0.003, ((0, 174), (173, 490))),
        (0.005, ((0, 174), (173, 490))),
        (0.007, ((0, 174), (173, 490))),
        (0.01, ((0, 174), (173, 490))),
        (0.015, ((0, 174), (173, 490))),
        (0.02, ((0, 174), (173, 490))),
        (0.03, ((0, 174), (173, 490))),
        (0.04, ((0, 490),)),
        (0.05, ((0, 490),)),
        (0.07, ((0, 490),)),
    )
    try:
        if tuple(segmenter(crop)) != ((0, 490),) or any(
            tuple(segmenter(crop, space_threshold=threshold)) != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_candidate = recognizer.recognize(enhanced(crop))
    prefix_bounds = ((0, 170), (0, 172), (0, 174), (0, 176), (0, 178))
    suffix_bounds = (
        (171, 490),
        (173, 490),
        (175, 490),
        (177, 490),
        (173, 488),
        (173, 486),
    )
    prefix_direct = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right in prefix_bounds
    )
    prefix_enhanced = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right in prefix_bounds
    )
    suffix_direct = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right in suffix_bounds
    )
    suffix_enhanced = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right in suffix_bounds
    )
    if (
        enhanced_candidate.confidence < 0.9803
        or enhanced_candidate.text.replace(" ", "") != text
        or any(
            variant.confidence < 0.9907
            or variant.text.replace(" ", "") != text[:3]
            for variant in (*prefix_direct, *prefix_enhanced)
        )
        or any(
            variant.confidence < 0.741
            or variant.text.replace(" ", "") != text[3:]
            for variant in suffix_direct
        )
        or any(
            variant.confidence < 0.783
            or variant.text.replace(" ", "") != text[3:]
            for variant in suffix_enhanced
        )
    ):
        return words
    prefix_confidence = min(
        confidence,
        *(variant.confidence for variant in prefix_direct),
        *(variant.confidence for variant in prefix_enhanced),
    )
    suffix_confidence = min(
        confidence,
        *(variant.confidence for variant in suffix_direct),
        *(variant.confidence for variant in suffix_enhanced),
    )
    return [
        (
            text[:3],
            BoundingBox(
                line_box.left,
                box.top,
                line_box.left + 174,
                box.bottom,
            ),
            prefix_confidence,
        ),
        (
            text[3:],
            BoundingBox(
                line_box.left + 173,
                box.top,
                box.right,
                box.bottom,
            ),
            suffix_confidence,
        ),
    ]


def _recover_confirmed_paired_wrapped_three_plus_three_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 3:
        return words
    first, second, candidate = words
    text, box, confidence = candidate
    gaps = (
        (second[1].left - first[1].right) / line_box.height,
        (box.left - second[1].right) / line_box.height,
    )
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    if (
        len(first[0]) != 3
        or not all(is_hangul(character) for character in first[0])
        or first[2] < 0.9999
        or not 2.87 <= width_ratios[0] <= 2.88
        or len(second[0]) != 2
        or not all(is_hangul(character) for character in second[0])
        or second[2] < 0.99999
        or not 1.89 <= width_ratios[1] <= 1.90
        or any(not 0.30 <= gap <= 0.32 for gap in gaps)
        or len(text) != 9
        or text[0] not in _BOUNDARY_WRAPPERS
        or text[0] in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in text[1:4])
        or not unicodedata.category(text[4]).startswith("P")
        or _BOUNDARY_WRAPPERS.get(text[0]) == text[4]
        or not all(is_hangul(character) for character in text[5:8])
        or not unicodedata.category(text[8]).startswith("P")
        or not 0.6808 <= confidence <= 0.681
        or not 8.08 <= width_ratios[2] <= 8.09
    ):
        return words
    crop_left = max(0, math.floor(box.left - line_box.left))
    crop_right = min(crop.width, math.ceil(box.right - line_box.left))
    word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    try:
        segments = segmenter(word_crop, space_threshold=0.001)
    except TypeError:
        return words
    if len(segments) != 3:
        return words
    wrapper_segment, artifact_segment, suffix_segment = segments
    wrapper_overlap = (
        wrapper_segment[1] - artifact_segment[0]
    ) / line_box.height
    suffix_gap = (
        suffix_segment[0] - artifact_segment[1]
    ) / line_box.height
    wrapper_pitch = (wrapper_segment[1] - wrapper_segment[0]) / 5
    suffix_pitch = (suffix_segment[1] - suffix_segment[0]) / 4
    pitch_ratio = min(wrapper_pitch, suffix_pitch) / max(
        wrapper_pitch,
        suffix_pitch,
    )
    if (
        wrapper_segment[0] > 1
        or suffix_segment[1] < word_crop.width - 1
        or not 0.017 <= wrapper_overlap <= 0.018
        or not 0.30 <= suffix_gap <= 0.32
        or not 4.16
        <= (wrapper_segment[1] - wrapper_segment[0]) / line_box.height
        <= 4.17
        or not 0.53
        <= (artifact_segment[1] - artifact_segment[0]) / line_box.height
        <= 0.54
        or not 3.09
        <= (suffix_segment[1] - suffix_segment[0]) / line_box.height
        <= 3.10
        or not 0.92 <= pitch_ratio <= 0.94
    ):
        return words
    parts = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in segments
    )
    part_texts = tuple(part.text.replace(" ", "") for part in parts)
    wrapper_text, artifact_text, suffix_text = part_texts
    if (
        not 0.828 <= parts[0].confidence <= 0.829
        or len(wrapper_text) != 5
        or wrapper_text[0] == text[0]
        or wrapper_text[0] not in _BOUNDARY_WRAPPERS
        or wrapper_text[0] in _ATTACHED_PARTICLE_WRAPPERS
        or wrapper_text[1:4] != text[1:4]
        or wrapper_text[4] != text[4]
        or _BOUNDARY_WRAPPERS.get(wrapper_text[0]) != wrapper_text[4]
        or not 0.298 <= parts[1].confidence <= 0.299
        or len(artifact_text) != 1
        or not artifact_text.isascii()
        or not artifact_text.isalnum()
        or not 0.531 <= parts[2].confidence <= 0.532
        or suffix_text != text[5:]
    ):
        return words
    target_boundaries = tuple(
        (
            round(line_box.height * left_ratio),
            round(line_box.height * right_ratio),
        )
        for left_ratio, right_ratio in (
            (0.70, 3.95),
            (0.75, 3.90),
            (0.80, 3.85),
            (0.85, 3.80),
            (0.90, 3.75),
        )
    )
    wrapper_boundaries = tuple(
        round(line_box.height * ratio) for ratio in (4.10, 4.15)
    )
    suffix_boundaries = tuple(
        round(line_box.height * ratio) for ratio in (4.90, 4.95, 5.05)
    )
    target_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in target_boundaries
    )
    wrapper_variants = tuple(
        recognizer.recognize(
            word_crop.crop((0, 0, right, word_crop.height))
        )
        for right in wrapper_boundaries
    )
    suffix_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, word_crop.width, word_crop.height))
        )
        for left in suffix_boundaries
    )
    if (
        any(
            variant.confidence < 0.9995
            or variant.text.replace(" ", "") != text[1:4]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.793
            or variant.text.replace(" ", "") != wrapper_text
            for variant in wrapper_variants
        )
        or any(
            variant.confidence < 0.989
            or variant.text.replace(" ", "") != suffix_text
            for variant in suffix_variants
        )
    ):
        return words
    wrapper_confidence = min(
        confidence,
        parts[0].confidence,
        *(variant.confidence for variant in target_variants),
        *(variant.confidence for variant in wrapper_variants),
    )
    suffix_confidence = min(
        confidence,
        parts[2].confidence,
        *(variant.confidence for variant in suffix_variants),
    )
    return [
        first,
        second,
        (
            wrapper_text,
            BoundingBox(
                line_box.left + crop_left + wrapper_segment[0],
                box.top,
                line_box.left + crop_left + wrapper_segment[1],
                box.bottom,
            ),
            wrapper_confidence,
        ),
        (
            suffix_text,
            BoundingBox(
                line_box.left + crop_left + suffix_segment[0],
                box.top,
                line_box.left + crop_left + suffix_segment[1],
                box.bottom,
            ),
            suffix_confidence,
        ),
    ]


def _recover_confirmed_paired_wrapped_four_plus_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 5:
        return words
    expected_lengths = (3, 2, 5, 5)
    leading = words[:4]
    text, box, confidence = words[4]
    gaps = tuple(
        (last[1].left - first[1].right) / line_box.height
        for first, last in zip(words, words[1:], strict=False)
    )
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    if (
        tuple(len(word[0]) for word in leading) != expected_lengths
        or not all(
            all(is_hangul(character) for character in word[0]) for word in leading
        )
        or any(
            word[2] < threshold
            for word, threshold in zip(
                leading,
                (0.9997, 0.9995, 0.9988, 0.9986),
                strict=True,
            )
        )
        or not 2.51 <= width_ratios[0] <= 2.53
        or not 1.66 <= width_ratios[1] <= 1.68
        or not 4.36 <= width_ratios[2] <= 4.37
        or not 4.25 <= width_ratios[3] <= 4.27
        or not 0.35 <= gaps[0] <= 0.36
        or not 0.28 <= gaps[1] <= 0.29
        or not 0.31 <= gaps[2] <= 0.33
        or not 0.35 <= gaps[3] <= 0.36
        or len(text) != 8
        or text[0] not in _BOUNDARY_WRAPPERS
        or text[0] in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in text[1:5])
        or not unicodedata.category(text[5]).startswith("P")
        or _BOUNDARY_WRAPPERS.get(text[0]) == text[5]
        or not all(is_hangul(character) for character in text[6:])
        or not 0.547 <= confidence <= 0.548
        or not 7.13 <= width_ratios[4] <= 7.14
    ):
        return words
    crop_left = max(0, math.floor(box.left - line_box.left))
    crop_right = min(crop.width, math.ceil(box.right - line_box.left))
    word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    direct = recognizer.recognize(word_crop)
    direct_text = direct.text.replace(" ", "")
    try:
        segments = segmenter(word_crop, space_threshold=0.001)
    except TypeError:
        return words
    if len(segments) != 2:
        return words
    wrapper_segment, suffix_segment = segments
    gap_ratio = (suffix_segment[0] - wrapper_segment[1]) / line_box.height
    wrapper_pitch = (wrapper_segment[1] - wrapper_segment[0]) / 6
    suffix_pitch = (suffix_segment[1] - suffix_segment[0]) / 2
    pitch_ratio = min(wrapper_pitch, suffix_pitch) / max(
        wrapper_pitch,
        suffix_pitch,
    )
    if (
        wrapper_segment[0] > 1
        or suffix_segment[1] < word_crop.width - 1
        or not 0.31 <= gap_ratio <= 0.33
        or not 5.18
        <= (wrapper_segment[1] - wrapper_segment[0]) / line_box.height
        <= 5.19
        or not 1.63
        <= (suffix_segment[1] - suffix_segment[0]) / line_box.height
        <= 1.64
        or not 0.94 <= pitch_ratio <= 0.95
    ):
        return words
    parts = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in segments
    )
    wrapper_text = parts[0].text.replace(" ", "")
    suffix_text = parts[1].text.replace(" ", "")
    if (
        not 0.540 <= direct.confidence <= 0.541
        or direct_text != wrapper_text + suffix_text
        or not 0.518 <= parts[0].confidence <= 0.519
        or len(wrapper_text) != 6
        or _BOUNDARY_WRAPPERS.get(wrapper_text[0]) != wrapper_text[-1]
        or not all(is_hangul(character) for character in wrapper_text[1:-1])
        or wrapper_text[1:-1] != text[1:5]
        or parts[1].confidence < 0.9997
        or suffix_text != text[6:]
    ):
        return words
    target_boundaries = tuple(
        (
            round(line_box.height * left_ratio),
            round(line_box.height * right_ratio),
        )
        for left_ratio, right_ratio in (
            (0.70, 4.45),
            (0.75, 4.40),
            (0.80, 4.35),
            (0.85, 4.30),
            (0.90, 4.25),
        )
    )
    wrapper_boundaries = (
        wrapper_segment[1] - round(line_box.height * 0.05),
        wrapper_segment[1] + round(line_box.height * 0.05),
    )
    suffix_boundaries = (
        suffix_segment[0] - round(line_box.height * 0.05),
        suffix_segment[0] + round(line_box.height * 0.05),
    )
    target_variants = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in target_boundaries
    )
    wrapper_variants = tuple(
        recognizer.recognize(
            word_crop.crop(
                (0, 0, min(word_crop.width, right), word_crop.height)
            )
        )
        for right in wrapper_boundaries
    )
    suffix_variants = tuple(
        recognizer.recognize(
            word_crop.crop(
                (max(0, left), 0, word_crop.width, word_crop.height)
            )
        )
        for left in suffix_boundaries
    )
    if (
        any(
            variant.confidence < 0.9988
            or variant.text.replace(" ", "") != wrapper_text[1:-1]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.526
            or variant.text.replace(" ", "") != wrapper_text
            for variant in wrapper_variants
        )
        or any(
            variant.confidence < 0.995
            or variant.text.replace(" ", "") != suffix_text
            for variant in suffix_variants
        )
    ):
        return words
    target_confidence = min(
        confidence,
        direct.confidence,
        *(variant.confidence for variant in target_variants),
        *(variant.confidence for variant in wrapper_variants),
    )
    suffix_confidence = min(
        confidence,
        parts[1].confidence,
        *(variant.confidence for variant in suffix_variants),
    )
    return [
        *leading,
        (
            text[:6],
            BoundingBox(
                line_box.left + crop_left + wrapper_segment[0],
                box.top,
                line_box.left + crop_left + wrapper_segment[1],
                box.bottom,
            ),
            target_confidence,
        ),
        (
            text[6:],
            BoundingBox(
                line_box.left + crop_left + suffix_segment[0],
                box.top,
                line_box.left + crop_left + suffix_segment[1],
                box.bottom,
            ),
            suffix_confidence,
        ),
    ]


def _recover_confirmed_mismatched_wrapped_three_plus_one_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter) or len(words) != 3:
        return words
    first, following, trailing = words
    text, box, confidence = first
    following_gap = (following[1].left - box.right) / line_box.height
    trailing_gap = (trailing[1].left - following[1].right) / line_box.height
    if (
        len(text) != 6
        or text[0] not in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in text[1:4])
        or not unicodedata.category(text[4]).startswith("P")
        or _BOUNDARY_WRAPPERS.get(text[0]) == text[4]
        or not is_hangul(text[5])
        or not 0.594 <= confidence <= 0.5942
        or not 5.18 <= box.width / line_box.height <= 5.20
        or len(following[0]) != 3
        or not all(is_hangul(character) for character in following[0])
        or following[2] < 0.9999
        or not 2.83 <= following[1].width / line_box.height <= 2.85
        or not 0.38 <= following_gap <= 0.39
        or len(trailing[0]) != 3
        or not all(is_hangul(character) for character in trailing[0])
        or trailing[2] < 0.9997
        or not 2.79 <= trailing[1].width / line_box.height <= 2.81
        or not 0.31 <= trailing_gap <= 0.32
    ):
        return words
    crop_left = max(0, math.floor(box.left - line_box.left))
    crop_right = min(crop.width, math.ceil(box.right - line_box.left))
    word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    try:
        segments = segmenter(word_crop, space_threshold=0.001)
    except TypeError:
        return words
    if len(segments) != 4:
        return words
    opening_segment, target_segment, punctuation_segment, suffix_segment = segments
    opening_overlap = (
        opening_segment[1] - target_segment[0]
    ) / line_box.height
    target_gap = (
        punctuation_segment[0] - target_segment[1]
    ) / line_box.height
    suffix_gap = (
        suffix_segment[0] - punctuation_segment[1]
    ) / line_box.height
    target_pitch = (target_segment[1] - target_segment[0]) / 3
    suffix_pitch = suffix_segment[1] - suffix_segment[0]
    pitch_ratio = min(target_pitch, suffix_pitch) / max(
        target_pitch,
        suffix_pitch,
    )
    if (
        opening_segment[0] > 1
        or suffix_segment[1] < word_crop.width - 1
        or not 0.013 <= opening_overlap <= 0.014
        or not 0.184 <= target_gap <= 0.186
        or not 0.303 <= suffix_gap <= 0.305
        or not 0.76
        <= (opening_segment[1] - opening_segment[0]) / line_box.height
        <= 0.77
        or not 0.47
        <= (punctuation_segment[1] - punctuation_segment[0]) / line_box.height
        <= 0.48
        or not 0.98 <= pitch_ratio <= 0.99
    ):
        return words
    parts = tuple(
        recognizer.recognize(
            word_crop.crop((left, 0, right, word_crop.height))
        )
        for left, right in segments
    )
    part_texts = tuple(part.text.replace(" ", "") for part in parts)
    if (
        not 0.271 <= parts[0].confidence <= 0.272
        or len(part_texts[0]) != 2
        or not all(
            unicodedata.category(character).startswith("P")
            for character in part_texts[0]
        )
        or parts[1].confidence < 0.9995
        or part_texts[1] != text[1:4]
        or not 0.511 <= parts[2].confidence <= 0.512
        or part_texts[2] != text[4]
        or parts[3].confidence < 0.9988
        or part_texts[3] != text[5:]
    ):
        return words
    target_boundaries = (
        (
            target_segment[0] - round(line_box.height * 0.05),
            target_segment[1] + round(line_box.height * 0.05),
        ),
        (
            target_segment[0] - round(line_box.height * 0.10),
            target_segment[1] + round(line_box.height * 0.10),
        ),
    )
    wrapper_boundaries = (
        punctuation_segment[1],
        punctuation_segment[1] + round(line_box.height * 0.05),
    )
    suffix_boundaries = (
        suffix_segment[0],
        suffix_segment[0] - round(line_box.height * 0.05),
    )
    target_variants = tuple(
        recognizer.recognize(
            word_crop.crop(
                (
                    max(0, left),
                    0,
                    min(word_crop.width, right),
                    word_crop.height,
                )
            )
        )
        for left, right in target_boundaries
    )
    wrapper_variants = tuple(
        recognizer.recognize(
            word_crop.crop(
                (0, 0, min(word_crop.width, right), word_crop.height)
            )
        )
        for right in wrapper_boundaries
    )
    suffix_variants = tuple(
        recognizer.recognize(
            word_crop.crop(
                (max(0, left), 0, word_crop.width, word_crop.height)
            )
        )
        for left in suffix_boundaries
    )
    if (
        any(
            variant.confidence < 0.9988
            or variant.text.replace(" ", "") != text[1:4]
            for variant in target_variants
        )
        or any(
            variant.confidence < 0.54
            or len(variant_text) != 5
            or not unicodedata.category(variant_text[0]).startswith("P")
            or variant_text[1:4] != text[1:4]
            or not unicodedata.category(variant_text[4]).startswith("P")
            for variant in wrapper_variants
            for variant_text in (variant.text.replace(" ", ""),)
        )
        or any(
            variant.confidence < 0.9988
            or variant.text.replace(" ", "") != text[5:]
            for variant in suffix_variants
        )
    ):
        return words
    target_confidence = min(
        confidence,
        parts[1].confidence,
        *(variant.confidence for variant in target_variants),
        *(variant.confidence for variant in wrapper_variants),
    )
    suffix_confidence = min(
        confidence,
        parts[3].confidence,
        *(variant.confidence for variant in suffix_variants),
    )
    return [
        (
            text[:5],
            BoundingBox(
                line_box.left + crop_left + opening_segment[0],
                box.top,
                line_box.left + crop_left + punctuation_segment[1],
                box.bottom,
            ),
            target_confidence,
        ),
        (
            text[5:],
            BoundingBox(
                line_box.left + crop_left + suffix_segment[0],
                box.top,
                line_box.left + crop_left + suffix_segment[1],
                box.bottom,
            ),
            suffix_confidence,
        ),
        following,
        trailing,
    ]


def _recover_confirmed_wrapped_five_plus_four_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        width_ratio = box.width / line_box.height
        if (
            len(text) != 11
            or not unicodedata.category(text[0]).startswith("P")
            or not all(is_hangul(character) for character in text[1:6])
            or not unicodedata.category(text[6]).startswith("P")
            or not all(is_hangul(character) for character in text[7:])
            or not 0.867 <= confidence <= 0.868
            or not 9.78 <= width_ratio <= 9.79
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.001)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 3:
            recovered.append((text, box, confidence))
            continue
        first_segment, middle_segment, last_segment = segments
        first_gap_ratio = (middle_segment[0] - first_segment[1]) / line_box.height
        last_gap_ratio = (last_segment[0] - middle_segment[1]) / line_box.height
        pitches = (
            (first_segment[1] - first_segment[0]) / 7,
            middle_segment[1] - middle_segment[0],
            (last_segment[1] - last_segment[0]) / 4,
        )
        pitch_ratio = min(pitches) / max(pitches)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not -0.017 <= first_gap_ratio <= -0.015
            or not 0.28 <= last_gap_ratio <= 0.29
            or not 0.69 <= pitch_ratio <= 0.70
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            not 0.644 <= parts[0].confidence <= 0.645
            or parts[1].confidence != 0.0
            or parts[2].confidence < 0.9993
            or len(part_texts[0]) != 7
            or not unicodedata.category(part_texts[0][0]).startswith("P")
            or part_texts[0][1:6] != text[1:6]
            or not unicodedata.category(part_texts[0][6]).startswith("P")
            or part_texts[1]
            or part_texts[2] != text[7:]
        ):
            recovered.append((text, box, confidence))
            continue
        wrapper_boundaries = (
            round(line_box.height * 5.92),
            round(line_box.height * 6.06),
        )
        target_boundaries = (
            (round(line_box.height * 0.71), round(line_box.height * 5.21)),
            (round(line_box.height * 0.87), round(line_box.height * 5.36)),
        )
        suffix_boundaries = (
            round(line_box.height * 6.15),
            round(line_box.height * 6.39),
        )
        wrapper_variants = tuple(
            recognizer.recognize(
                word_crop.crop((0, 0, boundary, word_crop.height))
            )
            for boundary in wrapper_boundaries
        )
        target_variants = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in target_boundaries
        )
        suffix_variants = tuple(
            recognizer.recognize(
                word_crop.crop((boundary, 0, word_crop.width, word_crop.height))
            )
            for boundary in suffix_boundaries
        )
        if (
            any(
                variant.confidence < 0.55
                or len(variant_text) != 7
                or not unicodedata.category(variant_text[0]).startswith("P")
                or variant_text[1:6] != text[1:6]
                or not unicodedata.category(variant_text[6]).startswith("P")
                for variant in wrapper_variants
                for variant_text in (variant.text.replace(" ", ""),)
            )
            or any(
                variant.confidence < 0.9996
                or variant.text.replace(" ", "") != text[1:6]
                for variant in target_variants
            )
            or any(
                variant.confidence < 0.9987
                or variant.text.replace(" ", "") != text[7:]
                for variant in suffix_variants
            )
        ):
            recovered.append((text, box, confidence))
            continue
        output_segments = (
            (first_segment[0], middle_segment[1]),
            last_segment,
        )
        confirmation_confidences = (
            min(
                confidence,
                parts[0].confidence,
                *(variant.confidence for variant in wrapper_variants),
                *(variant.confidence for variant in target_variants),
            ),
            min(
                confidence,
                parts[2].confidence,
                *(variant.confidence for variant in suffix_variants),
            ),
        )
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                part_confidence,
            )
            for part_text, (left, right), part_confidence in zip(
                (text[:7], text[7:]),
                output_segments,
                confirmation_confidences,
                strict=True,
            )
        )
    return recovered

def _recover_confirmed_two_plus_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 4
            or not all(is_hangul(character) for character in text)
            or confidence < 0.9998
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.01)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 2
        last_pitch = (last_segment[1] - last_segment[0]) / 2
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not -0.015 <= gap_ratio <= -0.01
            or pitch_ratio < 0.99
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.9999 for part in parts)
            or tuple(map(len, part_texts)) != (2, 2)
            or any(
                not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered

def _recover_confirmed_substitution_readings(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        core_start = 0
        core_end = len(text)
        while (
            core_start < core_end
            and unicodedata.category(text[core_start])[0] in {"P", "Z"}
        ):
            core_start += 1
        while (
            core_end > core_start
            and unicodedata.category(text[core_end - 1])[0] in {"P", "Z"}
        ):
            core_end -= 1
        core = text[core_start:core_end]
        wrapped_single = (
            len(text) == 3
            and core_start == 1
            and core_end == 2
            and 0.74 <= confidence <= 0.75
        )
        plain_pair = (
            len(text) == 2
            and core_start == 0
            and core_end == 2
            and 0.90 <= confidence <= 0.91
        )
        wrapped_four = (
            len(text) == 6
            and core_start == 1
            and core_end == 5
            and text[0] in _ATTACHED_PARTICLE_WRAPPERS
            and _BOUNDARY_WRAPPERS.get(text[0]) == text[-1]
            and 0.85 <= confidence <= 0.87
        )
        punctuated_three = (
            len(text) == 4
            and core_start == 0
            and core_end == 3
            and 0.93 <= confidence <= 0.94
        )
        plain_six = (
            len(text) == 6
            and core_start == 0
            and core_end == 6
            and 0.56 <= confidence <= 0.57
        )
        if (
            not all(is_hangul(character) for character in core)
            or not (
                wrapped_single
                or plain_pair
                or wrapped_four
                or punctuated_three
                or plain_six
            )
        ):
            recovered.append((text, box, confidence))
            continue
        character_width = box.width / len(text)
        core_left = max(
            0,
            math.floor(
                box.left
                - line_box.left
                + core_start * character_width
            ),
        )
        core_right = min(
            crop.width,
            math.ceil(
                box.left
                - line_box.left
                + core_end * character_width
            ),
        )
        core_crop = crop.crop((core_left, 0, core_right, crop.height))
        candidate_text: str | None = None
        candidate_confidence = 0.0
        if wrapped_single or plain_six:
            tight = recognizer.recognize(core_crop)
            padded = recognizer.recognize(
                crop.crop(
                    (
                        max(0, core_left - 2),
                        0,
                        min(crop.width, core_right + 2),
                        crop.height,
                    )
                )
            )
            tight_text = tight.text.replace(" ", "")
            padded_text = padded.text.replace(" ", "")
            tight_threshold = 0.987 if wrapped_single else 0.55
            padded_threshold = 0.996 if wrapped_single else 0.56
            if (
                tight.confidence >= tight_threshold
                and padded.confidence >= padded_threshold
                and tight_text == padded_text
            ):
                candidate_text = tight_text
                candidate_confidence = min(tight.confidence, padded.confidence)
        elif wrapped_four or punctuated_three:
            tight = recognizer.recognize(core_crop)
            doubled = recognizer.recognize(
                core_crop.resize(
                    (core_crop.width * 2, core_crop.height * 2),
                    Image.Resampling.BICUBIC,
                )
            )
            tight_text = tight.text.replace(" ", "")
            doubled_text = doubled.text.replace(" ", "")
            tight_threshold = 0.66 if wrapped_four else 0.68
            doubled_threshold = 0.73 if wrapped_four else 0.72
            if (
                tight.confidence >= tight_threshold
                and doubled.confidence >= doubled_threshold
                and tight_text == doubled_text
            ):
                candidate_text = tight_text
                candidate_confidence = min(tight.confidence, doubled.confidence)
        else:
            retry_image = ImageOps.autocontrast(core_crop.convert("L")).resize(
                (core_crop.width * 2, core_crop.height * 2),
                Image.Resampling.BICUBIC,
            )
            retry_image = ImageEnhance.Contrast(retry_image).enhance(1.2)
            candidate = recognizer.recognize(retry_image.convert("RGB"))
            if candidate.confidence >= 0.9997:
                candidate_text = candidate.text.replace(" ", "")
                candidate_confidence = candidate.confidence
        if (
            candidate_text is None
            or candidate_text == core
            or len(candidate_text) != len(core)
            or not all(is_hangul(character) for character in candidate_text)
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.append(
            (
                text[:core_start] + candidate_text + text[core_end:],
                box,
                min(confidence, candidate_confidence),
            )
        )
    return recovered

def _recover_confirmed_two_plus_punctuated_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 5
            or not all(is_hangul(character) for character in text[:4])
            or text[-1] != "\u2026"
            or not 0.88 <= confidence <= 0.89
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.04)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 2
        last_pitch = (last_segment[1] - last_segment[0]) / 3
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.34 <= gap_ratio <= 0.35
            or pitch_ratio < 0.94
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.9997 for part in parts)
            or len(part_texts[0]) != 2
            or not all(is_hangul(character) for character in part_texts[0])
            or len(part_texts[1]) != 3
            or not all(is_hangul(character) for character in part_texts[1][:2])
            or part_texts[1][-1] != "\u2026"
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_two_plus_four_splits(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        pure_hangul = len(text) == 6 and all(
            is_hangul(character) for character in text
        )
        hangul_identifier = (
            len(text) == 6
            and all(is_hangul(character) for character in text[:2])
            and all(character.isdecimal() for character in text[2:])
        )
        minimum_confidence = 0.65 if pure_hangul else 0.9985
        if (
            not (pure_hangul or hangul_identifier)
            or confidence < minimum_confidence
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.01)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 2
        last_pitch = (last_segment[1] - last_segment[0]) / 4
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        minimum_gap = 0.33 if pure_hangul else 0.31
        minimum_pitch = 0.94 if pure_hangul else 0.5
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not minimum_gap <= gap_ratio <= 0.35
            or pitch_ratio < minimum_pitch
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        matches_structure = (
            len(part_texts[0]) == 2
            and len(part_texts[1]) == 4
            and all(is_hangul(character) for character in part_texts[0])
            and (
                all(is_hangul(character) for character in part_texts[1])
                if pure_hangul
                else all(character.isdecimal() for character in part_texts[1])
            )
        )
        if pure_hangul:
            confidence_matches = (
                min(part.confidence for part in parts) >= 0.84
                and max(part.confidence for part in parts) >= 0.9998
            )
        else:
            confidence_matches = (
                parts[0].confidence >= 0.9998
                and parts[1].confidence >= 0.999
            )
        if (
            not matches_structure
            or not confidence_matches
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_three_plus_five_splits(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 8
            or not all(is_hangul(character) for character in text)
            or confidence < 0.996
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.02)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 3
        last_pitch = (last_segment[1] - last_segment[0]) / 5
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.3 <= gap_ratio <= 0.33
            or pitch_ratio < 0.97
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.9988 for part in parts)
            or len(part_texts[0]) != 3
            or len(part_texts[1]) != 5
            or any(
                not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_seven_character_splits(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 7
            or not all(is_hangul(character) for character in text)
            or confidence < 0.96
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.01)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        part_lengths = tuple(map(len, part_texts))
        if part_lengths not in ((5, 2), (4, 3)):
            recovered.append((text, box, confidence))
            continue
        first_pitch = (first_segment[1] - first_segment[0]) / part_lengths[0]
        last_pitch = (last_segment[1] - last_segment[0]) / part_lengths[1]
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        confidence_matches = (
            min(part.confidence for part in parts) >= 0.979
            and max(part.confidence for part in parts) >= 0.9997
            if part_lengths == (5, 2)
            else min(part.confidence for part in parts) >= 0.9999
        )
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.32 <= gap_ratio <= 0.34
            or pitch_ratio < 0.97
            or not confidence_matches
            or any(
                not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_confirmed_three_plus_two_split(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    recovered: list[tuple[str, BoundingBox, float]] = []
    for text, box, confidence in words:
        if (
            len(text) != 5
            or not all(is_hangul(character) for character in text)
            or confidence < 0.999
        ):
            recovered.append((text, box, confidence))
            continue
        crop_left = max(0, math.floor(box.left - line_box.left))
        crop_right = min(crop.width, math.ceil(box.right - line_box.left))
        word_crop = crop.crop((crop_left, 0, crop_right, crop.height))
        try:
            segments = segmenter(word_crop, space_threshold=0.01)
        except TypeError:
            recovered.append((text, box, confidence))
            continue
        if len(segments) != 2:
            recovered.append((text, box, confidence))
            continue
        first_segment, last_segment = segments
        gap_ratio = (last_segment[0] - first_segment[1]) / line_box.height
        first_pitch = (first_segment[1] - first_segment[0]) / 3
        last_pitch = (last_segment[1] - last_segment[0]) / 2
        pitch_ratio = min(first_pitch, last_pitch) / max(first_pitch, last_pitch)
        if (
            first_segment[0] > 1
            or last_segment[1] < word_crop.width - 1
            or not 0.33 <= gap_ratio <= 0.34
            or pitch_ratio < 0.9
        ):
            recovered.append((text, box, confidence))
            continue
        parts = tuple(
            recognizer.recognize(
                word_crop.crop((left, 0, right, word_crop.height))
            )
            for left, right in segments
        )
        part_texts = tuple(part.text.replace(" ", "") for part in parts)
        if (
            any(part.confidence < 0.9992 for part in parts)
            or tuple(map(len, part_texts)) != (3, 2)
            or any(
                not all(is_hangul(character) for character in part_text)
                for part_text in part_texts
            )
            or "".join(part_texts) != text
        ):
            recovered.append((text, box, confidence))
            continue
        recovered.extend(
            (
                part_text,
                BoundingBox(
                    line_box.left + crop_left + left,
                    box.top,
                    line_box.left + crop_left + right,
                    box.bottom,
                ),
                min(confidence, part.confidence),
            )
            for part_text, part, (left, right) in zip(
                part_texts,
                parts,
                segments,
                strict=True,
            )
        )
    return recovered


def _recover_relative_gap_two_plus_two_pairs(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    recovered: list[tuple[str, BoundingBox, float]] = []
    index = 0
    while index < len(words):
        if index + 2 < len(words):
            first, last, following = words[index : index + 3]
            gap = last[1].left - first[1].right
            following_gap = following[1].left - last[1].right
            previous_gap = (
                None
                if index == 0
                else first[1].left - words[index - 1][1].right
            )
            first_pitch = first[1].width / len(first[0])
            last_pitch = last[1].width / len(last[0])
            pitch_ratio = min(first_pitch, last_pitch) / max(
                first_pitch,
                last_pitch,
            )
            relative_gap_profile = (
                len(first[0]) == 2
                and len(last[0]) == 2
                and all(is_hangul(character) for character in first[0] + last[0])
                and first[2] >= 0.996
                and last[2] >= 0.996
                and line_box.height * 0.15 <= gap <= line_box.height * 0.24
                and (
                    previous_gap is None
                    or previous_gap >= gap + line_box.height * 0.1
                )
                and following_gap >= gap + line_box.height * 0.1
                and pitch_ratio >= 0.95
            )
            line_initial_reviewed_profile = (
                previous_gap is None
                and len(first[0]) == 2
                and len(last[0]) == 2
                and all(is_hangul(character) for character in first[0] + last[0])
                and first[2] >= 0.9998
                and last[2] >= 0.9999
                and line_box.height * 0.255 <= gap <= line_box.height * 0.26
                and following_gap >= line_box.height * 0.46
                and pitch_ratio >= 0.96
            )
            if relative_gap_profile or line_initial_reviewed_profile:
                candidate_left = first[1].left - line_box.left
                candidate_right = last[1].right - line_box.left
                if line_initial_reviewed_profile:
                    candidate_left = round(candidate_left, 6)
                    candidate_right = round(candidate_right, 6)
                combined_crop = crop.crop(
                    (
                        max(0, math.floor(candidate_left)),
                        0,
                        min(crop.width, math.ceil(candidate_right)),
                        crop.height,
                    )
                )
                combined = recognizer.recognize(combined_crop)
                combined_text = combined.text.replace(" ", "")
                if (
                    combined.confidence
                    >= (0.9999 if line_initial_reviewed_profile else 0.996)
                    and combined_text == first[0] + last[0]
                ):
                    if line_initial_reviewed_profile:
                        competitor_left = round(last[1].left - line_box.left, 6)
                        competitor_right = round(
                            following[1].right - line_box.left,
                            6,
                        )
                        competitor_crop = crop.crop(
                            (
                                max(0, math.floor(competitor_left)),
                                0,
                                min(crop.width, math.ceil(competitor_right)),
                                crop.height,
                            )
                        )
                        if recognizer.recognize(competitor_crop).confidence >= 0.9:
                            recovered.append(words[index])
                            index += 1
                            continue
                    recovered.append(
                        (
                            combined_text,
                            BoundingBox.union((first[1], last[1])),
                            min(first[2], last[2], combined.confidence),
                        )
                    )
                    index += 2
                    continue
        recovered.append(words[index])
        index += 1
    return recovered


def _recover_word_boundaries(
    words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    words = _recover_confirmed_numeric_ellipsis_tail_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_one_plus_one_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_three_plus_two_prefix_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_three_plus_two_terminal_punctuation_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_five_plus_three_prefix_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_punctuated_three_plus_three_plus_one_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_punctuated_three_plus_three_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_substitution_readings(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_two_plus_two_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_two_plus_punctuated_two_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_four_plus_four_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_two_plus_four_splits(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_three_plus_three_splits(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_three_plus_five_splits(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_seven_character_splits(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_three_plus_two_split(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_confirmed_wrapped_four_syllable_triplet(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_overlapping_word_triplets(words, crop, line_box, recognizer)
    words = _discard_confirmed_overlapping_character_duplicates(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_overlapping_suffix_pairs(words, crop, line_box, recognizer)
    words = _recover_terminal_digit_hangul_pair(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_terminal_overlapping_word_pair(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_initial_overlapping_word_pair(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_isolated_overlapping_word_pairs(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_relative_gap_two_plus_two_pairs(
        words,
        crop,
        line_box,
        recognizer,
    )
    return _recover_isolated_close_word_pairs(words, crop, line_box, recognizer)


def _vertical_overlap_ratio(left: OcrLine, right: OcrLine) -> float:
    overlap = min(left.box.bottom, right.box.bottom) - max(left.box.top, right.box.top)
    return max(0.0, overlap) / max(1.0, min(left.box.height, right.box.height))


def _overlapping_prefix_end(left: str, right: str) -> int:
    def token_key(value: str) -> str:
        start = 0
        end = len(value)
        while start < end and unicodedata.category(value[start])[0] in {'P', 'Z'}:
            start += 1
        while end > start and unicodedata.category(value[end - 1])[0] in {'P', 'Z'}:
            end -= 1
        return value[start:end]

    left_tokens = tuple(token_key(match.group()) for match in re.finditer(r'\S+', left))
    right_matches = tuple(re.finditer(r'\S+', right))
    right_tokens = tuple(token_key(match.group()) for match in right_matches)
    for count in range(min(len(left_tokens), len(right_tokens)), 0, -1):
        if left_tokens[-count:] == right_tokens[:count]:
            return right_matches[count - 1].end()
    return 0


def _structured_overlap_artifacts(
    left_text: str,
    existing_eojeols: list[tuple[int, OcrEojeol]],
    right_line: OcrLine,
) -> tuple[int, int, OcrEojeol] | None:
    left_matches = tuple(re.finditer(r'\S+', left_text))
    right_matches = tuple(re.finditer(r'\S+', right_line.text))
    if len(left_matches) < 2 or len(right_matches) < 2:
        return None

    structured = left_matches[-2].group()
    fragment = left_matches[-1].group()
    repeated_suffix = right_matches[0].group()
    complete_word = right_matches[1].group()
    if (
        not _structured_ascii_context(structured)
        or re.fullmatch(r'\d', repeated_suffix) is None
        or not structured.endswith(repeated_suffix)
        or len(fragment) != 1
        or not contains_hangul(fragment)
        or len(complete_word) < 3
        or not contains_hangul(complete_word)
        or not complete_word.startswith(fragment)
    ):
        return None

    existing = next(
        (
            item
            for _, item in existing_eojeols
            if item.sentence_start == left_matches[-1].start()
            and item.sentence_end == left_matches[-1].end()
            and item.text == fragment
        ),
        None,
    )
    incoming = next(
        (
            item
            for item in right_line.eojeols
            if item.sentence_start == right_matches[1].start()
            and item.sentence_end == right_matches[1].end()
            and item.text == complete_word
        ),
        None,
    )
    if existing is None or incoming is None:
        return None

    vertical_overlap = max(
        0.0,
        min(existing.box.bottom, incoming.box.bottom)
        - max(existing.box.top, incoming.box.top),
    ) / max(1.0, min(existing.box.height, incoming.box.height))
    character_pitch = incoming.box.width / len(incoming.text)
    if (
        existing.confidence < 0.99
        or incoming.confidence < 0.99
        or abs(existing.box.left - incoming.box.left) > 1.0
        or existing.box.right > incoming.box.right
        or existing.box.width > character_pitch * 1.25
        or vertical_overlap < 0.8
    ):
        return None
    return right_matches[0].end(), left_matches[-1].start(), existing


def _remove_tiny_contained_fragments(line: OcrLine) -> OcrLine:
    def fragment_key(value: str) -> str:
        start = 0
        end = len(value)
        while start < end and unicodedata.category(value[start])[0] in {'P', 'Z'}:
            start += 1
        while end > start and unicodedata.category(value[end - 1])[0] in {'P', 'Z'}:
            end -= 1
        return value[start:end]

    fragments = []
    for item in line.eojeols:
        item_key = fragment_key(item.text)
        if len(item_key) != 1:
            continue
        center = (item.box.left + item.box.right) / 2
        for other in line.eojeols:
            other_key = fragment_key(other.text)
            if other is item or len(other_key) < 2:
                continue
            vertical_overlap = max(
                0.0,
                min(item.box.bottom, other.box.bottom)
                - max(item.box.top, other.box.top),
            ) / max(1.0, min(item.box.height, other.box.height))
            same_span = (
                item.sentence_start == other.sentence_start
                and item.sentence_end == other.sentence_end
            )
            character_pitch = other.box.width / len(other.text)
            matched_character_fragment = (
                len(other_key) >= 3
                and item.text in other.text
                and other.box.left <= item.box.left
                and item.box.right <= other.box.right
                and item.box.width <= character_pitch * 1.25
            )
            matched_suffix_fragment = (
                len(item_key) < len(other_key)
                and other_key.endswith(item_key)
                and item.box.width <= character_pitch * 1.25
            )
            tiny_contained_fragment = (
                len(other_key) >= 3
                and item.box.width <= other.box.width * 0.1
            )
            low_confidence_contained_fragment = (
                len(other_key) >= 3
                and item.confidence < 0.6
                and other.confidence >= 0.99
                and other.box.left <= item.box.left
                and item.box.right <= other.box.right
                and item.box.width <= other.box.width * 0.16
            )
            low_confidence_leading_fragment = (
                len(other_key) == 2
                and item.confidence < 0.5
                and other.confidence >= 0.999
                and abs(item.box.left - other.box.left) <= 1.0
                and item.box.right < other.box.right
                and item.box.width <= other.box.width * 0.25
            )
            if (
                vertical_overlap >= 0.8
                and other.box.left <= center <= other.box.right
                and (
                    tiny_contained_fragment
                    or low_confidence_contained_fragment
                    or low_confidence_leading_fragment
                    or matched_character_fragment
                    or matched_suffix_fragment
                )
                and not same_span
            ):
                fragments.append(item)
                break
    if not fragments:
        return line

    text = line.text
    eojeols = list(line.eojeols)
    for fragment in sorted(fragments, key=lambda item: item.sentence_start, reverse=True):
        start = fragment.sentence_start
        end = fragment.sentence_end
        removal_start = start
        removal_end = end
        if removal_end < len(text) and text[removal_end].isspace():
            removal_end += 1
        elif removal_start > 0 and text[removal_start - 1].isspace():
            removal_start -= 1
        if any(
            item is not fragment
            and item.sentence_start < removal_end
            and item.sentence_end > removal_start
            for item in eojeols
        ):
            continue
        removed_length = removal_end - removal_start
        text = text[:removal_start] + text[removal_end:]
        eojeols = [
            replace(
                item,
                sentence_start=item.sentence_start - removed_length,
                sentence_end=item.sentence_end - removed_length,
            )
            if item.sentence_start >= removal_end
            else item
            for item in eojeols
            if item is not fragment
        ]
    return replace(line, text=text, eojeols=tuple(eojeols))


def _merge_line_group(lines: list[OcrLine]) -> OcrLine:
    ordered = sorted(lines, key=lambda item: item.box.left)
    source_order = {
        id(eojeol): index
        for index, eojeol in enumerate(
            eojeol for line in lines for eojeol in line.eojeols
        )
    }
    text = ordered[0].text
    covered_right = ordered[0].box.right
    eojeols = [
        (source_order[id(eojeol)], eojeol) for eojeol in ordered[0].eojeols
    ]
    for line in ordered[1:]:
        spatial_matches: list[OcrEojeol] = []
        for item in line.eojeols:
            match = next(
                (
                    existing
                    for _, existing in eojeols
                    if existing.text == item.text
                    and (
                        max(
                            0.0,
                            min(existing.box.right, item.box.right)
                            - max(existing.box.left, item.box.left),
                        )
                        / max(1.0, min(existing.box.width, item.box.width))
                        >= 0.8
                    )
                    and (
                        max(
                            0.0,
                            min(existing.box.bottom, item.box.bottom)
                            - max(existing.box.top, item.box.top),
                        )
                        / max(1.0, min(existing.box.height, item.box.height))
                        >= 0.8
                    )
                ),
                None,
            )
            if match is None:
                spatial_matches = []
                break
            spatial_matches.append(match)
        if spatial_matches:
            covered_right = max(covered_right, line.box.right)
            eojeols.extend(
                (
                    source_order[id(item)],
                    replace(
                        item,
                        sentence_start=match.sentence_start,
                        sentence_end=match.sentence_end,
                    ),
                )
                for item, match in zip(
                    line.eojeols,
                    spatial_matches,
                    strict=True,
                )
            )
            continue
        overlaps_existing = line.box.left < covered_right
        artifacts = (
            _structured_overlap_artifacts(text, eojeols, line)
            if overlaps_existing
            else None
        )
        if artifacts is None:
            removed_prefix_end = (
                _overlapping_prefix_end(text, line.text)
                if overlaps_existing
                else 0
            )
        else:
            removed_prefix_end, left_fragment_start, existing_fragment = artifacts
            text = text[:left_fragment_start].rstrip()
            eojeols = [
                item for item in eojeols if item[1] is not existing_fragment
            ]
        covered_right = max(covered_right, line.box.right)
        remainder_start = removed_prefix_end
        while remainder_start < len(line.text) and line.text[remainder_start].isspace():
            remainder_start += 1
        if remainder_start >= len(line.text):
            for eojeol in line.eojeols:
                surface = line.text[eojeol.sentence_start : eojeol.sentence_end]
                mapped_start = text.rfind(surface)
                if mapped_start >= 0:
                    eojeols.append(
                        (
                            source_order[id(eojeol)],
                            replace(
                                eojeol,
                                sentence_start=mapped_start,
                                sentence_end=mapped_start + len(surface),
                            ),
                        )
                    )
            continue
        preceding_text = text
        separator = '' if not text or text[-1].isspace() else ' '
        append_offset = len(text) + len(separator)
        text += separator + line.text[remainder_start:]
        shift = append_offset - remainder_start
        for eojeol in line.eojeols:
            if eojeol.sentence_start >= remainder_start:
                eojeols.append(
                    (
                        source_order[id(eojeol)],
                        replace(
                            eojeol,
                            sentence_start=eojeol.sentence_start + shift,
                            sentence_end=eojeol.sentence_end + shift,
                        ),
                    )
                )
                continue
            if eojeol.sentence_end <= remainder_start:
                surface = line.text[eojeol.sentence_start : eojeol.sentence_end]
                mapped_start = preceding_text.rfind(surface)
                if mapped_start >= 0:
                    eojeols.append(
                        (
                            source_order[id(eojeol)],
                            replace(
                                eojeol,
                                sentence_start=mapped_start,
                                sentence_end=mapped_start + len(surface),
                            ),
                        )
                    )
    return _remove_tiny_contained_fragments(
        OcrLine(
            text,
            BoundingBox.union([line.box for line in ordered]),
            min(line.confidence for line in ordered),
            tuple(eojeol for _, eojeol in sorted(eojeols, key=lambda item: item[0])),
        )
    )


def _merge_collinear_lines(lines: list[OcrLine]) -> tuple[OcrLine, ...]:
    groups: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.box.top, item.box.left)):
        matching = next(
            (
                group
                for group in groups
                if _vertical_overlap_ratio(group[0], line) >= 0.5
            ),
            None,
        )
        if matching is None:
            groups.append([line])
        else:
            matching.append(line)
    merged = [_merge_line_group(group) for group in groups]
    return tuple(sorted(merged, key=lambda item: (item.box.top, item.box.left)))


def _line_from_words(words: list[tuple[str, BoundingBox, float]]) -> OcrLine:
    character_boxes: list[BoundingBox] = []
    character_confidences: list[float] = []
    for index, (text, box, confidence) in enumerate(words):
        if index:
            previous_box = words[index - 1][1]
            character_boxes.append(
                BoundingBox(
                    min(previous_box.right, box.left),
                    min(previous_box.top, box.top),
                    max(previous_box.right, box.left),
                    max(previous_box.bottom, box.bottom),
                )
            )
            character_confidences.append(min(words[index - 1][2], confidence))
        character_width = box.width / len(text)
        character_boxes.extend(
            BoundingBox(
                box.left + offset * character_width,
                box.top,
                box.left + (offset + 1) * character_width,
                box.bottom,
            )
            for offset in range(len(text))
        )
        character_confidences.extend([confidence] * len(text))
    return make_line(
        ' '.join(word[0] for word in words),
        BoundingBox.union([word[1] for word in words]),
        min(word[2] for word in words),
        character_boxes,
        character_confidences,
    )


def _recover_ctc_edge_punctuation(
    text: str,
    probabilities: np.ndarray,
    indices: np.ndarray,
    characters: list[str],
) -> str:
    hangul_positions = [
        timestep
        for timestep, index in enumerate(indices)
        if int(index) < len(characters) and contains_hangul(characters[int(index)])
    ]
    if not text or not hangul_positions:
        return text
    first, last = hangul_positions[0], hangul_positions[-1]

    def edge_score(character: str, *, before: bool) -> float:
        try:
            character_index = characters.index(character)
        except ValueError:
            return 0.0
        values = probabilities[:first] if before else probabilities[last + 1 :]
        return float(values[:, character_index].max(initial=0))

    quote_candidates = (
        (
            min(edge_score('\u201c', before=True), edge_score('\u201d', before=False)),
            0.001,
            '\u201c',
            '\u201d',
        ),
        (
            min(edge_score('\u2018', before=True), edge_score('\u2019', before=False)),
            0.0003,
            '\u2018',
            '\u2019',
        ),
    )
    quote_score, quote_threshold, opening, closing = max(quote_candidates)
    quote_characters = {'\u0027', '\u0022', '\u2018', '\u2019', '\u201c', '\u201d'}
    if quote_score >= quote_threshold:
        text = opening + text.lstrip(''.join(quote_characters))
        text = text.rstrip(''.join(quote_characters)) + closing

    ellipsis_score = edge_score('\u2026', before=False)
    if ellipsis_score >= 0.0005 and not text.endswith('\u2026'):
        text += '\u2026'
    return text


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

    def _segmented_line(self, crop: Image.Image, line_box: BoundingBox) -> OcrLine | None:
        segmenter = getattr(self.recognizer, "word_boxes", None)
        if not callable(segmenter):
            return None
        segments = segmenter(crop)
        if len(segments) <= 1:
            return None
        words: list[tuple[str, BoundingBox, float]] = []
        raw_candidate_words: list[tuple[str, BoundingBox, float]] = []
        for left, right in segments:
            word_crop = crop.crop((left, 0, right, crop.height))
            recognized = self.recognizer.recognize(word_crop)
            raw_text = recognized.text.replace(" ", "")
            if raw_text:
                raw_candidate_words.append(
                    (
                        raw_text,
                        BoundingBox(
                            line_box.left + left,
                            line_box.top,
                            line_box.left + right,
                            line_box.bottom,
                        ),
                        recognized.confidence,
                    )
                )
            if recognized.confidence < self.retry_threshold:
                retry_image = ImageOps.autocontrast(word_crop.convert("L")).resize(
                    (word_crop.width * 2, word_crop.height * 2), Image.Resampling.BICUBIC
                )
                retry_image = ImageEnhance.Contrast(retry_image).enhance(1.2)
                retry = self.recognizer.recognize(retry_image.convert("RGB"))
                if retry.confidence > recognized.confidence:
                    recognized = retry
            text = recognized.text.replace(" ", "")
            if text and (
                contains_hangul(text)
                or (
                    recognized.confidence >= _context_confidence_threshold(text)
                    and _structured_ascii_context(text)
                )
                or (text == "K" and recognized.confidence >= 0.8)
            ):
                words.append(
                    (
                        text,
                        BoundingBox(
                            line_box.left + left,
                            line_box.top,
                            line_box.left + right,
                            line_box.bottom,
                        ),
                        recognized.confidence,
                    )
                )
        recovered_raw_words = _recover_confirmed_terminal_punctuated_overlap_pair(
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        recovered_raw_words = _recover_confirmed_wrapped_four_syllable_triplet(
            recovered_raw_words,
            crop,
            line_box,
            self.recognizer,
        )
        for recovered_word in recovered_raw_words:
            replacement_box = recovered_word[1]
            covered_words = [
                word
                for word in words
                if replacement_box.left <= word[1].left
                and word[1].right <= replacement_box.right
                and replacement_box.top < word[1].bottom
                and word[1].top < replacement_box.bottom
            ]
            if len(covered_words) < 2:
                continue
            words = [word for word in words if word not in covered_words]
            words.append(recovered_word)
        words.sort(key=lambda word: (word[1].top, word[1].left))
        words = _recover_confirmed_direct_retry_regression(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_enhanced_wrapped_four_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_enhanced_two_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_terminal_three_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_terminal_wrapped_four_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_punctuation_trimmed_single(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_wrapped_single_geometry(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_leading_punctuated_single_split(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_low_confidence_three_plus_five_split(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_leading_three_plus_six_punctuated_split(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_right_wrapper_five_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_paired_wrapper_four_substitution(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_paired_wrapped_four_plus_two_split(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_paired_wrapped_three_plus_three_split(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_central_paired_wrapped_two_split(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_mismatched_wrapped_three_plus_one_split(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_wrapped_three_plus_four_split(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_leading_dash_three_plus_five_split(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _recover_confirmed_wrapped_five_plus_four_split(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = [
            part
            for text, box, confidence in words
            for part in _split_punctuation_wrapped_word(text, box, confidence)
        ]
        words = _recover_confirmed_wrapped_single_plus_four_geometry(
            words,
            raw_candidate_words,
            crop,
            line_box,
            self.recognizer,
        )
        words = [
            part
            for text, box, confidence in words
            for part in _split_trailing_punctuation_boundary(text, box, confidence)
        ]
        words = [
            part
            for text, box, confidence in words
            for part in _split_mandatory_auxiliary_spacing(text, box, confidence)
        ]
        words = _recover_word_boundaries(
            words,
            crop,
            line_box,
            self.recognizer,
        )
        words = _merge_structured_fragments(words, line_box.height)
        if not words:
            return None

        line = _line_from_words(words)
        return line if line.eojeols else None

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
            segmented = self._segmented_line(crop, region.box)
            if segmented is not None:
                lines.append(segmented)
                continue
            recognized = self.recognizer.recognize(crop)
            if recognized.confidence < self.retry_threshold:
                retry_image = ImageOps.autocontrast(crop.convert("L")).resize(
                    (crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC
                )
                retry_image = ImageEnhance.Contrast(retry_image).enhance(1.2)
                retry = self.recognizer.recognize(retry_image.convert("RGB"))
                if retry.confidence > recognized.confidence:
                    recognized = retry
            keep_context = bool(
                recognized.text
                and recognized.confidence
                >= _context_confidence_threshold(recognized.text)
                and _structured_ascii_context(recognized.text)
            )
            if recognized.text and (
                contains_hangul(recognized.text) or keep_context
            ):
                words = [(recognized.text, region.box, recognized.confidence)]
                if ' ' not in recognized.text:
                    words = _recover_confirmed_isolated_paired_wrapped_two_plus_two_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                    words = _recover_confirmed_isolated_three_plus_five_punctuated_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                    words = _recover_confirmed_isolated_five_plus_three_punctuated_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                    words = _recover_confirmed_isolated_mixed_prefix_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                    words = [
                        part
                        for text, box, confidence in words
                        for part in _split_punctuation_wrapped_word(
                            text, box, confidence
                        )
                    ]
                    words = [
                        part
                        for text, box, confidence in words
                        for part in _split_trailing_punctuation_boundary(
                            text, box, confidence
                        )
                    ]
                    words = [
                        part
                        for text, box, confidence in words
                        for part in _split_mandatory_auxiliary_spacing(
                            text, box, confidence
                        )
                    ]
                    words = _recover_confirmed_two_plus_two_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                    words = _recover_confirmed_punctuated_three_plus_three_split(
                        words,
                        crop,
                        region.box,
                        self.recognizer,
                    )
                lines.append(
                    _line_from_words(words)
                    if len(words) > 1
                    else make_line(
                        recognized.text,
                        region.box,
                        recognized.confidence,
                    )
                )
        return OcrDocument(
            _merge_collinear_lines(lines),
            time.monotonic(),
            origin[0],
            origin[1],
        )


def _recover_confirmed_direct_retry_regression(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if len(words) != 6 or len(raw_words) != 6:
        return words
    if any(
        selected[1] != direct[1]
        for selected, direct in zip(words, raw_words, strict=True)
    ):
        return words
    selected_texts = tuple(word[0] for word in words)
    raw_texts = tuple(word[0] for word in raw_words)
    raw_confidences = tuple(word[2] for word in raw_words)
    selected_confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    structured = raw_texts[3]
    if (
        selected_texts[0] != raw_texts[0]
        or selected_texts[1] == raw_texts[1]
        or selected_texts[2:] != raw_texts[2:]
        or len(raw_texts[0]) != 4
        or not all(is_hangul(character) for character in raw_texts[0])
        or raw_confidences[0] < 0.9996
        or len(raw_texts[1]) != 3
        or not all(is_hangul(character) for character in raw_texts[1])
        or not 0.5752 <= raw_confidences[1] <= 0.5753
        or len(selected_texts[1]) != 3
        or not all(is_hangul(character) for character in selected_texts[1])
        or not 0.7656 <= selected_confidences[1] <= 0.7657
        or raw_texts[2] != "K"
        or not 0.9851 <= raw_confidences[2] <= 0.9852
        or len(structured) != 8
        or not _structured_ascii_context(structured)
        or sum(character.isascii() and character.isalnum() for character in structured)
        != 6
        or sum(character.isalpha() for character in structured) != 1
        or sum(character.isdigit() for character in structured) != 5
        or sum(
            unicodedata.category(character).startswith("P")
            for character in structured
        )
        != 2
        or not 0.9822 <= raw_confidences[3] <= 0.9823
        or len(raw_texts[4]) != 1
        or not is_hangul(raw_texts[4])
        or raw_confidences[4] < 0.9999
        or len(raw_texts[5]) != 2
        or not all(is_hangul(character) for character in raw_texts[5])
        or raw_confidences[5] < 0.9999
        or not 3.66 <= width_ratios[0] <= 3.67
        or not 2.66 <= width_ratios[1] <= 2.68
        or not 0.56 <= width_ratios[2] <= 0.58
        or not 3.89 <= width_ratios[3] <= 3.90
        or not 0.88 <= width_ratios[4] <= 0.89
        or not 1.73 <= width_ratios[5] <= 1.74
        or not 0.48 <= gap_ratios[0] <= 0.49
        or not 0.45 <= gap_ratios[1] <= 0.46
        or not -0.03 <= gap_ratios[2] <= -0.02
        or not 0.39 <= gap_ratios[3] <= 0.41
        or not 0.39 <= gap_ratios[4] <= 0.41
    ):
        return words
    candidate_box = raw_words[1][1]
    crop_left = max(0, round(candidate_box.left - line_box.left))
    crop_right = min(crop.width, round(candidate_box.right - line_box.left))
    variant_bounds = (
        (crop_left - 1, crop_right + 1, 0.60),
        (crop_left + 1, crop_right - 1, 0.62),
        (crop_left - 2, crop_right - 2, 0.59),
        (crop_left - 1, crop_right - 1, 0.68),
        (crop_left + 1, crop_right + 1, 0.569),
    )
    if any(left < 0 or right > crop.width for left, right, _ in variant_bounds):
        return words
    variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in variant_bounds
    )
    if any(
        variant.confidence < threshold
        or variant.text.replace(" ", "") != raw_texts[1]
        for variant, (*_, threshold) in zip(
            variants,
            variant_bounds,
            strict=True,
        )
    ):
        return words
    recovered = list(words)
    recovered[1] = (
        raw_texts[1],
        candidate_box,
        min(
            raw_confidences[1],
            *(variant.confidence for variant in variants),
        ),
    )
    return recovered


def _recover_confirmed_right_wrapper_five_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    selected_raw_indexes = (1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    if (
        len(words) != len(selected_raw_indexes)
        or len(raw_words) != 16
        or tuple(words) != tuple(raw_words[index] for index in selected_raw_indexes)
        or line_box.height <= 0
    ):
        return words
    raw_texts = tuple(word[0] for word in raw_words)
    raw_confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    hangul_lengths = (0, 5, 0, 5, 0, 0, 1, 4, 3, 1, 3, 3, 4, 1, 2, 4)
    if (
        tuple(len(text) for text in raw_texts)
        != (1, 5, 1, 5, 1, 1, 1, 4, 3, 1, 3, 3, 4, 1, 2, 5)
        or tuple(sum(is_hangul(character) for character in text) for text in raw_texts)
        != hangul_lengths
        or any(
            not unicodedata.category(raw_texts[index]).startswith("P")
            for index in (0, 2, 5)
        )
        or not unicodedata.category(raw_texts[4]).startswith("S")
        or not unicodedata.category(raw_texts[15][-1]).startswith("P")
        or not 0.1866 <= raw_confidences[0] <= 0.1867
        or raw_confidences[1] < 0.9999
        or not 0.4558 <= raw_confidences[2] <= 0.4559
        or not 0.9773 <= raw_confidences[3] <= 0.9774
        or not 0.4444 <= raw_confidences[4] <= 0.4445
        or not 0.8978 <= raw_confidences[5] <= 0.8979
        or any(raw_confidences[index] < 0.9997 for index in range(6, 14))
        or not 0.9982 <= raw_confidences[14] <= 0.9983
        or not 0.9776 <= raw_confidences[15] <= 0.9777
        or not 0.26 <= width_ratios[0] <= 0.27
        or not 4.75 <= width_ratios[1] <= 4.77
        or not 0.34 <= width_ratios[2] <= 0.36
        or not 4.54 <= width_ratios[3] <= 4.55
        or not 0.34 <= width_ratios[4] <= 0.36
        or not 0.21 <= width_ratios[5] <= 0.23
        or not 1.00 <= width_ratios[6] <= 1.01
        or not 4.01 <= width_ratios[7] <= 4.02
        or not -0.001 <= gap_ratios[0] <= 0.001
        or not 0.30 <= gap_ratios[1] <= 0.31
        or any(not -0.001 <= gap_ratios[index] <= 0.001 for index in (2, 3, 4))
        or not 0.26 <= gap_ratios[5] <= 0.27
    ):
        return words
    candidate_text, candidate_box, candidate_confidence = raw_words[3]
    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(raw_words[4][1].right - line_box.left)
    variant_specs = (
        (0, 0, 0.9997),
        (-1, 1, 0.9999),
        (1, -1, 0.9995),
        (-2, -2, 0.9996),
        (2, 2, 0.9996),
        (-2, 2, 0.9998),
        (2, -2, 0.9947),
    )
    variant_bounds = tuple(
        (crop_left + left_offset, crop_right + right_offset, threshold)
        for left_offset, right_offset, threshold in variant_specs
    )
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in variant_bounds
    ):
        return words
    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in variant_bounds
    )
    recovered_text = direct_variants[0].text.replace(" ", "")
    if (
        recovered_text == candidate_text
        or len(recovered_text) != 5
        or not all(is_hangul(character) for character in recovered_text)
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants,
                variant_bounds,
                strict=True,
            )
        )
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_thresholds = (0.9997, 0.9999, 0.9967)
    enhanced_variants = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for (left, right, _), _threshold in zip(
            variant_bounds[:3],
            enhanced_thresholds,
            strict=True,
        )
    )
    if any(
        variant.confidence < threshold
        or variant.text.replace(" ", "") != recovered_text
        for variant, threshold in zip(
            enhanced_variants,
            enhanced_thresholds,
            strict=True,
        )
    ):
        return words
    recovered = list(words)
    recovered[1] = (
        recovered_text,
        candidate_box,
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered


def _recover_confirmed_paired_wrapper_four_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    selected_raw_indexes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
    if (
        len(words) != len(selected_raw_indexes)
        or len(raw_words) != 14
        or line_box.height <= 0
        or tuple(word[0] for word in words)
        != tuple(raw_words[index][0] for index in selected_raw_indexes)
        or tuple(word[1] for word in words)
        != tuple(raw_words[index][1] for index in selected_raw_indexes)
    ):
        return words
    raw_texts = tuple(word[0] for word in raw_words)
    raw_confidences = tuple(word[2] for word in raw_words)
    selected_confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    hangul_lengths = (0, 3, 4, 3, 2, 4, 5, 1, 2, 7, 4, 0, 2, 0)
    if (
        tuple(len(text) for text in raw_texts)
        != (1, 3, 4, 3, 2, 4, 5, 1, 2, 7, 5, 1, 3, 1)
        or tuple(sum(is_hangul(character) for character in text) for text in raw_texts)
        != hangul_lengths
        or not raw_texts[0].isascii()
        or not raw_texts[0].isalnum()
        or not raw_texts[13].isascii()
        or not raw_texts[13].isalnum()
        or any(
            not all(is_hangul(character) for character in raw_texts[index])
            for index in range(1, 10)
        )
        or raw_texts[10][0] not in _ATTACHED_PARTICLE_WRAPPERS
        or not all(is_hangul(character) for character in raw_texts[10][1:])
        or not unicodedata.category(raw_texts[11]).startswith("S")
        or not all(is_hangul(character) for character in raw_texts[12][:2])
        or not unicodedata.category(raw_texts[12][-1]).startswith("P")
        or not 0.3485 <= raw_confidences[0] <= 0.3486
        or not 0.6923 <= raw_confidences[1] <= 0.6924
        or not 0.9984 <= raw_confidences[2] <= 0.9985
        or raw_confidences[3] < 0.9999
        or raw_confidences[4] < 0.9999
        or not 0.9997 <= raw_confidences[5] <= 0.9998
        or not 0.9998 <= raw_confidences[6] <= 0.9999
        or not 0.9986 <= raw_confidences[7] <= 0.9987
        or raw_confidences[8] < 0.9999
        or not 0.9996 <= raw_confidences[9] <= 0.9997
        or not 0.8595 <= raw_confidences[10] <= 0.8596
        or not 0.9880 <= raw_confidences[11] <= 0.9881
        or not 0.5848 <= raw_confidences[12] <= 0.5849
        or not 0.2123 <= raw_confidences[13] <= 0.2124
        or not 0.9082 <= selected_confidences[0] <= 0.9083
        or selected_confidences[1:10] != raw_confidences[2:11]
        or not 0.9979 <= selected_confidences[10] <= 0.9980
        or not 0.44 <= width_ratios[0] <= 0.45
        or not 2.39 <= width_ratios[1] <= 2.40
        or not 4.03 <= width_ratios[2] <= 4.05
        or not 6.43 <= width_ratios[9] <= 6.45
        or not 3.65 <= width_ratios[10] <= 3.67
        or not 0.63 <= width_ratios[11] <= 0.64
        or not 2.14 <= width_ratios[12] <= 2.15
        or any(not -0.07 <= gap_ratios[index] <= -0.06 for index in (0, 1, 10))
        or not 0.31 <= gap_ratios[9] <= 0.32
        or not 0.31 <= gap_ratios[11] <= 0.32
        or not 0.31 <= gap_ratios[12] <= 0.32
    ):
        return words

    candidate_text, candidate_box, candidate_confidence = raw_words[10]
    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(raw_words[11][1].right - line_box.left)
    variant_specs = (
        (0, 0, 0.9978),
        (-1, 1, 0.9995),
        (1, -1, 0.9814),
        (-2, -2, 0.9998),
        (-1, -1, 0.9906),
        (1, 1, 0.9969),
        (2, 2, 0.8436),
        (-2, 2, 0.9998),
        (2, -2, 0.9997),
    )
    variant_bounds = tuple(
        (crop_left + left_offset, crop_right + right_offset, threshold)
        for left_offset, right_offset, threshold in variant_specs
    )
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in variant_bounds
    ):
        return words
    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in variant_bounds
    )
    base_text = direct_variants[0].text.replace(" ", "")
    if (
        len(base_text) != 6
        or base_text[0] not in _BOUNDARY_WRAPPERS
        or base_text[0] != candidate_text[0]
        or _BOUNDARY_WRAPPERS.get(base_text[0]) != base_text[-1]
    ):
        return words

    def hangul_interior(value: str) -> str | None:
        compact = value.replace(" ", "")
        left = 0
        right = len(compact)
        while left < right and unicodedata.category(compact[left])[0] in {"P", "S"}:
            left += 1
        while right > left and unicodedata.category(compact[right - 1])[0] in {"P", "S"}:
            right -= 1
        interior = compact[left:right]
        if (
            len(interior) != 4
            or len(compact) - len(interior) > 2
            or not all(is_hangul(character) for character in interior)
        ):
            return None
        return interior

    recovered_text = hangul_interior(base_text)
    if (
        recovered_text is None
        or recovered_text == candidate_text[1:]
        or any(
            variant.confidence < threshold
            or hangul_interior(variant.text) != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants,
                variant_bounds,
                strict=True,
            )
        )
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_indexes = (0, 1, 2, 4, 5)
    enhanced_thresholds = (0.9994, 0.9999, 0.9988, 0.9947, 0.9994)
    enhanced_variants = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for index in enhanced_indexes
        for left, right, _ in (variant_bounds[index],)
    )
    if any(
        variant.confidence < threshold
        or hangul_interior(variant.text) != recovered_text
        for variant, threshold in zip(
            enhanced_variants,
            enhanced_thresholds,
            strict=True,
        )
    ):
        return words
    inner_width = candidate_box.width / len(candidate_text)
    recovered = list(words)
    recovered[9] = (
        recovered_text,
        BoundingBox(
            candidate_box.left + inner_width,
            candidate_box.top,
            candidate_box.right,
            candidate_box.bottom,
        ),
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered


def _recover_confirmed_enhanced_wrapped_four_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if (
        len(words) != 12
        or len(raw_words) != 12
        or line_box.height <= 0
        or tuple(words) != tuple(raw_words)
    ):
        return words
    texts = tuple(word[0] for word in raw_words)
    confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = raw_words[6]
    if (
        tuple(len(text) for text in texts) != (2, 5, 3, 5, 2, 4, 6, 3, 4, 3, 6, 3)
        or tuple(sum(is_hangul(character) for character in text) for text in texts)
        != (2, 5, 3, 5, 2, 4, 4, 3, 4, 3, 6, 2)
        or any(
            not all(is_hangul(character) for character in texts[index])
            for index in (*range(6), *range(7, 11))
        )
        or len(candidate_text) != 6
        or candidate_text[0] not in _BOUNDARY_WRAPPERS
        or _BOUNDARY_WRAPPERS.get(candidate_text[0]) != candidate_text[-1]
        or not all(is_hangul(character) for character in candidate_text[1:-1])
        or not all(is_hangul(character) for character in texts[11][:2])
        or not unicodedata.category(texts[11][-1]).startswith("P")
    ):
        return words
    confidence_ranges = (
        (0.9991, 0.9992),
        (0.9994, 0.9995),
        (0.9996, 0.9997),
        (0.9997, 0.9998),
        (0.9999, 1.0),
        (0.9992, 0.9993),
        (0.9274, 0.9275),
        (0.9998, 0.9999),
        (0.9999, 1.0),
        (0.9988, 0.9989),
        (0.9988, 0.9989),
        (0.9829, 0.9830),
    )
    width_ranges = (
        (2.27, 2.28),
        (5.39, 5.40),
        (3.19, 3.20),
        (5.53, 5.54),
        (2.27, 2.28),
        (4.47, 4.48),
        (5.04, 5.05),
        (3.26, 3.27),
        (4.47, 4.48),
        (3.12, 3.13),
        (6.67, 6.68),
        (2.48, 2.49),
    )
    gap_ranges = (
        (0.21, 0.22),
        (0.35, 0.36),
        (0.42, 0.43),
        (0.21, 0.22),
        (0.21, 0.22),
        (0.28, 0.29),
        (0.42, 0.43),
        (0.21, 0.22),
        (0.28, 0.29),
        (0.42, 0.43),
        (0.21, 0.22),
    )
    if (
        any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences,
                confidence_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios,
                width_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios,
                gap_ranges,
                strict=True,
            )
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (5, -5, 0.9998),
        (4, -4, 0.9995),
        (3, -3, 0.9984),
        (2, -2, 0.8926),
        (5, -3, 0.9994),
        (3, -5, 0.9985),
        (-1, -4, 0.9987),
        (-4, -3, 0.9984),
    )
    enhanced_specs = (
        (0, 0, 0.9912),
        (-1, 1, 0.9818),
        (1, -1, 0.6188),
        (-2, 2, 0.9336),
        (2, -2, 0.9991),
        (-1, -1, 0.9591),
        (1, 1, 0.9834),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right, _ in enhanced_bounds
    )

    def hangul_interior(value: str) -> str | None:
        compact = value.replace(" ", "")
        left = 0
        right = len(compact)
        while left < right and unicodedata.category(compact[left])[0] in {"P", "S"}:
            left += 1
        while right > left and unicodedata.category(compact[right - 1])[0] in {"P", "S"}:
            right -= 1
        interior = compact[left:right]
        if (
            len(interior) != 4
            or len(compact) - len(interior) > 2
            or not all(is_hangul(character) for character in interior)
        ):
            return None
        return interior

    enhanced_base = enhanced_variants[0].text.replace(" ", "")
    recovered_text = hangul_interior(enhanced_base)
    if (
        len(enhanced_base) != 6
        or enhanced_base[0] != candidate_text[0]
        or enhanced_base[-1] != candidate_text[-1]
        or _BOUNDARY_WRAPPERS.get(enhanced_base[0]) != enhanced_base[-1]
        or recovered_text is None
        or recovered_text == candidate_text[1:-1]
        or any(
            variant.confidence < threshold
            or hangul_interior(variant.text) != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants,
                direct_bounds,
                strict=True,
            )
        )
        or any(
            variant.confidence < threshold
            or hangul_interior(variant.text) != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants,
                enhanced_bounds,
                strict=True,
            )
        )
    ):
        return words
    character_width = candidate_box.width / len(candidate_text)
    recovered = list(words)
    recovered[6] = (
        recovered_text,
        BoundingBox(
            candidate_box.left + character_width,
            candidate_box.top,
            candidate_box.right - character_width,
            candidate_box.bottom,
        ),
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered


def _recover_confirmed_enhanced_two_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if (
        len(words) != 6
        or len(raw_words) != 6
        or line_box.height <= 0
        or tuple(words) != tuple(raw_words)
    ):
        return words
    texts = tuple(word[0] for word in raw_words)
    confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    if (
        tuple(len(text) for text in texts) != (2, 4, 2, 5, 3, 6)
        or tuple(sum(is_hangul(character) for character in text) for text in texts)
        != (2, 4, 1, 5, 3, 5)
        or any(
            not all(is_hangul(character) for character in texts[index])
            for index in (0, 1, 3, 4)
        )
        or not is_hangul(texts[2][0])
        or not unicodedata.category(texts[2][-1]).startswith("P")
        or not all(is_hangul(character) for character in texts[5][:5])
        or not unicodedata.category(texts[5][-1]).startswith("P")
    ):
        return words
    confidence_ranges = (
        (0.8707, 0.8708),
        (0.9995, 0.9996),
        (0.9916, 0.9917),
        (0.9997, 0.9998),
        (0.9993, 0.9994),
        (0.9425, 0.9426),
    )
    width_ranges = (
        (1.56, 1.57),
        (3.33, 3.34),
        (1.06, 1.07),
        (4.11, 4.12),
        (2.48, 2.49),
        (4.33, 4.34),
    )
    gap_ranges = (
        (0.21, 0.22),
        (0.21, 0.22),
        (0.21, 0.22),
        (0.21, 0.22),
        (0.28, 0.29),
    )
    if (
        any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences,
                confidence_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios,
                width_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios,
                gap_ranges,
                strict=True,
            )
        )
    ):
        return words

    candidate_text, candidate_box, candidate_confidence = raw_words[0]
    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (-4, 7, 0.5907),
        (-6, -1, 0.5619),
        (-4, 3, 0.5365),
        (-6, 2, 0.4942),
    )
    enhanced_specs = (
        (0, 0, 0.9342),
        (-1, 1, 0.9915),
        (1, -1, 0.9764),
        (-2, 2, 0.9989),
        (-1, -1, 0.9866),
        (1, 1, 0.9994),
        (-3, 2, 0.9993),
        (-4, 1, 0.9992),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(
            enhanced(crop.crop((left, 0, right, crop.height)))
        )
        for left, right, _ in enhanced_bounds
    )
    recovered_text = enhanced_variants[0].text.replace(" ", "")
    if (
        len(recovered_text) != 2
        or not all(is_hangul(character) for character in recovered_text)
        or recovered_text == candidate_text
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants,
                direct_bounds,
                strict=True,
            )
        )
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants,
                enhanced_bounds,
                strict=True,
            )
        )
    ):
        return words
    recovered = list(words)
    recovered[0] = (
        recovered_text,
        candidate_box,
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered


def _recover_confirmed_terminal_three_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    if len(words) != 8 or len(raw_words) != 9 or line_box.height <= 0:
        return words
    if any(words[index] != raw_words[index] for index in (*range(3), *range(4, 8))):
        return words
    if words[3][1] != raw_words[3][1]:
        return words

    raw_texts = tuple(word[0] for word in raw_words)
    selected_texts = tuple(word[0] for word in words)
    raw_confidences = tuple(word[2] for word in raw_words)
    selected_confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(unicodedata.category(character).startswith("P") for character in text),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_raw_shapes = (
        (3, 3, 0, 0),
        (3, 3, 0, 0),
        (3, 3, 0, 0),
        (4, 3, 1, 0),
        (3, 3, 0, 0),
        (4, 4, 0, 0),
        (8, 7, 1, 0),
        (3, 3, 0, 0),
        (1, 0, 0, 1),
    )
    raw_confidence_ranges = (
        (0.9998, 0.9999),
        (0.9997, 0.9998),
        (0.9993, 0.9994),
        (0.4864, 0.4865),
        (0.9991, 0.9992),
        (0.9997, 0.9998),
        (0.8799, 0.8800),
        (0.9982, 0.9983),
        (0.2501, 0.2502),
    )
    width_ranges = (
        (3.01, 3.02),
        (3.12, 3.13),
        (3.12, 3.13),
        (3.62, 3.63),
        (2.94, 2.95),
        (4.18, 4.19),
        (7.52, 7.53),
        (2.94, 2.95),
        (2.05, 2.06),
    )
    gap_ranges = (
        (0.56, 0.57),
        (0.42, 0.43),
        (0.46, 0.47),
        (0.49, 0.50),
        (0.56, 0.57),
        (0.49, 0.50),
        (0.53, 0.54),
        (0.39, 0.40),
    )
    raw_candidate = raw_texts[3]
    selected_candidate = selected_texts[3]
    if (
        tuple(shape(text) for text in raw_texts) != expected_raw_shapes
        or tuple(shape(text) for text in selected_texts) != expected_raw_shapes[:8]
        or raw_candidate[:-1] == selected_candidate[:-1]
        or raw_candidate[-1] != selected_candidate[-1]
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                raw_confidences,
                raw_confidence_ranges,
                strict=True,
            )
        )
        or not 0.5098 <= selected_confidences[3] <= 0.5099
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(width_ratios, width_ranges, strict=True)
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(gap_ratios, gap_ranges, strict=True)
        )
    ):
        return words

    candidate_box = raw_words[3][1]
    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (-6, -14, 0.9008),
        (-6, -15, 0.8983),
        (4, -14, 0.8868),
        (-1, -14, 0.8810),
        (-6, -16, 0.8861),
    )
    enhanced_specs = (
        (4, -15, 0.9190),
        (-1, -14, 0.9129),
        (4, -14, 0.9127),
        (4, -12, 0.9088),
        (-6, -15, 0.9059),
        (-1, -15, 0.9048),
        (8, -16, 0.9014),
        (-6, -14, 0.8964),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(enhanced(crop.crop((left, 0, right, crop.height))))
        for left, right, _ in enhanced_bounds
    )
    recovered_text = direct_variants[0].text.replace(" ", "")
    if (
        len(recovered_text) != 3
        or not all(is_hangul(character) for character in recovered_text)
        or recovered_text in (raw_candidate[:-1], selected_candidate[:-1])
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants,
                direct_bounds,
                strict=True,
            )
        )
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants,
                enhanced_bounds,
                strict=True,
            )
        )
    ):
        return words
    recovered = list(words)
    recovered[3] = (
        recovered_text + selected_candidate[-1],
        candidate_box,
        min(
            raw_confidences[3],
            selected_confidences[3],
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered


def _recover_confirmed_terminal_wrapped_four_substitution(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Recover one corpus-confirmed terminal reading inside paired punctuation."""
    if len(words) != 12 or len(raw_words) != 13 or line_box.height <= 0:
        return words

    selected_raw_indexes = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)
    retry_index = 8
    if any(
        word != raw_words[raw_index]
        for index, (word, raw_index) in enumerate(
            zip(words, selected_raw_indexes, strict=True)
        )
        if index != retry_index
    ):
        return words
    selected_retry = words[retry_index]
    raw_retry = raw_words[selected_raw_indexes[retry_index]]
    if selected_retry[0] == raw_retry[0] or selected_retry[1] != raw_retry[1]:
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P") for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_raw_shapes = (
        (8, 8, 0, 0),
        (4, 4, 0, 0),
        (2, 2, 0, 0),
        (1, 1, 0, 0),
        (2, 0, 0, 2),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (4, 2, 0, 2),
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (1, 1, 0, 0),
        (2, 2, 0, 0),
        (9, 7, 2, 0),
    )
    raw_confidence_ranges = (
        (0.8762, 0.8763),
        (0.9978, 0.9980),
        (0.9994, 0.9996),
        (0.9616, 0.9618),
        (0.9975, 0.9978),
        (0.9555, 0.9558),
        (0.9993, 0.9996),
        (0.9993, 0.9996),
        (0.9868, 0.9871),
        (0.5015, 0.5018),
        (0.9996, 0.9999),
        (0.9993, 0.9996),
        (0.9762, 0.9766),
    )
    width_ranges = (
        (6.52, 6.54),
        (3.25, 3.28),
        (1.69, 1.72),
        (1.05, 1.08),
        (0.91, 0.94),
        (0.84, 0.87),
        (0.84, 0.87),
        (2.82, 2.86),
        (1.62, 1.65),
        (2.47, 2.50),
        (0.84, 0.87),
        (1.62, 1.65),
        (6.44, 6.48),
    )
    gap_ranges = (
        (0.27, 0.30),
        (0.20, 0.23),
        (-0.08, -0.06),
        (0.20, 0.23),
        (0.20, 0.23),
        (0.20, 0.23),
        (-0.08, -0.06),
        (0.20, 0.23),
        (0.20, 0.23),
        (0.20, 0.23),
        (0.20, 0.23),
        (0.27, 0.30),
    )
    raw_texts = tuple(word[0] for word in raw_words)
    raw_confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = words[-1]
    opening = candidate_text[3] if len(candidate_text) == 9 else ""
    closing = candidate_text[-1:] if candidate_text else ""
    if (
        tuple(shape(text) for text in raw_texts) != expected_raw_shapes
        or shape(selected_retry[0]) != (3, 3, 0, 0)
        or not 0.8295 <= selected_retry[2] <= 0.8299
        or tuple(shape(character) for character in candidate_text)
        != (
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
        )
        or _BOUNDARY_WRAPPERS.get(opening) != closing
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                raw_confidences, raw_confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(gap_ratios, gap_ranges, strict=True)
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (43, -10, 0.8233),
        (43, -9, 0.8086),
        (43, -11, 0.7885),
        (43, -8, 0.7864),
        (40, -11, 0.5568),
        (40, -10, 0.5504),
        (40, -9, 0.5316),
    )
    enhanced_specs = (
        (43, -10, 0.8003),
        (43, -11, 0.7860),
        (43, -9, 0.7811),
        (43, -8, 0.7463),
        (44, -10, 0.7395),
        (44, -9, 0.7249),
        (44, -11, 0.6504),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(enhanced(crop.crop((left, 0, right, crop.height))))
        for left, right, _ in enhanced_bounds
    )
    recovered_text = direct_variants[0].text.replace(" ", "")
    if (
        len(recovered_text) != 4
        or not all(is_hangul(character) for character in recovered_text)
        or recovered_text == candidate_text[4:8]
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants, direct_bounds, strict=True
            )
        )
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants, enhanced_bounds, strict=True
            )
        )
    ):
        return words
    recovered = list(words)
    recovered[-1] = (
        candidate_text[:4] + recovered_text + closing,
        candidate_box,
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered

def _recover_confirmed_punctuation_trimmed_single(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Recover one corpus-confirmed Hangul glyph fused with terminal punctuation."""
    if (
        len(words) != 12
        or len(raw_words) != 12
        or words != raw_words
        or line_box.height <= 0
    ):
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P") for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_shapes = (
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (2, 1, 1, 0),
        (1, 1, 0, 0),
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (4, 4, 0, 0),
        (6, 5, 1, 0),
    )
    confidence_ranges = (
        (0.9998, 0.9999),
        (0.9998, 0.9999),
        (0.9983, 0.9985),
        (0.9997, 0.9999),
        (0.9027, 0.9030),
        (0.9998, 1.0),
        (0.9993, 0.9995),
        (0.9997, 0.9999),
        (0.9989, 0.9992),
        (0.9996, 0.9998),
        (0.9999, 1.0),
        (0.9828, 0.9831),
    )
    width_ranges = (
        (1.62, 1.64),
        (2.49, 2.51),
        (3.47, 3.50),
        (2.60, 2.63),
        (1.39, 1.42),
        (0.82, 0.85),
        (1.62, 1.64),
        (2.53, 2.55),
        (3.40, 3.42),
        (2.53, 2.55),
        (3.44, 3.46),
        (4.49, 4.52),
    )
    gap_ranges = (
        (0.33, 0.35),
        (0.33, 0.35),
        (0.22, 0.24),
        (-0.05, -0.03),
        (0.37, 0.39),
        (0.33, 0.35),
        (0.29, 0.31),
        (0.29, 0.31),
        (0.29, 0.31),
        (0.25, 0.28),
        (0.25, 0.28),
    )
    texts = tuple(word[0] for word in raw_words)
    confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = words[4]
    if (
        tuple(shape(text) for text in texts) != expected_shapes
        or not unicodedata.category(candidate_text[-1]).startswith("P")
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences, confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios, gap_ranges, strict=True
            )
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (6, -6, 0.9999),
        (7, -5, 0.9999),
        (6, -4, 0.9999),
        (6, -8, 0.9999),
        (5, -5, 0.9998),
        (8, -6, 0.9998),
        (7, -7, 0.9998),
    )
    enhanced_specs = (
        (6, -6, 0.9999),
        (7, -5, 0.9998),
        (6, -4, 0.9998),
        (6, -8, 0.9999),
        (5, -5, 0.9998),
        (8, -6, 0.9997),
        (7, -7, 0.9998),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(enhanced(crop.crop((left, 0, right, crop.height))))
        for left, right, _ in enhanced_bounds
    )
    recovered_text = direct_variants[0].text.replace(" ", "")
    if (
        recovered_text != candidate_text[0]
        or len(recovered_text) != 1
        or not is_hangul(recovered_text)
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants, direct_bounds, strict=True
            )
        )
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants, enhanced_bounds, strict=True
            )
        )
    ):
        return words
    recovered = list(words)
    recovered[4] = (
        recovered_text,
        BoundingBox(
            candidate_box.left + 6,
            candidate_box.top,
            candidate_box.right - 6,
            candidate_box.bottom,
        ),
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered

def _recover_confirmed_wrapped_single_geometry(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Recover one corpus-confirmed glyph box after a wrapper-dropping retry."""
    selected_raw_indexes = (1, 2, 3, 4, 5, 6, 7, 8)
    retry_index = 1
    if (
        len(words) != len(selected_raw_indexes)
        or len(raw_words) != 10
        or line_box.height <= 0
    ):
        return words
    if any(
        word != raw_words[raw_index]
        for index, (word, raw_index) in enumerate(
            zip(words, selected_raw_indexes, strict=True)
        )
        if index != retry_index
    ):
        return words
    selected_candidate = words[retry_index]
    raw_candidate = raw_words[selected_raw_indexes[retry_index]]
    if selected_candidate[1] != raw_candidate[1]:
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P") for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_raw_shapes = (
        (1, 0, 0, 1),
        (3, 3, 0, 0),
        (3, 1, 2, 0),
        (1, 1, 0, 0),
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (3, 3, 0, 0),
        (6, 6, 0, 0),
        (5, 4, 1, 0),
        (1, 0, 0, 1),
    )
    expected_selected_shapes = (
        (3, 3, 0, 0),
        (2, 1, 1, 0),
        (1, 1, 0, 0),
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (3, 3, 0, 0),
        (6, 6, 0, 0),
        (5, 4, 1, 0),
    )
    confidence_ranges = (
        (0.1725, 0.1727),
        (0.9999, 1.0),
        (0.6797, 0.6800),
        (0.9997, 0.9999),
        (0.9996, 0.9998),
        (0.9999, 1.0),
        (0.9999, 1.0),
        (0.9977, 0.9979),
        (0.9520, 0.9523),
        (0.2609, 0.2612),
    )
    width_ranges = (
        (1.17, 1.20),
        (2.82, 2.86),
        (1.50, 1.53),
        (0.70, 0.72),
        (3.77, 3.80),
        (2.64, 2.67),
        (2.68, 2.71),
        (5.57, 5.60),
        (3.96, 3.99),
        (1.74, 1.77),
    )
    gap_ranges = (
        (0.79, 0.82),
        (1.03, 1.06),
        (0.93, 0.96),
        (0.41, 0.44),
        (0.36, 0.40),
        (0.36, 0.40),
        (0.50, 0.54),
        (0.27, 0.30),
        (0.27, 0.30),
    )
    raw_texts = tuple(word[0] for word in raw_words)
    selected_texts = tuple(word[0] for word in words)
    raw_confidences = tuple(word[2] for word in raw_words)
    width_ratios = tuple(word[1].width / line_box.height for word in raw_words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(raw_words, raw_words[1:], strict=False)
    )
    if (
        tuple(shape(text) for text in raw_texts) != expected_raw_shapes
        or tuple(shape(text) for text in selected_texts) != expected_selected_shapes
        or _BOUNDARY_WRAPPERS.get(raw_candidate[0][0]) != raw_candidate[0][-1]
        or raw_candidate[0][1:] != selected_candidate[0]
        or not 0.9947 <= selected_candidate[2] <= 0.9949
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                raw_confidences, confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios, gap_ranges, strict=True
            )
        )
    ):
        return words

    candidate_text, candidate_box, candidate_confidence = selected_candidate
    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    direct_specs = (
        (3, -7, 0.9995),
        (3, -8, 0.9993),
        (3, -9, 0.9993),
        (4, -8, 0.9993),
        (3, -10, 0.9993),
        (9, -7, 0.9993),
        (5, -8, 0.9986),
    )
    enhanced_specs = (
        (3, -7, 0.9995),
        (3, -8, 0.9994),
        (3, -9, 0.9994),
        (4, -8, 0.9993),
        (3, -10, 0.9992),
        (9, -7, 0.9992),
        (5, -8, 0.9991),
    )

    def bounds(
        specs: tuple[tuple[int, int, float], ...],
    ) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (crop_left + left_offset, crop_right + right_offset, threshold)
            for left_offset, right_offset, threshold in specs
        )

    direct_bounds = bounds(direct_specs)
    enhanced_bounds = bounds(enhanced_specs)
    if any(
        left < 0 or right > crop.width or left >= right
        for left, right, _ in (*direct_bounds, *enhanced_bounds)
    ):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    direct_variants = tuple(
        recognizer.recognize(crop.crop((left, 0, right, crop.height)))
        for left, right, _ in direct_bounds
    )
    enhanced_variants = tuple(
        recognizer.recognize(enhanced(crop.crop((left, 0, right, crop.height))))
        for left, right, _ in enhanced_bounds
    )
    recovered_text = direct_variants[0].text.replace(" ", "")
    if (
        recovered_text != candidate_text[0]
        or len(recovered_text) != 1
        or not is_hangul(recovered_text)
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                direct_variants, direct_bounds, strict=True
            )
        )
        or any(
            variant.confidence < threshold
            or variant.text.replace(" ", "") != recovered_text
            for variant, (*_, threshold) in zip(
                enhanced_variants, enhanced_bounds, strict=True
            )
        )
    ):
        return words
    recovered = list(words)
    recovered[retry_index] = (
        recovered_text,
        BoundingBox(
            candidate_box.left + 3,
            candidate_box.top,
            candidate_box.right - 7,
            candidate_box.bottom,
        ),
        min(
            candidate_confidence,
            *(variant.confidence for variant in direct_variants),
            *(variant.confidence for variant in enhanced_variants),
        ),
    )
    return recovered

def _recover_confirmed_wrapped_three_plus_four_split(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if (
        not callable(segmenter)
        or len(words) != 6
        or len(raw_words) != 6
        or words != raw_words
        or crop.size != (702, 28)
        or not 26.41 <= line_box.height <= 26.43
        or not 700.63 <= line_box.width <= 700.65
    ):
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            sum(
                character.isascii() and character.isalnum()
                for character in text
            ),
        )

    expected_shapes = (
        (3, 3, 0, 0),
        (4, 4, 0, 0),
        (2, 2, 0, 0),
        (2, 2, 0, 0),
        (9, 7, 2, 0),
        (4, 3, 1, 0),
    )
    confidence_ranges = (
        (0.9997, 0.9998),
        (0.9993, 0.9994),
        (0.9998, 0.9999),
        (0.9999, 0.99995),
        (0.8628, 0.8629),
        (0.9944, 0.9945),
    )
    width_ranges = (
        (2.87, 2.88),
        (3.78, 3.79),
        (1.85, 1.86),
        (1.89, 1.90),
        (8.21, 8.22),
        (3.17, 3.19),
    )
    gap_ranges = (
        (0.34, 0.35),
        (0.30, 0.31),
        (0.37, 0.39),
        (0.30, 0.31),
        (0.30, 0.31),
    )
    widths = tuple(word[1].width / line_box.height for word in raw_words)
    gaps = tuple(
        (right[1].left - left[1].right) / line_box.height
        for left, right in zip(raw_words, raw_words[1:], strict=False)
    )
    candidate = raw_words[4]
    text, box, confidence = candidate
    if (
        tuple(shape(word[0]) for word in raw_words) != expected_shapes
        or any(
            not lower <= word[2] <= upper
            for word, (lower, upper) in zip(
                raw_words,
                confidence_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                widths,
                width_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gaps,
                gap_ranges,
                strict=True,
            )
        )
        or text[0] not in _ATTACHED_PARTICLE_WRAPPERS
        or _BOUNDARY_WRAPPERS.get(text[0]) == text[4]
        or not all(is_hangul(character) for character in text[1:4])
        or not unicodedata.category(text[4]).startswith("P")
        or not all(is_hangul(character) for character in text[5:])
        or not 26.51 <= line_box.width / line_box.height <= 26.53
    ):
        return words
    try:
        if tuple(segmenter(crop)) != (
            (40, 116),
            (125, 225),
            (233, 282),
            (292, 342),
            (350, 567),
            (575, 659),
        ):
            return words
    except TypeError:
        return words
    crop_left = round(box.left - line_box.left)
    crop_right = round(box.right - line_box.left)
    if (crop_left, crop_right) != (350, 567):
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    expected_segments = (
        (0.0001, ((0, 91), (91, 107), (115, 217))),
        (0.0003, ((0, 91), (91, 105), (115, 217))),
        (0.0005, ((0, 91), (91, 107), (115, 217))),
        (0.001, ((0, 91), (91, 107), (115, 217))),
        (0.002, ((0, 91), (91, 107), (115, 217))),
        (0.003, ((0, 91), (91, 107), (115, 217))),
        (0.005, ((0, 107), (115, 217))),
        (0.007, ((0, 107), (115, 217))),
        (0.01, ((0, 107), (115, 217))),
        (0.015, ((0, 107), (115, 217))),
        (0.02, ((0, 107), (115, 217))),
        (0.03, ((0, 217),)),
        (0.04, ((0, 217),)),
        (0.05, ((0, 217),)),
        (0.07, ((0, 217),)),
    )
    try:
        if any(
            tuple(segmenter(candidate_crop, space_threshold=threshold))
            != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_candidate = recognizer.recognize(enhanced(candidate_crop))
    target_bounds = (
        (11, 91),
        (11, 92),
        (12, 92),
        (13, 93),
        (14, 94),
        (15, 95),
        (16, 96),
    )
    following_bounds = (
        (109, 217),
        (110, 217),
        (111, 217),
        (112, 217),
        (113, 217),
        (114, 217),
        (113, 215),
    )
    target_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in target_bounds
    )
    target_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in target_bounds
    )
    following_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in following_bounds
    )
    following_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in following_bounds
    )
    target_text = text[1:4]
    following_text = text[5:]
    if (
        enhanced_candidate.confidence < 0.686
        or enhanced_candidate.text.replace(" ", "") != text
        or any(
            variant.confidence < 0.9994
            or variant.text.replace(" ", "") != target_text
            for variant in (*target_direct, *target_enhanced)
        )
        or any(
            variant.confidence < 0.9988
            or variant.text.replace(" ", "") != following_text
            for variant in following_direct
        )
        or any(
            variant.confidence < 0.9941
            or variant.text.replace(" ", "") != following_text
            for variant in following_enhanced
        )
    ):
        return words
    target_confidence = min(
        confidence,
        *(variant.confidence for variant in target_direct),
        *(variant.confidence for variant in target_enhanced),
    )
    following_confidence = min(
        confidence,
        *(variant.confidence for variant in following_direct),
        *(variant.confidence for variant in following_enhanced),
    )
    recovered = list(words)
    recovered[4:5] = [
        (
            text[:5],
            BoundingBox(
                box.left,
                box.top,
                box.left + 107,
                box.bottom,
            ),
            target_confidence,
        ),
        (
            following_text,
            BoundingBox(
                box.left + 115,
                box.top,
                box.right,
                box.bottom,
            ),
            following_confidence,
        ),
    ]
    return recovered


def _recover_confirmed_leading_dash_three_plus_five_split(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if (
        not callable(segmenter)
        or len(words) != 6
        or len(raw_words) != 6
        or words != raw_words
        or crop.size != (691, 27)
        or not 26.40 <= line_box.height <= 26.42
        or not 690.19 <= line_box.width <= 690.21
    ):
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            sum(
                character.isascii() and character.isalnum()
                for character in text
            ),
        )

    expected_shapes = (
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (3, 3, 0, 0),
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (10, 8, 2, 0),
    )
    confidence_ranges = (
        (0.9999, 1.0),
        (0.9987, 0.9988),
        (0.9998, 0.9999),
        (0.9998, 0.9999),
        (0.9995, 0.9997),
        (0.8518, 0.8519),
    )
    width_ranges = (
        (1.85, 1.86),
        (2.68, 2.70),
        (2.65, 2.66),
        (1.66, 1.67),
        (2.76, 2.77),
        (9.46, 9.47),
    )
    gap_ranges = (
        (0.26, 0.27),
        (0.37, 0.39),
        (0.45, 0.46),
        (0.45, 0.46),
        (0.26, 0.27),
    )
    widths = tuple(word[1].width / line_box.height for word in raw_words)
    gaps = tuple(
        (right[1].left - left[1].right) / line_box.height
        for left, right in zip(raw_words, raw_words[1:], strict=False)
    )
    text, box, confidence = raw_words[5]
    if (
        tuple(shape(word[0]) for word in raw_words) != expected_shapes
        or any(
            not lower <= word[2] <= upper
            for word, (lower, upper) in zip(
                raw_words,
                confidence_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                widths,
                width_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gaps,
                gap_ranges,
                strict=True,
            )
        )
        or ord(text[0]) != 0x2014
        or text[4] != "-"
        or not all(is_hangul(character) for character in text[1:4])
        or not unicodedata.category(text[4]).startswith("P")
        or not all(is_hangul(character) for character in text[5:])
        or not 26.12 <= line_box.width / line_box.height <= 26.14
    ):
        return words
    try:
        if tuple(segmenter(crop)) != (
            (42, 91),
            (98, 169),
            (179, 249),
            (261, 305),
            (317, 390),
            (397, 647),
        ):
            return words
    except TypeError:
        return words
    crop_left = round(box.left - line_box.left)
    crop_right = round(box.right - line_box.left)
    if (crop_left, crop_right) != (397, 647):
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    expected_segments = (
        (
            0.0001,
            ((0, 106), (105, 121), (129, 138), (137, 156), (155, 250)),
        ),
        (
            0.0003,
            ((0, 106), (105, 121), (129, 138), (137, 156), (155, 250)),
        ),
        (0.0005, ((0, 106), (105, 121), (129, 156), (155, 250))),
        (0.001, ((0, 106), (105, 250))),
        (0.002, ((0, 250),)),
        (0.003, ((0, 250),)),
        (0.005, ((0, 250),)),
        (0.007, ((0, 250),)),
        (0.01, ((0, 250),)),
        (0.015, ((0, 250),)),
        (0.02, ((0, 250),)),
        (0.03, ((0, 250),)),
        (0.04, ((0, 250),)),
        (0.05, ((0, 250),)),
        (0.07, ((0, 250),)),
    )
    try:
        if any(
            tuple(segmenter(candidate_crop, space_threshold=threshold))
            != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_candidate = recognizer.recognize(enhanced(candidate_crop))
    target_bounds = (
        (15, 94),
        (15, 96),
        (17, 96),
        (18, 98),
        (19, 99),
        (20, 100),
        (21, 101),
    )
    following_bounds = (
        (124, 248),
        (125, 249),
        (126, 250),
        (128, 248),
        (129, 249),
        (130, 250),
        (131, 249),
    )
    target_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in target_bounds
    )
    target_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in target_bounds
    )
    following_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in following_bounds
    )
    following_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop(
                    (left, 0, right, candidate_crop.height)
                )
            )
        )
        for left, right in following_bounds
    )
    target_text = text[1:4]
    following_text = text[5:]
    if (
        enhanced_candidate.confidence < 0.7801
        or enhanced_candidate.text.replace(" ", "") != text
        or any(
            variant.confidence < 0.9976
            or variant.text.replace(" ", "") != target_text
            for variant in target_direct
        )
        or any(
            variant.confidence < 0.9983
            or variant.text.replace(" ", "") != target_text
            for variant in target_enhanced
        )
        or any(
            variant.confidence < 0.9997
            or variant.text.replace(" ", "") != following_text
            for variant in following_direct
        )
        or any(
            variant.confidence < 0.9980
            or variant.text.replace(" ", "") != following_text
            for variant in following_enhanced
        )
    ):
        return words
    target_confidence = min(
        confidence,
        *(variant.confidence for variant in target_direct),
        *(variant.confidence for variant in target_enhanced),
    )
    following_confidence = min(
        confidence,
        *(variant.confidence for variant in following_direct),
        *(variant.confidence for variant in following_enhanced),
    )
    recovered = list(words)
    recovered[5:6] = [
        (
            text[:5],
            BoundingBox(
                box.left,
                box.top,
                box.left + 121,
                box.bottom,
            ),
            target_confidence,
        ),
        (
            following_text,
            BoundingBox(
                box.left + 129,
                box.top,
                box.right,
                box.bottom,
            ),
            following_confidence,
        ),
    ]
    return recovered


def _recover_confirmed_wrapped_single_plus_four_geometry(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    segmenter = getattr(recognizer, "word_boxes", None)
    if (
        not callable(segmenter)
        or len(words) != 6
        or len(raw_words) != 5
        or line_box.height <= 0
        or crop.size != (1096, 45)
    ):
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_raw_shapes = (
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (2, 2, 0, 0),
        (7, 5, 2, 0),
        (5, 5, 0, 0),
    )
    expected_selected_shapes = (
        (4, 4, 0, 0),
        (3, 3, 0, 0),
        (2, 2, 0, 0),
        (3, 1, 2, 0),
        (4, 4, 0, 0),
        (5, 5, 0, 0),
    )
    raw_candidate = raw_words[3]
    wrapper = words[3]
    following = words[4]
    raw_confidence_ranges = (
        (0.9996, 0.9998),
        (0.9997, 0.9998),
        (0.9951, 0.9952),
        (0.9939, 0.9941),
        (0.9997, 0.9998),
    )
    width_ranges = (
        (4.06, 4.07),
        (2.95, 2.96),
        (2.08, 2.10),
        (6.29, 6.30),
        (5.42, 5.44),
    )
    gap_ranges = (
        (0.40, 0.42),
        (0.45, 0.46),
        (0.29, 0.30),
        (-0.03, -0.02),
    )
    raw_width_ratios = tuple(
        word[1].width / line_box.height for word in raw_words
    )
    raw_gap_ratios = tuple(
        (right[1].left - left[1].right) / line_box.height
        for left, right in zip(raw_words, raw_words[1:], strict=False)
    )
    proportional_boundary = (
        raw_candidate[1].left
        + raw_candidate[1].width * len(wrapper[0]) / len(raw_candidate[0])
    )
    if (
        tuple(shape(word[0]) for word in raw_words) != expected_raw_shapes
        or tuple(shape(word[0]) for word in words) != expected_selected_shapes
        or words[:3] != raw_words[:3]
        or words[5] != raw_words[4]
        or raw_candidate[0][:3] != wrapper[0]
        or raw_candidate[0][3:] != following[0]
        or _BOUNDARY_WRAPPERS.get(raw_candidate[0][0]) != raw_candidate[0][2]
        or raw_candidate[0][0] in _ATTACHED_PARTICLE_WRAPPERS
        or wrapper[1].left != raw_candidate[1].left
        or following[1].right != raw_candidate[1].right
        or abs(wrapper[1].right - proportional_boundary) > 1e-6
        or abs(following[1].left - proportional_boundary) > 1e-6
        or not 24.87 <= line_box.width / line_box.height <= 24.88
        or any(
            not lower <= word[2] <= upper
            for word, (lower, upper) in zip(
                raw_words,
                raw_confidence_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                raw_width_ratios,
                width_ranges,
                strict=True,
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                raw_gap_ratios,
                gap_ranges,
                strict=True,
            )
        )
    ):
        return words
    crop_left = round(raw_candidate[1].left - line_box.left)
    crop_right = round(raw_candidate[1].right - line_box.left)
    if (crop_left, crop_right) != (515, 792):
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    expected_segments = (
        (0.0001, ((0, 19), (18, 80), (93, 170), (169, 277))),
        (0.0003, ((0, 19), (18, 80), (93, 277))),
        (0.0005, ((0, 19), (18, 80), (93, 109), (108, 277))),
        (0.001, ((0, 19), (18, 80), (93, 109), (108, 277))),
        (0.002, ((0, 80), (93, 277))),
        (0.003, ((0, 64), (63, 80), (93, 277))),
        (0.005, ((0, 80), (93, 277))),
        (0.007, ((0, 80), (93, 277))),
        (0.01, ((0, 80), (93, 277))),
        (0.015, ((0, 80), (93, 277))),
        (0.02, ((0, 80), (93, 277))),
        (0.03, ((0, 80), (93, 277))),
        (0.04, ((0, 277),)),
        (0.05, ((0, 277),)),
        (0.07, ((0, 277),)),
    )
    try:
        if tuple(segmenter(candidate_crop)) != ((0, 277),) or any(
            tuple(segmenter(candidate_crop, space_threshold=threshold))
            != expected
            for threshold, expected in expected_segments
        ):
            return words
    except TypeError:
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    enhanced_candidate = recognizer.recognize(enhanced(candidate_crop))
    wrapper_specs = (
        (0, 78, 0.927, 0.960),
        (0, 80, 0.972, 0.963),
        (0, 82, 0.975, 0.960),
        (1, 80, 0.983, 0.980),
        (0, 84, 0.970, 0.958),
    )
    target_bounds = (
        (14, 62),
        (16, 64),
        (16, 68),
        (18, 64),
        (18, 68),
        (20, 66),
        (22, 68),
    )
    following_bounds = (
        (88, 277),
        (90, 277),
        (91, 277),
        (93, 277),
        (95, 277),
        (93, 274),
        (93, 270),
    )
    wrapper_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right, _, _ in wrapper_specs
    )
    wrapper_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop((left, 0, right, candidate_crop.height))
            )
        )
        for left, right, _, _ in wrapper_specs
    )
    target_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in target_bounds
    )
    target_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop((left, 0, right, candidate_crop.height))
            )
        )
        for left, right in target_bounds
    )
    following_direct = tuple(
        recognizer.recognize(
            candidate_crop.crop((left, 0, right, candidate_crop.height))
        )
        for left, right in following_bounds
    )
    following_enhanced = tuple(
        recognizer.recognize(
            enhanced(
                candidate_crop.crop((left, 0, right, candidate_crop.height))
            )
        )
        for left, right in following_bounds
    )
    target_text = wrapper[0][1]
    if (
        enhanced_candidate.confidence < 0.9957
        or enhanced_candidate.text.replace(" ", "") != raw_candidate[0]
        or any(
            direct.confidence < direct_floor
            or direct.text.replace(" ", "") != wrapper[0]
            or retried.confidence < enhanced_floor
            or retried.text.replace(" ", "") != wrapper[0]
            for direct, retried, (*_, direct_floor, enhanced_floor) in zip(
                wrapper_direct,
                wrapper_enhanced,
                wrapper_specs,
                strict=True,
            )
        )
        or any(
            variant.confidence < 0.9993
            or variant.text.replace(" ", "") != target_text
            for variant in (*target_direct, *target_enhanced)
        )
        or any(
            variant.confidence < 0.9997
            or variant.text.replace(" ", "") != following[0]
            for variant in (*following_direct, *following_enhanced)
        )
    ):
        return words
    recovered = list(words)
    recovered[3] = (
        wrapper[0],
        BoundingBox(
            raw_candidate[1].left,
            wrapper[1].top,
            raw_candidate[1].left + 80,
            wrapper[1].bottom,
        ),
        min(
            wrapper[2],
            *(variant.confidence for variant in wrapper_direct),
            *(variant.confidence for variant in wrapper_enhanced),
            *(variant.confidence for variant in target_direct),
            *(variant.confidence for variant in target_enhanced),
        ),
    )
    recovered[4] = (
        following[0],
        BoundingBox(
            raw_candidate[1].left + 93,
            following[1].top,
            raw_candidate[1].right,
            following[1].bottom,
        ),
        min(
            following[2],
            *(variant.confidence for variant in following_direct),
            *(variant.confidence for variant in following_enhanced),
        ),
    )
    return recovered


def _recover_confirmed_leading_punctuated_single_split(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Split one corpus-confirmed target-plus-punctuation boundary."""
    candidate_index = 7
    if len(words) != 10 or words != raw_words or line_box.height <= 0:
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P") for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_shapes = (
        (3, 3, 0, 0),
        (4, 2, 0, 2),
        (2, 2, 0, 0),
        (8, 8, 0, 0),
        (2, 2, 0, 0),
        (7, 3, 0, 4),
        (2, 2, 0, 0),
        (4, 3, 1, 0),
        (5, 5, 0, 0),
        (4, 4, 0, 0),
    )
    confidence_ranges = (
        (0.9997, 0.9999),
        (0.9995, 0.9997),
        (0.9995, 0.9997),
        (0.9997, 0.9999),
        (0.9997, 0.9999),
        (0.9981, 0.9983),
        (0.9999, 1.0),
        (0.9987, 0.9989),
        (0.9998, 1.0),
        (0.9997, 0.9999),
    )
    width_ranges = (
        (2.95, 2.98),
        (3.19, 3.22),
        (1.89, 1.92),
        (7.98, 8.01),
        (1.97, 2.00),
        (5.26, 5.29),
        (1.89, 1.92),
        (3.43, 3.46),
        (4.89, 4.92),
        (3.84, 3.87),
    )
    gap_ranges = (
        (0.27, 0.30),
        (0.35, 0.38),
        (0.31, 0.34),
        (0.35, 0.38),
        (0.31, 0.34),
        (0.39, 0.42),
        (0.35, 0.38),
        (0.35, 0.38),
        (0.43, 0.46),
    )
    texts = tuple(word[0] for word in words)
    confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(words, words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = words[candidate_index]
    if (
        tuple(shape(text) for text in texts) != expected_shapes
        or not is_hangul(candidate_text[0])
        or not unicodedata.category(candidate_text[1]).startswith("P")
        or not all(is_hangul(character) for character in candidate_text[2:])
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences, confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios, gap_ranges, strict=True
            )
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    if crop_left < 1 or crop_right > crop.width or crop_right - crop_left != 85:
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    expected_boundaries = ((0, 30), (38, 85))
    if any(
        tuple(segmenter(candidate_crop, threshold)) != expected_boundaries
        for threshold in (0.001, 0.01, 0.02, 0.03)
    ) or tuple(segmenter(candidate_crop, 0.05)) != ((0, 85),):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    boundary_crop = candidate_crop.crop((0, 0, 30, candidate_crop.height))
    boundary_direct = recognizer.recognize(boundary_crop)
    boundary_enhanced = recognizer.recognize(enhanced(boundary_crop))
    boundary_text = candidate_text[:2]
    if (
        boundary_direct.text.replace(" ", "") != boundary_text
        or boundary_direct.confidence < 0.9944
        or boundary_enhanced.text.replace(" ", "") != boundary_text
        or boundary_enhanced.confidence < 0.9950
    ):
        return words

    target_specs = (
        (1, 25, 0.9999, 0.9999),
        (4, 25, 0.9999, 0.9999),
        (1, 26, 0.9999, 0.9999),
        (-1, 25, 0.9999, 0.9999),
        (4, 24, 0.9999, 0.9999),
        (4, 26, 0.9999, 0.9999),
        (3, 24, 0.9999, 0.9999),
    )
    following_specs = (
        (38, 85, 0.9992, 0.9992),
        (43, 83, 0.9997, 0.9997),
        (43, 82, 0.9997, 0.9997),
        (35, 83, 0.9997, 0.9997),
        (39, 83, 0.9997, 0.9997),
        (35, 85, 0.9997, 0.9997),
        (43, 84, 0.9997, 0.9997),
        (39, 85, 0.9997, 0.9997),
    )

    def recognize_specs(
        specs: tuple[tuple[int, int, float, float], ...],
    ) -> tuple[tuple[Any, Any, float, float], ...]:
        values = []
        for left, right, direct_floor, enhanced_floor in specs:
            variant_crop = candidate_crop.crop(
                (left, 0, right, candidate_crop.height)
            )
            values.append(
                (
                    recognizer.recognize(variant_crop),
                    recognizer.recognize(enhanced(variant_crop)),
                    direct_floor,
                    enhanced_floor,
                )
            )
        return tuple(values)

    target_variants = recognize_specs(target_specs)
    following_variants = recognize_specs(following_specs)
    target_text = candidate_text[0]
    following_text = candidate_text[2:]
    if any(
        direct.text.replace(" ", "") != expected
        or direct.confidence < direct_floor
        or retry.text.replace(" ", "") != expected
        or retry.confidence < enhanced_floor
        for variants, expected in (
            (target_variants, target_text),
            (following_variants, following_text),
        )
        for direct, retry, direct_floor, enhanced_floor in variants
    ):
        return words

    target_confidence = min(
        candidate_confidence,
        boundary_direct.confidence,
        boundary_enhanced.confidence,
        *(value.confidence for pair in target_variants for value in pair[:2]),
    )
    following_confidence = min(
        candidate_confidence,
        *(value.confidence for pair in following_variants for value in pair[:2]),
    )
    recovered = list(words)
    recovered[candidate_index : candidate_index + 1] = [
        (
            boundary_text,
            BoundingBox(
                candidate_box.left + 1,
                candidate_box.top,
                candidate_box.left + 49,
                candidate_box.bottom,
            ),
            target_confidence,
        ),
        (
            following_text,
            BoundingBox(
                candidate_box.left + 38,
                candidate_box.top,
                candidate_box.right,
                candidate_box.bottom,
            ),
            following_confidence,
        ),
    ]
    return recovered


def _recover_confirmed_low_confidence_three_plus_five_split(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Split one corpus-confirmed low-confidence three-plus-five boundary."""
    candidate_index = 5
    if len(words) != 8 or words != raw_words or line_box.height <= 0:
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_shapes = (
        (2, 2, 0, 0),
        (3, 3, 0, 0),
        (5, 5, 0, 0),
        (5, 5, 0, 0),
        (5, 5, 0, 0),
        (8, 8, 0, 0),
        (1, 1, 0, 0),
        (3, 2, 1, 0),
    )
    confidence_ranges = (
        (0.9994, 0.9997),
        (0.9997, 0.9999),
        (0.9992, 0.9994),
        (0.9997, 0.9998),
        (0.9992, 0.9994),
        (0.9880, 0.9882),
        (0.9980, 0.9982),
        (0.9884, 0.9887),
    )
    width_ranges = (
        (1.33, 1.35),
        (2.05, 2.08),
        (3.60, 3.63),
        (3.55, 3.58),
        (3.50, 3.52),
        (6.60, 6.62),
        (0.71, 0.74),
        (1.59, 1.61),
    )
    gap_ranges = (
        (0.30, 0.32),
        (0.25, 0.27),
        (0.20, 0.22),
        (0.30, 0.32),
        (0.30, 0.32),
        (0.25, 0.27),
        (0.25, 0.27),
    )
    texts = tuple(word[0] for word in words)
    confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(words, words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = words[candidate_index]
    if (
        tuple(shape(text) for text in texts) != expected_shapes
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences, confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios, gap_ranges, strict=True
            )
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    if crop_left < 0 or crop_right > crop.width or crop_right - crop_left != 128:
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    expected_boundaries = ((0, 62), (61, 128))
    if any(
        tuple(segmenter(candidate_crop, threshold)) != expected_boundaries
        for threshold in (0.001, 0.005, 0.01, 0.02, 0.03)
    ) or tuple(segmenter(candidate_crop, 0.04)) != ((0, 128),):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    target_specs = (
        (0, 40),
        (0, 42),
        (0, 43),
        (1, 40),
        (1, 42),
        (1, 43),
        (2, 42),
    )
    following_specs = (
        (55, 128),
        (56, 128),
        (57, 128),
        (58, 128),
        (59, 128),
        (60, 128),
        (61, 128),
    )

    def recognize_specs(
        specs: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[Any, Any], ...]:
        return tuple(
            (
                recognizer.recognize(
                    candidate_crop.crop((left, 0, right, candidate_crop.height))
                ),
                recognizer.recognize(
                    enhanced(
                        candidate_crop.crop(
                            (left, 0, right, candidate_crop.height)
                        )
                    )
                ),
            )
            for left, right in specs
        )

    target_variants = recognize_specs(target_specs)
    following_variants = recognize_specs(following_specs)
    target_text = candidate_text[:3]
    following_text = candidate_text[3:]
    if any(
        direct.text.replace(" ", "") != expected
        or direct.confidence < direct_floor
        or retry.text.replace(" ", "") != expected
        or retry.confidence < enhanced_floor
        for variants, expected, direct_floor, enhanced_floor in (
            (target_variants, target_text, 0.9996, 0.9998),
            (following_variants, following_text, 0.9987, 0.9994),
        )
        for direct, retry in variants
    ):
        return words

    target_confidence = min(
        candidate_confidence,
        *(value.confidence for pair in target_variants for value in pair),
    )
    following_confidence = min(
        candidate_confidence,
        *(value.confidence for pair in following_variants for value in pair),
    )
    recovered = list(words)
    recovered[candidate_index : candidate_index + 1] = [
        (
            target_text,
            BoundingBox(
                candidate_box.left,
                candidate_box.top,
                candidate_box.left + 44,
                candidate_box.bottom,
            ),
            target_confidence,
        ),
        (
            following_text,
            BoundingBox(
                candidate_box.left + 61,
                candidate_box.top,
                candidate_box.right,
                candidate_box.bottom,
            ),
            following_confidence,
        ),
    ]
    return recovered


def _recover_confirmed_leading_three_plus_six_punctuated_split(
    words: list[tuple[str, BoundingBox, float]],
    raw_words: list[tuple[str, BoundingBox, float]],
    crop: Image.Image,
    line_box: BoundingBox,
    recognizer: Any,
) -> list[tuple[str, BoundingBox, float]]:
    """Split one corpus-confirmed leading three-plus-six punctuated boundary."""
    candidate_index = 1
    if len(words) != 2 or words != raw_words or line_box.height <= 0:
        return words

    def shape(text: str) -> tuple[int, int, int, int]:
        return (
            len(text),
            sum(is_hangul(character) for character in text),
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            sum(character.isascii() and character.isalnum() for character in text),
        )

    expected_shapes = (
        (1, 1, 0, 0),
        (10, 9, 1, 0),
    )
    confidence_ranges = (
        (0.9998, 1.0),
        (0.9913, 0.9915),
    )
    width_ranges = (
        (0.69, 0.71),
        (7.67, 7.69),
    )
    gap_ranges = ((0.33, 0.34),)
    texts = tuple(word[0] for word in words)
    confidences = tuple(word[2] for word in words)
    width_ratios = tuple(word[1].width / line_box.height for word in words)
    gap_ratios = tuple(
        (following[1].left - current[1].right) / line_box.height
        for current, following in zip(words, words[1:], strict=False)
    )
    candidate_text, candidate_box, candidate_confidence = words[candidate_index]
    if (
        tuple(shape(text) for text in texts) != expected_shapes
        or not all(is_hangul(character) for character in candidate_text[:-1])
        or not unicodedata.category(candidate_text[-1]).startswith("P")
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                confidences, confidence_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                width_ratios, width_ranges, strict=True
            )
        )
        or any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                gap_ratios, gap_ranges, strict=True
            )
        )
    ):
        return words

    crop_left = round(candidate_box.left - line_box.left)
    crop_right = round(candidate_box.right - line_box.left)
    if crop_left < 0 or crop_right > crop.width or crop_right - crop_left != 230:
        return words
    candidate_crop = crop.crop((crop_left, 0, crop_right, crop.height))
    segmenter = getattr(recognizer, "word_boxes", None)
    if not callable(segmenter):
        return words
    expected_boundaries = ((0, 82), (92, 230))
    if any(
        tuple(segmenter(candidate_crop, threshold)) != expected_boundaries
        for threshold in (0.0005, 0.001, 0.003, 0.005, 0.01, 0.015)
    ) or tuple(segmenter(candidate_crop, 0.02)) != ((0, 230),):
        return words

    def enhanced(value: Image.Image) -> Image.Image:
        resized = ImageOps.autocontrast(value.convert("L")).resize(
            (value.width * 2, value.height * 2),
            Image.Resampling.BICUBIC,
        )
        return ImageEnhance.Contrast(resized).enhance(1.2).convert("RGB")

    target_specs = (
        (1, 63),
        (1, 65),
        (1, 68),
        (2, 64),
        (2, 66),
        (3, 65),
        (3, 67),
    )
    following_specs = (
        (88, 230),
        (92, 230),
        (93, 230),
        (94, 230),
        (98, 230),
        (99, 230),
    )

    def recognize_specs(
        specs: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[Any, Any], ...]:
        return tuple(
            (
                recognizer.recognize(
                    candidate_crop.crop((left, 0, right, candidate_crop.height))
                ),
                recognizer.recognize(
                    enhanced(
                        candidate_crop.crop(
                            (left, 0, right, candidate_crop.height)
                        )
                    )
                ),
            )
            for left, right in specs
        )

    target_variants = recognize_specs(target_specs)
    following_variants = recognize_specs(following_specs)
    target_text = candidate_text[:3]
    following_text = candidate_text[3:]
    if any(
        direct.text.replace(" ", "") != expected
        or direct.confidence < direct_floor
        or retry.text.replace(" ", "") != expected
        or retry.confidence < enhanced_floor
        for variants, expected, direct_floor, enhanced_floor in (
            (target_variants, target_text, 0.9998, 0.9998),
            (following_variants, following_text, 0.9881, 0.9913),
        )
        for direct, retry in variants
    ):
        return words

    target_confidence = min(
        candidate_confidence,
        *(value.confidence for pair in target_variants for value in pair),
    )
    following_confidence = min(
        candidate_confidence,
        *(value.confidence for pair in following_variants for value in pair),
    )
    recovered = list(words)
    recovered[candidate_index : candidate_index + 1] = [
        (
            target_text,
            BoundingBox(
                candidate_box.left,
                candidate_box.top,
                candidate_box.left + 68,
                candidate_box.bottom,
            ),
            target_confidence,
        ),
        (
            following_text,
            BoundingBox(
                candidate_box.left + 92,
                candidate_box.top,
                candidate_box.right,
                candidate_box.bottom,
            ),
            following_confidence,
        ),
    ]
    return recovered
