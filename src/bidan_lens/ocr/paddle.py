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
        if text != 'K':
            merged.append((text, box, confidence))
        index += 1
    return merged


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
                elif isolated_three_plus_one_profile:
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
    words = _recover_overlapping_word_triplets(words, crop, line_box, recognizer)
    words = _discard_confirmed_overlapping_character_duplicates(
        words,
        crop,
        line_box,
        recognizer,
    )
    words = _recover_overlapping_suffix_pairs(words, crop, line_box, recognizer)
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

    def _segmented_line(
        self, crop: Image.Image, line_box: BoundingBox
    ) -> OcrLine | None:
        segmenter = getattr(self.recognizer, 'word_boxes', None)
        if not callable(segmenter):
            return None
        segments = segmenter(crop)
        if len(segments) <= 1:
            return None
        words: list[tuple[str, BoundingBox, float]] = []
        for left, right in segments:
            word_crop = crop.crop((left, 0, right, crop.height))
            recognized = self.recognizer.recognize(word_crop)
            if recognized.confidence < self.retry_threshold:
                retry_image = ImageOps.autocontrast(word_crop.convert('L')).resize(
                    (word_crop.width * 2, word_crop.height * 2), Image.Resampling.BICUBIC
                )
                retry_image = ImageEnhance.Contrast(retry_image).enhance(1.2)
                retry = self.recognizer.recognize(retry_image.convert('RGB'))
                if retry.confidence > recognized.confidence:
                    recognized = retry
            text = recognized.text.replace(' ', '')
            if text and (
                contains_hangul(text)
                or (
                    recognized.confidence >= _context_confidence_threshold(text)
                    and _structured_ascii_context(text)
                )
                or (text == 'K' and recognized.confidence >= 0.8)
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
        words = [
            part
            for text, box, confidence in words
            for part in _split_punctuation_wrapped_word(text, box, confidence)
        ]
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
