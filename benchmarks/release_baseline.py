"""Deterministic, aggregate-only release baseline using synthetic in-memory images."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sqlite3
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from kiwipiepy import Kiwi
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer

FONT_FILES = (
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path(r"C:\Windows\Fonts\gulim.ttc"),
    Path(r"C:\Windows\Fonts\batang.ttc"),
)
FONT_SIZES = (24, 30, 36, 42, 48)
COMPLEX_FONT_SIZES = (28, 34, 40, 46, 52)
VERBS = [
    "가다",
    "오다",
    "먹다",
    "마시다",
    "보다",
    "읽다",
    "쓰다",
    "듣다",
    "말하다",
    "공부하다",
    "일하다",
    "자다",
    "일어나다",
    "앉다",
    "서다",
    "걷다",
    "뛰다",
    "만나다",
    "좋아하다",
    "사랑하다",
    "살다",
    "사다",
    "팔다",
    "만들다",
    "주다",
    "받다",
    "열다",
    "닫다",
    "찾다",
    "기다리다",
    "배우다",
    "가르치다",
    "시작하다",
    "끝나다",
    "끝내다",
    "알다",
    "모르다",
    "생각하다",
    "이해하다",
    "기억하다",
    "잊다",
    "입다",
    "벗다",
    "씻다",
    "요리하다",
    "운동하다",
    "전화하다",
    "여행하다",
    "선택하다",
    "준비하다",
]
NOUNS = [
    "학교",
    "집",
    "회사",
    "도서관",
    "식당",
    "카페",
    "공원",
    "병원",
    "은행",
    "시장",
    "친구",
    "가족",
    "선생님",
    "학생",
    "사람",
    "아이",
    "책",
    "음식",
    "물",
    "커피",
    "시간",
    "오늘",
    "내일",
    "어제",
    "아침",
    "점심",
    "저녁",
    "한국",
    "서울",
    "한국어",
    "영화",
    "음악",
    "사진",
    "컴퓨터",
    "전화",
    "가방",
    "옷",
    "신발",
    "버스",
    "지하철",
    "방",
    "문",
    "창문",
    "길",
    "날씨",
    "이름",
    "질문",
    "대답",
    "문제",
    "여행",
]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def _words(database: Path, count: int, stride: int, offset: int) -> list[str]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        available = [
            row[0]
            for row in connection.execute("SELECT DISTINCT headword FROM entries ORDER BY id")
            if re.fullmatch("[가-힣]{2,8}", row[0])
        ]
    if len(available) < count:
        raise RuntimeError("dictionary does not contain enough benchmark headwords")
    return [available[(index * stride + offset) % len(available)] for index in range(count)]


def _fonts() -> dict[tuple[Path, int], ImageFont.FreeTypeFont]:
    missing = [str(path) for path in FONT_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing Windows benchmark fonts: {', '.join(missing)}")
    return {
        (path, size): ImageFont.truetype(str(path), size)
        for path in FONT_FILES
        for size in (*FONT_SIZES, *COMPLEX_FONT_SIZES)
    }


def _measure_ocr(
    engine: PaddleOcrEngine,
    words: list[str],
    render: Callable[[int, str], Image.Image],
) -> dict[str, float | int]:
    exact = 0
    missing = 0
    durations = []
    for index, expected in enumerate(words):
        image = render(index, expected)
        started = time.perf_counter()
        document = engine.recognize(image)
        durations.append((time.perf_counter() - started) * 1000)
        recognized = " ".join(line.text for line in document.lines)
        exact += recognized == expected
        missing += not document.lines
    return {
        "samples": len(words),
        "whole_eojeol_exact_pct": round(exact / len(words) * 100, 2),
        "missing_pct": round(missing / len(words) * 100, 2),
        "latency_median_ms": round(statistics.median(durations), 2),
        "latency_p95_ms": round(_p95(durations), 2),
    }


def clean(engine: PaddleOcrEngine, database: Path) -> dict[str, float | int]:
    fonts = _fonts()
    words = _words(database, 500, 97, 0)

    def render(index: int, text: str) -> Image.Image:
        image = Image.new("RGB", (700, 120), (255 - (index % 3) * 8,) * 3)
        ImageDraw.Draw(image).text(
            (20 + (index % 7) * 3, 18 + (index % 5) * 2),
            text,
            font=fonts[(FONT_FILES[index % 4], FONT_SIZES[index % 5])],
            fill=(index % 4 * 10,) * 3,
        )
        return image

    return _measure_ocr(engine, words, render)


def subtitles(engine: PaddleOcrEngine, database: Path) -> dict[str, float | int]:
    fonts = _fonts()
    words = _words(database, 300, 131, 17)

    def render(index: int, text: str) -> Image.Image:
        y = np.arange(180, dtype=np.uint8)[:, None]
        x = np.arange(900, dtype=np.uint16)[None, :]
        pixels = np.empty((180, 900, 3), np.uint8)
        pixels[:, :, 0] = (20 + (x % 80) + (index % 4) * 15).astype(np.uint8)
        pixels[:, :, 1] = (15 + (y % 70) + (index % 3) * 20).astype(np.uint8)
        pixels[:, :, 2] = (30 + ((x // 5 + y) % 90)).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        font = fonts[(FONT_FILES[index % 3], FONT_SIZES[1 + index % 4])]
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        left = (900 - (bounds[2] - bounds[0])) // 2
        draw.rectangle((0, 45, 899, 135), fill="black")
        draw.text(
            (left, 60 + (index % 3) * 5),
            text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        return image

    return _measure_ocr(engine, words, render)


def complex_backgrounds(engine: PaddleOcrEngine, database: Path) -> dict[str, float | int]:
    fonts = _fonts()
    words = _words(database, 200, 173, 29)
    randomizer = random.Random(4401)

    def render(index: int, text: str) -> Image.Image:
        low_resolution = np.random.default_rng(index + 4401).integers(
            20, 236, (12, 45, 3), dtype=np.uint8
        )
        image = (
            Image.fromarray(low_resolution)
            .resize((900, 240), Image.Resampling.BICUBIC)
            .filter(ImageFilter.GaussianBlur(2))
        )
        draw = ImageDraw.Draw(image)
        for _ in range(8):
            left = randomizer.randrange(0, 820)
            top = randomizer.randrange(0, 190)
            color = tuple(randomizer.randrange(20, 236) for _ in range(3))
            draw.rounded_rectangle(
                (
                    left,
                    top,
                    left + randomizer.randrange(30, 160),
                    top + randomizer.randrange(20, 80),
                ),
                radius=10,
                fill=color,
            )
        font = fonts[(FONT_FILES[index % 4], COMPLEX_FONT_SIZES[index % 5])]
        layer = Image.new("RGBA", (700, 120), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        bounds = layer_draw.textbbox((0, 0), text, font=font, stroke_width=2)
        left = (700 - (bounds[2] - bounds[0])) // 2
        layer_draw.text(
            (left, 25),
            text,
            font=font,
            fill="white" if index % 2 else "black",
            stroke_width=2 + index % 2,
            stroke_fill="black" if index % 2 else "white",
        )
        layer = layer.rotate((-5, -2, 0, 2, 5)[index % 5], Image.Resampling.BICUBIC)
        image.paste(layer, (100, 60), layer)
        return image

    return _measure_ocr(engine, words, render)


def morphology(database: Path) -> dict[str, float | int]:
    kiwi = Kiwi()
    analyzer = KoreanAnalyzer(SqliteDictionaryStore(database), kiwi)
    cases: list[tuple[str, tuple[int, int], str, set[str]]] = []
    for lemma in VERBS:
        stem = lemma[:-1]
        forms = (
            (
                kiwi.join([(stem, "VV"), ("었", "EP"), ("습니다", "EF")]),
                {"past tense", "formal polite style"},
            ),
            (kiwi.join([(stem, "VV"), ("어요", "EF")]), {"polite style"}),
            (
                kiwi.join([(stem, "VV"), ("으시", "EP"), ("었", "EP"), ("어요", "EF")]),
                {"honorific", "past tense", "polite style"},
            ),
            (
                kiwi.join([(stem, "VV"), ("겠", "EP"), ("습니다", "EF")]),
                {"future or intention", "formal polite style"},
            ),
        )
        for surface, labels in forms:
            cases.append((f"저는 {surface}", (3, 3 + len(surface)), lemma, labels))
    for lemma in NOUNS:
        for particle, tag, tail in (("에서", "JKB", " 공부해요"), ("을", "JKO", " 봐요")):
            surface = kiwi.join([(lemma, "NNG"), (particle, tag)])
            cases.append((f"저는 {surface}{tail}", (3, 3 + len(surface)), lemma, {"particle"}))
    correct = 0
    definitions = 0
    combined = 0
    durations = []
    for sentence, span, lemma, labels in cases:
        started = time.perf_counter()
        candidates = analyzer.analyze(sentence, span)
        durations.append((time.perf_counter() - started) * 1000)
        first = candidates[0] if candidates else None
        is_correct = bool(
            first and first.lemma == lemma and labels <= {item.label for item in first.features}
        )
        has_definition = bool(first and first.dictionary_entries)
        correct += is_correct
        definitions += has_definition
        combined += is_correct and has_definition
    return {
        "samples": len(cases),
        "correct_lemma_and_breakdown_first_pct": round(correct / len(cases) * 100, 2),
        "first_candidate_definition_pct": round(definitions / len(cases) * 100, 2),
        "combined_first_result_pct": round(combined / len(cases) * 100, 2),
        "latency_median_ms": round(statistics.median(durations), 2),
        "latency_p95_ms": round(_p95(durations), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aggregate-only release baseline")
    parser.add_argument("assets", type=Path)
    parser.add_argument(
        "--category",
        choices=("clean", "subtitles", "complex", "morphology", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    database = arguments.assets / "dictionary.sqlite3"
    selected = (
        ("clean", "subtitles", "complex", "morphology")
        if arguments.category == "all"
        else (arguments.category,)
    )
    results: dict[str, object] = {
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        }
    }
    if any(item != "morphology" for item in selected):
        engine = PaddleOcrEngine(
            PaddleDetector(arguments.assets / "korean_detection.onnx"),
            PaddleRecognizer(
                arguments.assets / "korean_recognition.onnx",
                arguments.assets / "korean_characters.txt",
            ),
        )
    for category in selected:
        if category == "clean":
            results[category] = clean(engine, database)
        elif category == "subtitles":
            results[category] = subtitles(engine, database)
        elif category == "complex":
            results[category] = complex_backgrounds(engine, database)
        else:
            results[category] = morphology(database)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
