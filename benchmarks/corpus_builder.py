"""Build deterministic, provenance-carrying release corpora without model-generated truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmarks.locked_corpus import (
    EXPECTED_COUNTS,
    LOCK_NAME,
    SCHEMA_VERSION,
    SOURCE_MANIFEST_NAME,
    CorpusError,
    _lock_files,
    _sha256,
    load_morphology_cases,
    load_ocr_samples,
    load_sources,
)

_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3
_MORPH_ORACLE = "published-annotation-independent-map"
_RENDER_ORACLE = "known-render"
_AIHUB_ORACLE = "published-annotation"


@dataclass(frozen=True, slots=True)
class TextRecord:
    text: str
    source_sample_id: str
    source_split: str


@dataclass(frozen=True, slots=True)
class UdToken:
    token_id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    misc: str


@dataclass(frozen=True, slots=True)
class UdSentence:
    sent_id: str
    text: str
    tokens: tuple[UdToken, ...]


@dataclass(frozen=True, slots=True)
class MorphCandidate:
    sample_id: str
    sentence: str
    span: tuple[int, int]
    lemma: str
    labels: frozenset[str]


@dataclass(frozen=True, slots=True)
class AihubSample:
    sample_id: str
    image: Path
    lines: tuple[dict[str, object], ...]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise CorpusError(f"metadata must be an object: {path.name}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stable_order(values: Iterable[Any], seed: int, key: Any) -> list[Any]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{key(value)}".encode()).digest(),
    )


def _contains_hangul(text: str) -> bool:
    return any(_HANGUL_START <= ord(character) <= _HANGUL_END for character in text)


def _trim_eojeol(text: str) -> tuple[str, int]:
    start = 0
    end = len(text)
    while start < end and unicodedata.category(text[start])[0] in {"P", "Z"}:
        start += 1
    while end > start and unicodedata.category(text[end - 1])[0] in {"P", "Z"}:
        end -= 1
    return text[start:end], start


def _manifest_source_ids(root: Path) -> set[str]:
    manifest = _read_object(root / SOURCE_MANIFEST_NAME)
    values = manifest.get("sources")
    if not isinstance(values, list):
        raise CorpusError("sources.json must list sources")
    ids = {
        value.get("id")
        for value in values
        if isinstance(value, dict) and isinstance(value.get("id"), str)
    }
    if not ids:
        raise CorpusError("sources.json must list at least one source")
    return ids


def _require_sources(root: Path, *source_ids: str | None) -> None:
    known = _manifest_source_ids(root)
    unknown = {source_id for source_id in source_ids if source_id and source_id not in known}
    if unknown:
        raise CorpusError("builder references a source absent from sources.json")


def _provenance(
    source_id: str,
    sample_id: str,
    source_split: str,
    oracle: str,
    supporting_source_ids: Iterable[str] = (),
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_sample_id": sample_id,
        "source_split": source_split,
        "oracle": oracle,
        "supporting_source_ids": sorted(set(supporting_source_ids)),
    }


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise CorpusError(f"output category is not empty: {path.name}")
    path.mkdir(parents=True, exist_ok=True)


def _load_text_records(path: Path, default_split: str) -> list[TextRecord]:
    records: list[TextRecord] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CorpusError("cannot read render source JSONL") from error
    for raw in raw_lines:
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CorpusError("render source contains invalid JSON") from error
        if not isinstance(value, dict):
            raise CorpusError("render source records must be objects")
        text = value.get("text")
        sample_id = value.get("source_sample_id")
        source_split = value.get("source_split", default_split)
        if not all(isinstance(item, str) and item for item in (text, sample_id, source_split)):
            raise CorpusError("render source record has invalid fields")
        if not _contains_hangul(text):
            continue
        records.append(TextRecord(unicodedata.normalize("NFC", text), sample_id, source_split))
    return records


def _generated_background(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    first = tuple(rng.randint(28, 90) for _ in range(3))
    second = tuple(rng.randint(80, 170) for _ in range(3))
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        amount = y / max(1, height - 1)
        color = tuple(round(a + (b - a) * amount) for a, b in zip(first, second, strict=True))
        draw.line((0, y, width, y), fill=color)
    for _ in range(16):
        x = rng.randrange(width)
        y = rng.randrange(height)
        radius = rng.randrange(25, 130)
        color = tuple(min(255, channel + rng.randrange(5, 40)) for channel in second)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def _load_background(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as source:
            return ImageOps.fit(source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    except (OSError, ValueError) as error:
        raise CorpusError(f"cannot read background image: {path.name}") from error


def _layout_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    left: float,
    top: float,
    max_width: float,
    line_gap: int,
    fill: str,
    stroke_width: int = 0,
    stroke_fill: str | None = None,
) -> list[dict[str, object]]:
    raw_words = text.split()
    if not raw_words:
        raise CorpusError("cannot render an empty sentence")
    space_width = draw.textlength(" ", font=font)
    line_height = max(1, font.getbbox("한글")[3] - font.getbbox("한글")[1])
    rows: list[list[str]] = [[]]
    row_width = 0.0
    for word in raw_words:
        word_width = draw.textlength(word, font=font)
        proposed = word_width if not rows[-1] else row_width + space_width + word_width
        if rows[-1] and proposed > max_width:
            rows.append([word])
            row_width = word_width
        else:
            rows[-1].append(word)
            row_width = proposed

    lines: list[dict[str, object]] = []
    y = top
    for words in rows:
        x = left
        eojeols: list[dict[str, object]] = []
        word_boxes: list[tuple[float, float, float, float]] = []
        for raw_word in words:
            draw.text(
                (x, y),
                raw_word,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            raw_box = draw.textbbox(
                (x, y),
                raw_word,
                font=font,
                stroke_width=stroke_width,
            )
            word_boxes.append(tuple(float(item) for item in raw_box))
            eojeol, prefix = _trim_eojeol(raw_word)
            if eojeol and _contains_hangul(eojeol):
                eojeol_x = x + draw.textlength(raw_word[:prefix], font=font)
                eojeol_box = draw.textbbox(
                    (eojeol_x, y),
                    eojeol,
                    font=font,
                    stroke_width=stroke_width,
                )
                eojeols.append({"text": eojeol, "box": [float(item) for item in eojeol_box]})
            x += draw.textlength(raw_word, font=font) + space_width
        if not eojeols:
            continue
        line_box = [
            min(box[0] for box in word_boxes),
            min(box[1] for box in word_boxes),
            max(box[2] for box in word_boxes),
            max(box[3] for box in word_boxes),
        ]
        lines.append({"text": " ".join(words), "box": line_box, "eojeols": eojeols})
        y += line_height + line_gap
    return lines


def render_ocr(
    source: Path,
    corpus: Path,
    category: str,
    source_id: str,
    font_source_id: str,
    fonts: list[Path],
    count: int,
    seed: int,
    source_split: str,
    backgrounds: Path | None = None,
    background_source_id: str | None = None,
) -> dict[str, object]:
    if category not in {"clean", "subtitles"}:
        raise CorpusError("known rendering supports only clean and subtitles")
    _require_sources(corpus, source_id, font_source_id, background_source_id)
    if not fonts or any(not font.is_file() for font in fonts):
        raise CorpusError("at least one readable font file is required")
    records = _stable_order(
        _load_text_records(source, source_split), seed, lambda value: value.source_sample_id
    )
    if len(records) < count:
        raise CorpusError("render source has fewer eligible records than requested")
    background_files: list[Path] = []
    if backgrounds is not None:
        background_files = sorted(
            path
            for path in backgrounds.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not background_files or background_source_id is None:
            raise CorpusError("licensed backgrounds require files and a background source id")

    output = corpus / category
    _prepare_output(output)
    width, height = (1280, 720)
    for index, record in enumerate(records[:count], start=1):
        rng = random.Random(seed + index * 7919)
        if category == "subtitles":
            image = (
                _load_background(
                    background_files[(index - 1) % len(background_files)], (width, height)
                )
                if background_files
                else _generated_background(width, height, seed + index)
            )
            font_size = rng.randint(34, 52)
            fill = "white"
            stroke_width = max(2, font_size // 16)
            stroke_fill = "black"
            top = rng.randint(500, 590)
            max_width = 1080
        else:
            theme = rng.choice(
                (("#ffffff", "#202124"), ("#f7f8fa", "#1f2937"), ("#202124", "#f1f3f4"))
            )
            image = Image.new("RGB", (width, height), theme[0])
            draw_ui = ImageDraw.Draw(image)
            draw_ui.rounded_rectangle(
                (80, 70, 1200, 650), radius=18, fill=theme[0], outline="#9aa0a6", width=2
            )
            font_size = rng.randint(20, 34)
            fill = theme[1]
            stroke_width = 0
            stroke_fill = None
            top = rng.randint(145, 260)
            max_width = 1000
        font_path = fonts[(index - 1) % len(fonts)]
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except OSError as error:
            raise CorpusError(f"cannot load font: {font_path.name}") from error
        draw = ImageDraw.Draw(image)
        measured_width = sum(draw.textlength(word, font=font) for word in record.text.split())
        measured_width += max(0, len(record.text.split()) - 1) * draw.textlength(" ", font=font)
        left = (
            max(100.0, (width - min(max_width, measured_width)) / 2)
            if category == "subtitles"
            else 130.0
        )
        lines = _layout_text(
            draw,
            record.text,
            font,
            left,
            top,
            max_width,
            max(8, font_size // 2),
            fill,
            stroke_width,
            stroke_fill,
        )
        name = f"{index:04d}"
        image.save(output / f"{name}.png", optimize=True)
        supporting = [font_source_id]
        if background_source_id:
            supporting.append(background_source_id)
        _write_json(
            output / f"{name}.json",
            {
                "image": f"{name}.png",
                "lines": lines,
                "provenance": _provenance(
                    source_id,
                    record.source_sample_id,
                    record.source_split,
                    _RENDER_ORACLE,
                    supporting,
                ),
                "render": {
                    "renderer": "Pillow",
                    "font_file": font_path.name,
                    "font_size": font_size,
                    "seed": seed + index * 7919,
                },
            },
        )
    return {"category": category, "samples": count, "oracle": _RENDER_ORACLE}


def _parse_ud(path: Path) -> list[UdSentence]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CorpusError(f"cannot read CoNLL-U source: {path.name}") from error
    sentences: list[UdSentence] = []
    for block in raw.split("\n\n"):
        sent_id = text = ""
        tokens: list[UdToken] = []
        for line in block.splitlines():
            if line.startswith("# sent_id = "):
                sent_id = line.removeprefix("# sent_id = ").strip()
            elif line.startswith("# text = "):
                text = line.removeprefix("# text = ").strip()
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
                    continue
                tokens.append(
                    UdToken(fields[0], fields[1], fields[2], fields[3], fields[4], fields[9])
                )
        if sent_id and text and tokens:
            sentences.append(UdSentence(sent_id, text, tuple(tokens)))
    return sentences


def _misc_value(misc: str, key: str) -> str | None:
    for item in misc.split("|"):
        name, separator, value = item.partition("=")
        if separator and name == key:
            return value
    return None


def _morph_parts(token: UdToken) -> tuple[list[str], list[str]]:
    lemma = _misc_value(token.misc, "OrigLemma") or token.lemma
    forms = lemma.split("+") if lemma and lemma != "_" else []
    tags = token.xpos.lower().split("+") if token.xpos and token.xpos != "_" else []
    return forms, tags


def _expected_lemma(token: UdToken) -> str | None:
    forms, tags = _morph_parts(token)
    if not forms:
        return None
    verb_roots = {"pvg", "paa", "px", "vv", "va", "vx", "xsv", "xsa"}
    for index, tag in enumerate(tags):
        if (
            tag.startswith("n")
            and index + 1 < min(len(forms), len(tags))
            and tags[index + 1] in {"xsv", "xsa"}
        ):
            return forms[index] + forms[index + 1] + "다"
        if tag in verb_roots:
            return forms[index] if forms[index].endswith("다") else forms[index] + "다"
    for index, tag in enumerate(tags):
        if tag.startswith("n") and index < len(forms):
            return forms[index]
    if token.upos in {"VERB", "ADJ", "AUX"}:
        return forms[0] if forms[0].endswith("다") else forms[0] + "다"
    return forms[0]


def _expected_labels(token: UdToken) -> frozenset[str]:
    forms, tags = _morph_parts(token)
    labels: set[str] = set()
    for form, tag in zip(forms, tags, strict=False):
        if tag.startswith("j"):
            labels.add("particle")
        if tag == "ep":
            if any(marker in form for marker in ("았", "었", "였")):
                labels.add("past tense")
            if "겠" in form:
                labels.add("future or intention")
            if "시" in form:
                labels.add("honorific")
        if tag in {"ec", "ecc", "ecs", "ecx"}:
            if form == "고":
                labels.add("connecting ending")
            if form.endswith("지만"):
                labels.add("contrast ending")
            if form.endswith("면"):
                labels.add("conditional ending")
    if token.form.endswith("세요"):
        labels.add("polite request")
    elif token.form.endswith(("어요", "아요")):
        labels.add("polite style")
    if token.form.endswith("습니다") or token.form.endswith("ㅂ니다"):
        labels.add("formal polite style")
    return frozenset(labels)


def import_ud(
    inputs: list[Path],
    corpus: Path,
    source_id: str,
    source_split: str,
    count: int,
    seed: int,
) -> dict[str, object]:
    _require_sources(corpus, source_id)
    candidates: list[MorphCandidate] = []
    for path in inputs:
        for sentence in _parse_ud(path):
            cursor = 0
            for token in sentence.tokens:
                start = sentence.text.find(token.form, cursor)
                if start < 0:
                    continue
                end = start + len(token.form)
                cursor = end
                labels = _expected_labels(token)
                lemma = _expected_lemma(token)
                if labels and lemma and _contains_hangul(token.form):
                    candidates.append(
                        MorphCandidate(
                            f"{sentence.sent_id}:{token.token_id}",
                            sentence.text,
                            (start, end),
                            lemma,
                            labels,
                        )
                    )
    ordered = _stable_order(candidates, seed, lambda value: value.sample_id)
    if len(ordered) < count:
        raise CorpusError("CoNLL-U inputs have fewer eligible annotated cases than requested")
    output = corpus / "morphology.jsonl"
    if output.exists():
        raise CorpusError("morphology.jsonl already exists")
    selected = ordered[:count]
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for case in selected:
            value = {
                "sentence": case.sentence,
                "target_span": list(case.span),
                "expected_lemma": case.lemma,
                "expected_labels": sorted(case.labels),
                "provenance": _provenance(
                    source_id,
                    case.sample_id,
                    source_split,
                    _MORPH_ORACLE,
                ),
            }
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    label_counts = Counter(label for case in selected for label in case.labels)
    return {
        "category": "morphology",
        "samples": count,
        "oracle": _MORPH_ORACLE,
        "label_counts": dict(sorted(label_counts.items())),
    }


def _aihub_datasets(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for metadata in paths:
        value = _read_object(metadata)
        if isinstance(value.get("images"), list) and isinstance(value.get("annotations"), list):
            yield value


def _resolve_image(images_root: Path, file_name: str, by_name: dict[str, Path]) -> Path | None:
    direct = images_root / Path(file_name)
    if direct.is_file():
        return direct
    return by_name.get(Path(file_name).name)


def import_aihub(
    labels: Path,
    images: Path,
    corpus: Path,
    category: str,
    source_id: str,
    source_split: str,
    count: int,
    seed: int,
) -> dict[str, object]:
    if category not in {"clean", "subtitles", "complex"}:
        raise CorpusError("invalid OCR category")
    _require_sources(corpus, source_id)
    by_name = {
        path.name: path
        for path in images.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    samples: list[AihubSample] = []
    rejected = Counter()
    for dataset in _aihub_datasets(labels):
        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in dataset["annotations"]:
            if isinstance(annotation, dict):
                annotations_by_image[str(annotation.get("image_id", ""))].append(annotation)
        for image_value in dataset["images"]:
            if not isinstance(image_value, dict):
                continue
            image_id = str(image_value.get("id", ""))
            file_name = image_value.get("file_name")
            if not image_id or not isinstance(file_name, str):
                rejected["invalid_image_metadata"] += 1
                continue
            image_path = _resolve_image(images, file_name, by_name)
            if image_path is None:
                rejected["missing_image"] += 1
                continue
            expected_lines: list[dict[str, object]] = []
            for annotation in annotations_by_image.get(image_id, []):
                text = annotation.get("text")
                bbox = annotation.get("bbox")
                if not isinstance(text, str) or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                normalized = unicodedata.normalize("NFC", text.strip())
                if (
                    normalized.lower() == "xxx"
                    or " " in normalized
                    or not _contains_hangul(normalized)
                ):
                    continue
                try:
                    x, y, width, height = (float(item) for item in bbox)
                except (TypeError, ValueError):
                    continue
                if width <= 0 or height <= 0:
                    continue
                box = [x, y, x + width, y + height]
                expected_lines.append(
                    {"text": normalized, "box": box, "eojeols": [{"text": normalized, "box": box}]}
                )
            if expected_lines:
                samples.append(AihubSample(image_id, image_path, tuple(expected_lines)))
            else:
                rejected["no_eligible_eojeol"] += 1
    ordered = _stable_order(samples, seed, lambda value: value.sample_id)
    if len(ordered) < count:
        raise CorpusError("AI Hub annotations have fewer eligible samples than requested")
    output = corpus / category
    _prepare_output(output)
    for index, sample in enumerate(ordered[:count], start=1):
        suffix = sample.image.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"
        name = f"{index:04d}"
        shutil.copy2(sample.image, output / f"{name}{suffix}")
        _write_json(
            output / f"{name}.json",
            {
                "image": f"{name}{suffix}",
                "lines": sample.lines,
                "provenance": _provenance(
                    source_id,
                    sample.sample_id,
                    source_split,
                    _AIHUB_ORACLE,
                ),
            },
        )
    return {
        "category": category,
        "samples": count,
        "oracle": _AIHUB_ORACLE,
        "rejected": dict(sorted(rejected.items())),
    }


def lock_corpus(root: Path, corpus_id: str, allow_incomplete: bool) -> dict[str, object]:
    manifest = _read_object(root / SOURCE_MANIFEST_NAME)
    source_values = manifest.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise CorpusError("sources.json must list sources")
    evidence: list[str] = []
    for source in source_values:
        if not isinstance(source, dict) or not isinstance(source.get("license_evidence"), str):
            raise CorpusError("source has no license evidence")
        evidence.append(source["license_evidence"])
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {LOCK_NAME, f"{LOCK_NAME}.tmp"}
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "source_manifest": SOURCE_MANIFEST_NAME,
        "license_evidence": sorted(set(evidence)),
        "files": files,
    }
    temporary = root / f"{LOCK_NAME}.tmp"
    _write_json(temporary, value)
    temporary.replace(root / LOCK_NAME)
    _, locked = _lock_files(root)
    sources = load_sources(root, locked)
    counts: dict[str, int] = {}
    for category in ("clean", "subtitles", "complex"):
        counts[category] = len(load_ocr_samples(root, category, locked, sources))
    counts["morphology"] = len(load_morphology_cases(root, locked, sources))
    complete = all(counts[name] == expected for name, expected in EXPECTED_COUNTS.items())
    if not complete and not allow_incomplete:
        raise CorpusError("corpus does not have the exact release sample counts")
    return {"corpus_id": corpus_id, "counts": counts, "release_sample_counts": complete}


def validate_corpus(root: Path, allow_incomplete: bool) -> dict[str, object]:
    corpus_id, locked = _lock_files(root)
    sources = load_sources(root, locked)
    counts = {
        category: len(load_ocr_samples(root, category, locked, sources))
        for category in ("clean", "subtitles", "complex")
    }
    counts["morphology"] = len(load_morphology_cases(root, locked, sources))
    complete = all(counts[name] == expected for name, expected in EXPECTED_COUNTS.items())
    if not complete and not allow_incomplete:
        raise CorpusError("corpus does not have the exact release sample counts")
    return {
        "corpus_id": corpus_id,
        "sources": len(sources),
        "counts": counts,
        "release_sample_counts": complete,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an automated BiDan Lens corpus")
    commands = parser.add_subparsers(dest="command", required=True)

    acquire_plain_parser = commands.add_parser(
        "acquire-plain", help="download and verify pinned plain-v1 sources"
    )
    acquire_plain_parser.add_argument("destination", type=Path)
    acquire_plain_parser.add_argument(
        "--source-lock", type=Path, default=Path(__file__).with_name("plain_sources.lock.json")
    )
    acquire_plain_parser.add_argument(
        "--malgun", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf")
    )
    acquire_plain_parser.add_argument("--local-krdict", type=Path)

    build_plain_parser = commands.add_parser(
        "build-plain", help="render the deterministic v3 plain-v1 corpus"
    )
    build_plain_parser.add_argument("acquired", type=Path)
    build_plain_parser.add_argument("corpus", type=Path)
    build_plain_parser.add_argument("--profile", choices=("dev", "release"), required=True)
    build_plain_parser.add_argument("--seed", type=int, default=20260822)
    build_plain_parser.add_argument("--count", type=int, default=2000)
    build_plain_parser.add_argument("--stress-count", type=int, default=250)

    lock_plain_parser = commands.add_parser(
        "lock-plain", help="hash-lock and validate a v3 plain-v1 corpus"
    )
    lock_plain_parser.add_argument("corpus", type=Path)
    lock_plain_parser.add_argument("--corpus-id", required=True)
    lock_plain_parser.add_argument("--allow-incomplete", action="store_true")

    validate_plain_parser = commands.add_parser(
        "validate-plain", help="verify a v3 plain-v1 corpus lock"
    )
    validate_plain_parser.add_argument("corpus", type=Path)
    validate_plain_parser.add_argument("--allow-incomplete", action="store_true")

    render = commands.add_parser("render-ocr", help="render known Korean text with exact boxes")
    render.add_argument("source", type=Path)
    render.add_argument("corpus", type=Path)
    render.add_argument("category", choices=("clean", "subtitles"))
    render.add_argument("--source-id", required=True)
    render.add_argument("--font-source-id", required=True)
    render.add_argument("--font", action="append", type=Path, required=True)
    render.add_argument("--count", type=int, required=True)
    render.add_argument("--seed", type=int, default=20260822)
    render.add_argument("--source-split", default="test")
    render.add_argument("--backgrounds", type=Path)
    render.add_argument("--background-source-id")

    ud = commands.add_parser("import-ud", help="derive morphology truth from CoNLL-U")
    ud.add_argument("corpus", type=Path)
    ud.add_argument("input", nargs="+", type=Path)
    ud.add_argument("--source-id", required=True)
    ud.add_argument("--source-split", default="test")
    ud.add_argument("--count", type=int, default=EXPECTED_COUNTS["morphology"])
    ud.add_argument("--seed", type=int, default=20260822)

    aihub = commands.add_parser("import-aihub", help="import AI Hub COCO-style OCR labels")
    aihub.add_argument("labels", type=Path)
    aihub.add_argument("images", type=Path)
    aihub.add_argument("corpus", type=Path)
    aihub.add_argument("category", choices=("clean", "subtitles", "complex"))
    aihub.add_argument("--source-id", required=True)
    aihub.add_argument("--source-split", default="test")
    aihub.add_argument("--count", type=int, required=True)
    aihub.add_argument("--seed", type=int, default=20260822)

    lock = commands.add_parser("lock", help="hash-lock and validate the corpus")
    lock.add_argument("corpus", type=Path)
    lock.add_argument("--corpus-id", required=True)
    lock.add_argument("--allow-incomplete", action="store_true")

    validate = commands.add_parser("validate", help="verify an existing corpus lock")
    validate.add_argument("corpus", type=Path)
    validate.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.command == "acquire-plain":
        from benchmarks.plain_corpus import acquire_plain

        result = acquire_plain(
            arguments.destination,
            source_lock=arguments.source_lock,
            malgun=arguments.malgun,
            local_krdict=arguments.local_krdict,
        )
    elif arguments.command == "build-plain":
        from benchmarks.plain_corpus import build_plain

        result = build_plain(
            arguments.acquired,
            arguments.corpus,
            arguments.profile,
            seed=arguments.seed,
            count=arguments.count,
            stress_count=arguments.stress_count,
        )
    elif arguments.command == "lock-plain":
        from benchmarks.plain_evaluator import lock_plain_corpus

        result = lock_plain_corpus(
            arguments.corpus,
            arguments.corpus_id,
            allow_incomplete=arguments.allow_incomplete,
        )
    elif arguments.command == "validate-plain":
        from benchmarks.plain_evaluator import validate_plain_corpus

        result = validate_plain_corpus(
            arguments.corpus, allow_incomplete=arguments.allow_incomplete
        )
    elif arguments.command == "render-ocr":
        result = render_ocr(
            arguments.source,
            arguments.corpus,
            arguments.category,
            arguments.source_id,
            arguments.font_source_id,
            arguments.font,
            arguments.count,
            arguments.seed,
            arguments.source_split,
            arguments.backgrounds,
            arguments.background_source_id,
        )
    elif arguments.command == "import-ud":
        result = import_ud(
            arguments.input,
            arguments.corpus,
            arguments.source_id,
            arguments.source_split,
            arguments.count,
            arguments.seed,
        )
    elif arguments.command == "import-aihub":
        result = import_aihub(
            arguments.labels,
            arguments.images,
            arguments.corpus,
            arguments.category,
            arguments.source_id,
            arguments.source_split,
            arguments.count,
            arguments.seed,
        )
    elif arguments.command == "lock":
        result = lock_corpus(arguments.corpus, arguments.corpus_id, arguments.allow_incomplete)
    else:
        result = validate_corpus(arguments.corpus, arguments.allow_incomplete)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
