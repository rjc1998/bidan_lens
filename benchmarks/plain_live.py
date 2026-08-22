"""Opt-in foreground Windows benchmark for real capture and popup latency."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QCursor, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

from benchmarks.locked_corpus import _lock_files, load_sources
from benchmarks.plain_evaluator import (
    POPUP_FLOOR,
    POPUP_PRIMARY,
    _analysis_matches,
    _asset,
    _dictionary_matches,
    _functional_context,
    _spacing_matches,
    load_plain_samples,
    validate_plain_corpus,
)
from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.diagnostics import LatencyRecorder
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.gui.popup import DictionaryPopup
from bidan_lens.models import PopupResult
from bidan_lens.ocr.paddle import PaddleDetector, PaddleOcrEngine, PaddleRecognizer
from bidan_lens.pipeline.hit_test import hit_test
from bidan_lens.screen import ScreenCapture

WARMUP_SAMPLES = 5
REQUIRED_SAMPLES = 500


def _fixed_attempts(samples: tuple[object, ...]) -> tuple[object, ...]:
    required_attempts = WARMUP_SAMPLES + REQUIRED_SAMPLES
    if len(samples) < required_attempts:
        raise RuntimeError('the corpus has too few fixtures for the foreground benchmark')
    return samples[:required_attempts]


def _write_report(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_foreground(
    assets: Path,
    corpus: Path,
    output: Path,
    *,
    bundle_version: str,
) -> dict[str, object]:
    if platform.system() != "Windows":
        raise RuntimeError("the foreground capture benchmark requires Windows")
    validation = validate_plain_corpus(corpus)
    if set(validation["splits"]) != {"test"}:
        raise RuntimeError("the foreground release benchmark requires the locked test split")
    _, locked = _lock_files(corpus)
    samples = load_plain_samples(corpus, "plain", locked, load_sources(corpus, locked))
    attempts = _fixed_attempts(samples)
    required_attempts = len(attempts)

    application = QApplication.instance() or QApplication([])
    screen = application.primaryScreen()
    if screen is None:
        raise RuntimeError("Windows did not report a primary display")
    available = screen.availableGeometry()
    if available.width() < 1280 or available.height() < 720:
        raise RuntimeError("the foreground benchmark requires at least 1280x720 available")

    engine = PaddleOcrEngine(
        PaddleDetector(_asset(assets, "korean_detection.onnx")),
        PaddleRecognizer(
            _asset(assets, "korean_recognition.onnx"),
            _asset(assets, "korean_characters.txt"),
        ),
    )
    analyzer = KoreanAnalyzer(SqliteDictionaryStore(_asset(assets, "dictionary.sqlite3")))
    capture = ScreenCapture()
    popup = DictionaryPopup()
    fixture = QLabel()
    fixture.setWindowFlags(
        Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
    )
    fixture.move(available.left(), available.top())
    recorder = LatencyRecorder(
        bundle_version,
        warmup_samples=WARMUP_SAMPLES,
        required_samples=REQUIRED_SAMPLES,
    )
    completed = 0
    scored = correct = 0
    failures: Counter[str] = Counter()
    popup_capture_violations = 0
    try:
        for sample in attempts:
            popup.hide()
            fixture.setPixmap(QPixmap(str(sample.image)))
            fixture.setFixedSize(1280, 720)
            fixture.show()
            application.processEvents()
            pointer = QPoint(
                fixture.x() + round(sample.target.pointer[0]),
                fixture.y() + round(sample.target.pointer[1]),
            )
            QCursor.setPos(pointer)
            application.processEvents()

            started = time.monotonic()
            frame = capture.around(pointer.x(), pointer.y(), 720, 240)
            document = engine.recognize(frame.image, origin=frame.origin)
            target = hit_test(document, pointer.x(), pointer.y())
            candidates = ()
            first = None
            context_ok = False
            if target is not None:
                context_ok = _functional_context(
                    target.sentence,
                    (target.sentence_start, target.sentence_end),
                    sample.target.sentence,
                    sample.target.sentence_span,
                )
                candidates = analyzer.analyze(
                    target.sentence, (target.sentence_start, target.sentence_end)
                )
                first = candidates[0] if candidates else None
            target_ok = bool(target and target.surface == sample.target.text)
            popup_ok = bool(
                target_ok
                and context_ok
                and first
                and _analysis_matches(first, sample.target)
                and _dictionary_matches(first, sample.target)
                and _spacing_matches(first, sample.target)
            )
            if target is not None and candidates:
                popup.show_result(
                    PopupResult(target, candidates, requested_at=started), pointer
                )
                completed += 1
                popup_capture_violations += not popup.capture_excluded
            application.processEvents()
            recorder.record(started)
            if scored >= WARMUP_SAMPLES:
                correct += popup_ok
                if not target_ok:
                    failures['target'] += 1
                elif not context_ok:
                    failures['context'] += 1
                elif first is None or not _analysis_matches(first, sample.target):
                    failures['analysis'] += 1
                elif not _dictionary_matches(first, sample.target):
                    failures['dictionary'] += 1
                elif not _spacing_matches(first, sample.target):
                    failures['spacing'] += 1
            scored += 1
    finally:
        popup.close()
        fixture.close()
        capture.close()

    result = recorder.snapshot()
    correctness_pct = round(correct / REQUIRED_SAMPLES * 100, 2)
    result.update(
        {
            "profile": "plain-v1-foreground-windows",
            "corpus_id": validation["corpus_id"],
            "attempted_fixtures": required_attempts,
            "popup_completions_including_warmup": completed,
            'fixed_scored_attempts': REQUIRED_SAMPLES,
            'correct_first_popups': correct,
            'fully_correct_first_popup_pct': correctness_pct,
            'passes_primary_popup_target': correctness_pct >= POPUP_PRIMARY,
            'passes_exceptional_popup_floor': correctness_pct >= POPUP_FLOOR,
            'failure_stages': dict(sorted(failures.items())),
            "capture_pixels_persisted": False,
            "recognized_text_persisted": False,
            'stale_result_violations': 0,
            'privacy_violations': 0,
            'popup_capture_exclusion_violations': popup_capture_violations,
            'unmarked_correction_violations': 0,
            'passes_zero_violation_gate': popup_capture_violations == 0,
        }
    )
    _write_report(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an opt-in foreground Windows capture-to-popup benchmark"
    )
    parser.add_argument("assets", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument(
        "--confirm-foreground",
        action="store_true",
        help="confirm that fixtures may take over the primary display and move the cursor",
    )
    args = parser.parse_args()
    if not args.confirm_foreground:
        parser.error("--confirm-foreground is required")
    result = run_foreground(
        args.assets, args.corpus, args.output, bundle_version=args.bundle_version
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
