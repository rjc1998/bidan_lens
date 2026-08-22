# Development status

## Implemented in the repository

- local source-layout Python application and Windows tray shell;
- automatic scanning and hold-key activation modes;
- Paddle-compatible ONNX CPU detector/recognizer and one low-confidence retry;
- immutable OCR lines, glyphs, eojeols and whole-eojeol pointer hit testing;
- sentence-aware Kiwi adapter with lemma recovery and learner-facing grammar labels;
- KRDict JSON adapter, versioned SQLite builder and read-only lookup;
- ranked alternative results with wheel/hotkey navigation and copy actions;
- verified atomic bundle download/import with an offline path;
- popup capture exclusion and privacy-preserving pipeline boundaries;
- deterministic unit/integration tests and Windows CI/build definitions;
- hash-locked official PP-OCRv5 detector/Korean recognizer archives and reproducible ONNX
  outputs, with exact preprocessing and output-contract validation;
- audited August 2026 KRDict archive import with 54,134 English-bearing entries and 74,255
  senses, including collision-safe handling of related idioms and proverbs;
- complete PaddleOCR/KRDict asset licenses, KRDict attribution, and provenance inside a
  locally built and atomically installed release-candidate bundle;
- fail-closed collection of 21 runtime/CPython license payloads into the tested Windows
  distribution, including reviewed fallbacks for wheels that omit their license file;
- deterministic synthetic baselines at 96.60% clean, 97.00% subtitle, 80.50% complex,
  and 95.33% morphology accuracy;
- an aggregate-only locked-corpus evaluator that verifies every sample and its license
  evidence by SHA-256, enforces release sample counts, and measures marked correction
  false promotions without printing recognized or expected text;
- an opt-in runtime latency recorder covering capture start through popup event flush,
  with warm-up exclusion, release sample-count enforcement, and aggregate-only output.

## Required before a public v1 release

- publish the validated asset bundle at an immutable project release location and record
  its checksum outside the bundle;
- curate and run the evaluator against a locked, licensed real website/subtitle/game corpus
  and record comparable results;
- curate the locked morphology corpus and record its marked-correction false-promotion rate;
- record at least 500 successful clean-text popup timings and confirm the 500 ms median and
  1 second p95 latency targets on the release machine class;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
