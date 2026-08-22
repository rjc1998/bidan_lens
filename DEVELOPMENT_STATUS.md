# Development status

## Implemented in the repository

- local source-layout Python application and Windows tray shell;
- automatic scanning and hold-key activation modes;
- Paddle-compatible ONNX CPU detector/recognizer with CTC-guided eojeol crops, conservative
  visual-gap refinement, adaptive recognition width, conservative edge-punctuation recovery,
  structured ASCII sentence context, exact reconstructed word geometry, and one
  low-confidence retry;
- immutable OCR lines, glyphs, eojeols and whole-eojeol pointer hit testing;
- sentence-aware Kiwi adapter with lemma recovery, learner-facing grammar labels, and a
  dictionary-backed known-particle fallback when the first analysis has no definition;
- KRDict JSON adapter, versioned SQLite builder and read-only exact-headword/alias lookup;
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
- a pinned, automated `plain-v1` acquisition workflow for UD Korean GSD/KAIST 2.18,
  four open Korean font families, local Malgun Gothic evidence, and KRDict;
- deterministic 2,000-sample development/release corpus rendering split equally across
  Chromium and offscreen Qt, with exact line/eojeol/target/pointer geometry, a locked
  200-sample quick subset, and a separate 250-sample 10 px stress tier;
- an independent v3 oracle for KAIST-derived learner labels and exact KRDict entry/sense
  order, with source/test-split separation and no production Kiwi participation;
- an aggregate-only `plain-v1` evaluator covering OCR, pointer selection, sentence spans,
  morphology, dictionary fidelity, first-popup correctness, alternative recovery, marked
  corrections, latency, Wilson intervals, and all required render strata;
- strict v3 locking and validation for every corpus file, source, license, renderer/font
  record, duplicate identity, forbidden training split, and required balanced stratum;
- the earlier version-two subtitle/complex evaluator remains compatible as an optional,
  non-release measurement path;
- an opt-in runtime latency recorder covering capture start through popup event flush,
  with warm-up exclusion, release sample-count enforcement, and aggregate-only output.

## Local plain-v1 development evidence

Pinned sources were acquired and the complete development and release corpora were built,
locked, and validated outside Git. The release corpus has not been evaluated or inspected.
The complete 2,000-sample development run records 96.19% whole-eojeol OCR, 49.75% fully
correct first popups, 0% false promotions, and 213.65 ms median / 331.67 ms p95 automated
pipeline latency. It passes the aggregate exceptional OCR floor but misses the primary OCR
target, the popup floor, and required size/punctuation strata, so release evaluation remains
blocked. See
`docs/RELEASE_BASELINE_2026-08.md` for the measurement breakdown.

## Required before a public v1 release

- publish the validated asset bundle at an immutable project release location and record
  its checksum outside the bundle;
- improve against the locked development corpus until the mandatory floors pass, then
  record the complete release report against the untouched official test split;
- meet the aggregate and every size/punctuation exceptional floor, the false-promotion
  gate, and the primary OCR/fully-correct-popup targets (or explicitly approve a documented
  exceptional release);
- run the opt-in foreground benchmark and record 500 successful capture-to-popup timings,
  confirming the 500 ms median and 1 second p95 targets on the release machine class;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
