# Development status

## Implemented in the repository

- local source-layout Python application and Windows tray shell;
- automatic scanning and hold-key activation modes;
- Paddle-compatible ONNX CPU detector/recognizer with CTC-guided eojeol crops, conservative
  visual-gap refinement, adaptive recognition width, conservative edge-punctuation recovery,
  structured ASCII sentence context, exact reconstructed word geometry, and one
  low-confidence retry;
- immutable OCR lines, glyphs, eojeols and whole-eojeol pointer hit testing;
- sentence-aware Kiwi adapter with lemma recovery, ordered lexical components, contextual
  auxiliary roles, role-first KRDict grouping, verified spacing notes, learner-facing grammar
  labels, and a dictionary-backed known-particle fallback;
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
- an independent v4 oracle for ordered components, contextual roles, verified spacing, and
  exact role-grouped KRDict entry/sense order, with 400 held-out language cases;
- an aggregate-only `plain-v1` evaluator covering OCR, pointer selection, functional context,
  exact-transcription diagnostics, complete components, dictionary conformance, first-popup
  correctness, negative-pointer classes, corrections, latency, and render strata;
- strict v4 locking and validation for every corpus file, source, license, renderer/font
  record, duplicate identity, forbidden training split, and required balanced stratum;
- the earlier version-two subtitle/complex evaluator remains compatible as an optional,
  non-release measurement path;
- an opt-in runtime latency recorder covering capture start through popup event flush,
  with warm-up exclusion, release sample-count enforcement, and aggregate-only output.

## Local plain-v1 development evidence

The latest schema-v4.2 development and release corpora are built, locked, and validated under
`F:\bidan-lens-eval-ud218-v4.2`; the previous v3, v4, and v4.1 roots remain preserved. The
v4.2 release corpus has not been evaluated. The earlier complete v4 development run records
96.03% whole-eojeol OCR, 73.75%
functional context, 57.95% complete ordered components, 42.30% fully correct first popups,
0% false promotions,
3.08% negative activation, and 223.77 ms median / 336.05 ms p95 automated latency. It passes
latency and false-promotion gates but misses popup, negative-activation, dictionary-conformance,
and required-stratum gates.

The latest v4.2 development language tier records 88.00% fully correct first popups, 93.50%
auxiliary correctness, 82.50% multi-lexical correctness, and 100% independent direct KRDict
conformance across 243 unique groups. Conservative score-bounded component promotion and an
undefined-leader isolated-eojeol fallback improve the multi-lexical tier by 3.5 percentage
points over the corrected v4.2 baseline; KRDict-verified noun-prefix particle recovery adds
another half point without a language or quick-tier regression. Deeper internal Kiwi search
reduces primary-lemma failures with no navigable matching candidate from 23 to 17.
Its 200-sample rendered quick tier records 96.43% whole-eojeol OCR, 74.50% functional context,
60.50% component accuracy, 46.50% fully correct popups, and 2.11% negative activation, with
231.20 ms median / 371.56 ms p95 automated latency. A complete v4.2 development render
evaluation has not yet been run. Multi-lexical language, rendered popup, functional-context,
negative-activation, and required-stratum gates still block release and foreground evidence. See
`docs/RELEASE_BASELINE_2026-08.md` for the measurement breakdown.

## Required before a public v1 release

- publish the validated asset bundle at an immutable project release location and record
  its checksum outside the bundle;
- improve against the locked development corpus until the mandatory floors pass, then
  record the complete release report against the untouched official test split;
- meet the aggregate and every size/punctuation exceptional floor, the false-promotion
  gate, and the primary OCR/fully-correct-popup targets (or explicitly approve a documented
  exceptional release);
- improve the held-out multi-lexical tier from 82.50% to at least the 88% exceptional floor;
- run the opt-in foreground benchmark with five warmups plus 500 fixed attempts, meeting
  correctness and latency targets with zero safety violations;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
