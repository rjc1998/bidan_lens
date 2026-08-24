# Plain-v1 evaluation workflow

The mandatory version-one gate covers plain websites and ordinary Windows desktop text.
Release corpora remain outside Git. The repository commits only acquisition/build code,
the v4 schema validator, tests, and `plain_sources.lock.json`. Normal application use still
keeps screenshots in memory; corpus rendering is an explicit developer workflow using
public text and fonts.

The workflow produces two independent corpus directories:

- `dev`: 2,000 samples from official UD `dev` splits, 250 nonblocking 10 px stress samples,
  400 held-out language cases, and a locked 200-sample quick subset;
- `release`: 2,000 samples from official UD `test` splits, plus 250 nonblocking 10 px
  stress samples and 400 held-out language cases.

Never optimize against, prune, or replace failed release samples. Generate and lock the
release corpus before the production evaluation. Keep the development and release roots
separate so their source identities cannot be mixed.

## Install evaluation tools

Use Python 3.12 on Windows. Playwright is evaluation-only and its bundled Chromium revision
is recorded and hash-locked in each generated corpus.

```powershell
python -m pip install -e ".[dev,evaluation]"
playwright install chromium
```

## Acquire pinned sources

`acquire-plain` downloads and verifies every artifact in `plain_sources.lock.json` before
activation. It includes immutable UD Korean GSD/KAIST 2.18 files and licenses, Noto Sans KR,
Noto Serif KR, Nanum Gothic, Nanum Myeongjo, and the KRDict English export. Malgun Gothic is
read from Windows, hashed, and recorded without redistribution. A previously downloaded
KRDict archive may be supplied to avoid downloading it again.

```powershell
$env:PYTHONPATH = "src;."
python -m benchmarks.corpus_builder acquire-plain D:\bidan-eval\acquired `
  --local-krdict D:\downloads\krdict-2026-08-json.zip
```

Any network failure, size mismatch, or SHA-256 mismatch fails acquisition. Downloaded
datasets, fonts, licenses, and KRDict files remain in the external destination.

## Build and lock corpora

Build development and release corpora into empty, separate directories. Browser and Qt
renderers supply exact geometry; neither production OCR nor production Kiwi is consulted.

```powershell
python -m benchmarks.corpus_builder build-plain D:\bidan-eval\acquired `
  D:\bidan-eval\dev --profile dev
python -m benchmarks.corpus_builder lock-plain D:\bidan-eval\dev `
  --corpus-id bidan-plain-v4-dev-ud218

python -m benchmarks.corpus_builder build-plain D:\bidan-eval\acquired `
  D:\bidan-eval\release --profile release
python -m benchmarks.corpus_builder lock-plain D:\bidan-eval\release `
  --corpus-id bidan-plain-v4-release-ud218
```

The lock covers every image, annotation, source/license record, font/browser/renderer record,
and quick manifest. Validation enforces exact sample counts and balanced release strata:

```powershell
python -m benchmarks.corpus_builder validate-plain D:\bidan-eval\release
```

`--count`, `--stress-count`, and `--allow-incomplete` are available only for developer
fixtures and tests. An incomplete corpus is never release eligible.

## Evaluate

Use the 200-sample development subset while iterating:

```powershell
python -m benchmarks.locked_corpus assets\runtime\installed\2026.08.1 `
  D:\bidan-eval\dev --profile plain-v1 --quick `
  --diagnostics D:\bidan-eval\reports\dev-failures.json
```

Diagnostics include stable sample IDs, failed stages, and render strata only. They never
include recognized text, expected text, definitions, or pixels.

Review development language disagreements locally with
`python -m benchmarks.language_review ASSETS CORPUS DECISIONS`. The interactive command
displays one locked public-corpus case at a time, but its JSON output persists only the corpus
ID, stable sample IDs, failure categories, and categorical decisions. Use `--inspect` to view
unresolved cases without writing and `--audit` to require a complete, current review.

The available decisions are Kiwi analysis error, annotation-convention difference,
equivalent learner interpretation, corpus-oracle defect, and genuinely ambiguous Korean.
Review only development data; do not inspect or classify release disagreements before the
development thresholds are frozen.

Review quick-tier functional-context disagreements locally with
`python -m benchmarks.context_review ASSETS CORPUS DECISIONS`. It follows the same privacy
boundary: expected and recognized public-corpus text is displayed only during local review,
while the decision file contains only the corpus ID, stable sample IDs, and categorical
decisions. Use `--inspect` for unresolved cases and `--audit` to require complete coverage.
The available decisions are missed or merged OCR word boundary, incorrect line/sentence
reconstruction, punctuation or structured-ASCII handling, incorrect target span, and genuinely
ambiguous layout.
With `--inspect`, `--decision CATEGORY` limits local display to current cases in one reviewed
category, and `--sample-id ID` selects one current stable ID. `--geometry-only` emits only IDs,
render metadata, lengths, spans, Unicode-category counts, per-eojeol boxes and confidence, and
text-equality/geometry signals between adjacent eojeols. These inspection views remain
local-only and never change the decision report.
For a single stable ID, `--segmentation-only` additionally reports detector regions, raw word
segments, recognition lengths/confidence/category counts, and equality flags for overlapping
triplets, overlapping pairs, and unusually close pairs. Close-pair diagnostics also expose only
prefix/suffix equality and an added-terminal-punctuation flag, never the characters themselves.
The view never emits recognized or oracle text.
Target-selection failures and negative activations can be inspected without entering the
context-review set by repeating `--target-geometry ID`. This read-only view emits only stable
IDs, render categories, target/probe pointers, expected and recognized geometry, text lengths,
confidence, Hangul counts, and boolean target-match signals. It never writes decisions or
emits recognized or oracle text. Add `--target-segmentation` to include the same privacy-safe
detector and raw segment evidence used by the context inspector.

Review first-popup analysis and dictionary disagreements only after target and functional
context are correct with
`python -m benchmarks.popup_review ASSETS CORPUS DECISIONS`. Its report persists only stable
sample IDs, categorical failure stages, and the same five categorical language decisions.
`--failure-stage STAGE` and `--decision CATEGORY` can revisit current reviewed cases during
local inspection. `--structure-only` emits only stable IDs, categorical decisions, lengths,
roles, counts, scores, and equality flags; it omits corpus and analysis text. `--audit` detects
missing, resolved, or stage-changed decisions without storing corpus text.
After structure-only review, record one decision without redisplaying corpus text with
`--sample-id ID --record-decision CATEGORY`; the reviewer derives and persists the current
categorical failure stage itself.

Run the complete development corpus until its provisional gates pass and the thresholds are
explicitly frozen. Build and lock the release corpus beforehand, but do not evaluate it until
then. The eventual release command is:

```powershell
python -m benchmarks.locked_corpus assets\runtime\installed\2026.08.1 `
  D:\bidan-eval\release --profile plain-v1 `
  > D:\bidan-eval\reports\plain-v1-release.json
```

The aggregate report includes Wilson intervals, functional context and exact-transcription
metrics, ordered-component and contextual dictionary checks, negative-pointer categories,
held-out language classes, all required render strata, and the separately marked nonblocking
10 px stress result. A release run requires the official `test` split; a development split is
always reported as ineligible.

## Foreground Windows latency benchmark

This opt-in command displays 505 locked release fixtures on the primary screen, moves the
mouse pointer, exercises real MSS capture and popup rendering, discards five warmups, and
scores 500 fixed attempts without replacing failures. Do not use the computer during the run.

```powershell
python -m benchmarks.plain_live assets\runtime\installed\2026.08.1 `
  D:\bidan-eval\release D:\bidan-eval\reports\foreground-latency.json `
  --bundle-version 2026.08.1 --confirm-foreground
```

The foreground report contains aggregate machine, correctness, failure-stage, safety-counter,
sample-count, median, and p95 timing metadata. Captured pixels and recognized text remain
memory-only.

## Legacy optional evaluators

The version-two clean/subtitle/complex/morphology builder and evaluator commands remain
compatible for exploratory work. They are not part of `plain-v1` release eligibility.
`release_baseline.py` is likewise a deterministic synthetic precursor, not release evidence.
