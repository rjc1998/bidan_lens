# August 2026 production-asset baseline

This report records the first deterministic baseline for the exact hash-locked assets in
`assets/release-assets.lock.json`. All test images were generated and processed in memory.
No private screenshots, recognized strings, or individual errors were stored or logged.

## Results

| Category | Samples | Primary result | Median | p95 | Target status |
| --- | ---: | ---: | ---: | ---: | --- |
| Clean Windows fonts | 500 | 96.60% whole-eojeol exact | 18.6 ms | 23.2 ms | Passes 95% primary OCR target |
| Subtitle-style overlays | 300 | 97.00% whole-eojeol exact | 30.4 ms | 35.8 ms | Passes 90% primary OCR target |
| Complex colorful backgrounds | 200 | 80.50% whole-eojeol exact | 64.7 ms | 101.3 ms | Meets the 80-85% primary range |
| Learner morphology | 300 | 95.33% lemma/breakdown first; 93.33% including a first-result definition | 12.6 ms | 37.2 ms | Passes 90% primary morphology target |

The run used Windows on an AMD64 Family 26 Model 68 CPU with Python 3.12.6 and ONNX
Runtime's CPU provider. OCR timings begin with an already-created model session and end
after OCR document construction. Morphology timings cover Kiwi analysis, ranking, and
local SQLite lookup.

## Reproduction

Install the project and release extra, verify the assets, then run each category:

```powershell
bidan-lens-release-assets verify outputs assets/runtime/onnx
$env:PYTHONPATH = "src"
python benchmarks/release_baseline.py assets/runtime/onnx --category clean
python benchmarks/release_baseline.py assets/runtime/onnx --category subtitles
python benchmarks/release_baseline.py assets/runtime/onnx --category complex
python benchmarks/release_baseline.py assets/runtime/onnx --category morphology
```

The OCR samples deterministically select Korean headwords from the locked KRDict database
and render four Windows Korean fonts across varied sizes. Subtitle samples use high-contrast
outlined text on image-like gradients. Complex samples add smooth multicolor backgrounds,
shapes, outline variations, and small rotations. The morphology set uses 50 reviewed common
verbs across four ending patterns and 50 reviewed common nouns across two particle contexts.

## Plain-v1 schema-v4.2 follow-up

The corrected corpora are locked under `F:\bidan-lens-eval-ud218-v4.2`; the release split is
unevaluated. Independent direct KRDict conformance is now separated from analyzer correctness
and passes 243/243 unique groups. After conservative score-bounded multi-component promotion,
the 400-case development language tier is 86.50% overall, 93.50% for auxiliary cases, and
79.50% for multi-lexical cases. The remaining aggregate language failures are 31 primary
lemmas, 10 grammar roles, eight component roles, four component surfaces, and one
component-count mismatch. Internal Kiwi search now examines ten analyses while the popup remains
capped at five; primary-lemma failures with no matching navigable candidate decreased from 23
to 18.

The 200-case rendered quick tier is 96.43% whole-eojeol OCR, 74.50% functional context,
59.00% component accuracy, 45.00% fully correct popup, and 2.11% negative activation, with
271.72 ms median / 403.47 ms p95 automated latency. These are provisional development
measurements; a complete v4.2 render run has not been performed and release remains blocked.

The development/release lock SHA-256 values are
`46ced15df84f27bc858f8700c68cc9fa58f36f9407e6610bc3050d0056475567` and
`55babf1b1c53c101016b65b371bdd56b540e36aff6c9edbb93623c9e0acf6aeb`.
The latest aggregate-only language and quick reports are
`report-language-multilexical.json` and `report-quick-multilexical.json`; their SHA-256 values
are `06bb601d45c1a4c7e60a0c39efd9844e619a42594b379603eb662de1ec71c613` and
`58bb895d28816e6bd80286792577af0b8205c3f96cc33e7f53f720c7d498c21f`. The preceding v4.2
reports remain preserved under their original filenames.

## Superseded plain-v1 schema-v4 development evidence

The v4 corpora are under `F:\bidan-lens-eval-ud218-v4`. Development and release locks both
validate with 2,000 main, 250 stress, 400 held-out language, and 200 quick cases. The release
split remains unevaluated. The complete development results are:

| Development metric | Result |
| --- | ---: |
| Whole-eojeol OCR | 96.03% |
| Target selection | 92.80% |
| Functional context / exact transcription | 73.75% / 60.90% |
| Complete ordered components first | 57.95% |
| Exact KRDict fidelity first | 74.70% |
| Fully correct first popup | 42.30% |
| Held-out language overall | 72.75% |
| Multi-lexical / auxiliary | 76.00% / 69.50% |
| Direct KRDict conformance | 77.00% |
| Negative activation | 3.08% |
| False promotions | 0% |
| Automated latency | 223.77 ms median / 336.05 ms p95 |
| Nonblocking 10 px popup correctness | 31.20% |

Blank and English probes have zero activations; near-miss, punctuation, and whitespace are
2.25%, 9.21%, and 5.29%. The report SHA-256 is
`16fd9c33fac42f78c5754f05b532c510c7b92102fdf34a7dbae23ebeb1aa3ae4`.
The development and release lock SHA-256 values are
`104849023d87f8cb375acd88c857d4dd8ff7f89d7192e98f26940d00917849ce` and
`fae049ae16018e30c1f88788da3ebf23b77f26b93d6b887a01b31b5a7253c837`.
These provisional gates do not pass, so thresholds are not frozen and neither release nor
foreground evaluation has begun.

## Superseded schema-v3 development evidence

The preserved `F:\bidan-lens-eval-ud218` evidence predates contextual components and the
functional-context contract. Pinned acquisition completed outside Git, and complete development and release corpora
were independently built, locked, and validated. The release corpus uses only official
test splits and remains unevaluated. The development report uses all 2,000 primary samples;
the 250-sample 10 px stress tier remains nonblocking.

| Development metric | Result |
| --- | ---: |
| Whole-eojeol OCR | 96.19% (95% CI 95.94-96.43%) |
| Missing eojeols | 1.78% |
| Target selection | 92.80% |
| Exact containing-sentence span | 60.95% |
| Ground-truth-input lemma/breakdown first | 85.85% |
| Exact KRDict fidelity first | 85.25% |
| Fully correct first popup | 49.75% (95% CI 47.56-51.94%) |
| Alternative-candidate recovery | 83.95% |
| False promotions | 0% |
| Automated pipeline latency | 213.65 ms median / 331.67 ms p95 |
| Nonblocking 10 px OCR | 94.52% |

The primary corpus passes the aggregate 95% exceptional OCR floor but remains below the 97%
primary target. Required 32 and 40 px strata and the quotes and mixed punctuation strata miss
the 95% OCR floor. Exact sentence reconstruction and first-choice morphology/dictionary
ordering keep the full-popup result far below its 88% floor. The automated latency and
false-promotion gates pass, but neither can make this development candidate release eligible.

The aggregate report SHA-256 is
`7b06c6030319897dea5a3a3f698ac1e0b12cdfd6929ce72af7486ac685f0dc4f`.
The stable-ID-only diagnostic report SHA-256 is
`1136312eec4d6af81210770e2538af305a697ea02800ecb4b5e8c95a5e2185c1`.

## What this does not close

These exploratory synthetic results do not replace the locked `plain-v1` browser/desktop
corpus. Subtitle and complex-image figures are optional future metrics, not version-one
release gates.
The timings also exclude screen capture, queueing, pointer hit testing, popup rendering, and
cold model startup, so they do not by themselves close the pointer-to-popup latency gate.
Clean Windows 10 VM, mixed-DPI multi-monitor, packaged-build, false-correction, and complete
license-payload tests remain release blockers until separately recorded.
