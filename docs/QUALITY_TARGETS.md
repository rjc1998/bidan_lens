# Version-one quality gates

Version one is evaluated on plain websites and ordinary Windows desktop text. Subtitles,
games, comics, stylized text, and text over images remain useful optional measurements but
do not determine `plain-v1` release eligibility.

The release machine is Windows 10 22H2 x64 with a four-core AVX2 CPU and 8 GB RAM after
model warm-up. Primary targets are the expected release bar. Exceptional floors are a
fallback only after documented, materially different optimization attempts have reached a
credible plateau and the project owner explicitly approves the exception.

| Metric | Primary target | Exceptional floor |
| --- | ---: | ---: |
| Whole-eojeol OCR | at least 97% | 95% |
| Fully correct first popup | at least 92% | 88% |

Every required font-size and punctuation stratum must independently meet the corresponding
exceptional floor. Marked automatic corrections must have a false-promotion rate below
0.5%. Warm end-to-end latency must be at most 500 ms median and 1 second at p95.

## Plain-v1 corpus contract

The deterministic development and locked release corpora each contain 2,000 samples. The
development corpus uses only official UD `dev` splits; the release corpus uses only official
UD `test` splits. A separate 250-sample 10 px stress tier is reported but can neither pass
nor fail a release. A deterministic 200-sample development subset supports quick iteration.
Downloaded sources, fonts, KRDict data, and rendered corpora stay outside Git.

Required release sizes are 12, 14, 16, 18, 20, 24, 32, and 40 px, with 250 samples at each
size. The corpus balances browser and desktop renderers, five Korean font families, normal
and bold weights, light and dark themes, single- and multi-line layouts, four display scales,
and eight punctuation classes. Punctuation remains in sentence context but outside the
hoverable Korean eojeol.

Each version-three sample stores exact line, eojeol, target-span, pointer, and bounding-box
geometry from DOM or Qt layout. It also stores an expected first lemma, evaluator-owned
learner labels from published KAIST annotations, and English-bearing KRDict entries parsed
independently during construction. Production OCR, Kiwi output, and dictionary lookup never
choose or prune release samples.

The source lock pins immutable URLs, versions, SHA-256 hashes, byte sizes, and license
locations for UD Korean GSD/KAIST 2.18, open Korean fonts, and KRDict. The installed Malgun
Gothic file is hashed locally and is never redistributed. A corpus lock then hashes every
source record, license, renderer record, image, annotation, and quick-subset manifest. The
validator rejects tampering, missing evidence, unsafe paths, training splits, unknown
oracles, duplicate source identities, and development/release overlap.

## Measurements and reports

The `plain-v1` evaluator measures:

- whole-eojeol OCR accuracy and missing-eojeol rate;
- pointer target selection and containing-sentence span accuracy;
- first-result lemma and learner-label accuracy from ground-truth OCR input;
- exact first-result KRDict entry, definition, and sense-order fidelity;
- fully correct first popup, alternative-candidate recovery, and false promotions;
- warm median and p95 pipeline latency.

It reports aggregate values and 95% Wilson intervals, plus strata by size, font, renderer,
scale, punctuation, weight, theme, and layout. Release output contains no sample text or
pixels. Optional development diagnostics contain only stable sample IDs, failed stages, and
render strata.

Exact KRDict fidelity certifies faithful local dictionary presentation. It does not certify
contextual best-sense ranking or pedagogical quality. Synthetic browser and desktop fixtures
provide strong evidence for ordinary text, not arbitrary image backgrounds.

An opt-in foreground Windows run displays 505 locked fixtures, discards five warmups, and
collects 500 real capture-to-popup timings. It moves the pointer and uses the primary display,
so it requires explicit confirmation. Its report is aggregate-only and never persists live
screenshots or recognized text.

## Exceptional release process

Before approving an exception, record the baseline, material approaches attempted,
comparable per-stratum results, regressions, blockers, and remaining work. Every aggregate,
required-size, and punctuation exceptional floor must still pass. Privacy, latency,
packaging, provenance, failure handling, and clean-machine gates remain unconditional.
Release notes must publish the measured aggregate results and clearly identify any approved
missed primary target.
