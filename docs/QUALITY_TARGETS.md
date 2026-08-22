# Version-one quality gates

The targets are measured on a Windows 10 22H2 x64 computer with a four-core AVX2 CPU and
8 GB RAM after model warm-up.

The primary targets are the normal optimization goals and release expectation. Exceptional
release floors are not alternative targets, and reaching one does not justify ending
optimization while credible, materially different approaches remain.

| Corpus | Primary OCR target | Exceptional release floor |
| --- | ---: | ---: |
| Clean websites and standard fonts | at least 95% whole-eojeol accuracy | 90% |
| Subtitles and common image backgrounds | at least 90% | 85% |
| Games, comics, stylized or complex backgrounds | 80-85% | 70-75% |

Morphology acceptance uses a curated 300-item learner corpus. At least 90% of correctly
recognized forms should put the correct lemma and primary breakdown first; its exceptional
release floor is 85%. Marked automatic corrections must have a false-promotion rate below
0.5%.

Warm pointer-to-popup latency should be at most 500 ms median and 1 second at p95 for the
prioritized clean corpus. The benchmark set is 500 clean desktop samples, 300 subtitle
samples, 200 complex-background samples, and 300 morphology cases. Corpus images must be
licensed or synthetic and contain no private user captures.

## Exceptional release process

Releasing below a primary accuracy target is an exception, not the normal path. Consider
an exception only after repeated, materially different, measurement-driven optimization
attempts on the locked corpora demonstrate a genuine plateau and diminishing returns.

Before approval, prepare an exception report that records the baseline, every material
approach attempted, comparable results by category, regressions or tradeoffs, known
blockers, remaining risks, and plausible next work. The project owner must explicitly
approve an exception for each category that misses its primary target.

Every exceptional release floor must still be met; a result below any floor blocks release.
Latency, privacy, packaging, and failure-handling gates remain unconditional. Release notes
must publish the measured results, identify every missed primary target, and disclose the
approved exception rationale and remaining limitations.
