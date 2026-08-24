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

## Plain-v1 schema-v4.11 development follow-up

The current development-only corpus is locked under
`F:\bidan-lens-eval-ud218-v4.11\dev`. It contains 2,000 main, 250 stress, 400 held-out
language, and 200 quick cases and records the `viewport-v3` policy for both renderers. The policy
shifts only targets outside the 1280 by 720 image into a 10 px safe band and removes clipped words
from rendering and expected geometry together. The intermediate v4.10 card-anchoring build is
preserved as rejected evidence because its correction was broader than the viewport defect.

The v4.11 quick tier records 98.83% whole-eojeol OCR, 99.00% target selection, 91.00%
functional context, 72.50% exact sentence transcription, 88.00% component accuracy, 90.50%
exact KRDict fidelity, 80.50% fully correct first popups, 95.50% alternative recovery, and zero
false promotions. Automated latency is 219.73 ms median / 332.21 ms p95. There are 21 analysis,
16 context, and two target failures. Two near-miss probes activate, for 1.00% in that category;
all other negative categories are zero.

The v4.11 lock SHA-256 is
`b9cd0e46fcad9e3c3692c5fa2eb9de31cd693e7b9c4e8022e13476408d9c9da9`.
The aggregate quick report and privacy-safe diagnostic SHA-256 values are
`830637a9489b7e585e2ababef9edf711759fa22f533ecf2051cc47d3cea982a6` and
`f5d94680450559bb1626954cdfa32c4c7e6be50344fe3b55c02e33d3eaded68a`.
The complete run remains deferred, so the v4.11 held-out language tier and required strata have
not been evaluated. Accumulated candidate-builder changes mean v4.9 decisions cannot be mapped to
v4.11 by numeric ID without a fresh review audit.

## Historical plain-v1 schema-v4.9 development follow-up

The historical development-only corpus is locked under
`F:\bidan-lens-eval-ud218-v4.9\dev`. It contains 2,000 main, 250 nonblocking stress,
400 held-out language, and 200 locked quick cases. Its language tier is 90.50% overall,
93.00% auxiliary, and 88.00% multi-lexical, with 100% direct KRDict conformance across
244 independent groups. The multi-lexical exceptional floor therefore remains met.

The v4.9 oracle uses independent GSD/KAIST morphology to expose adverb, determiner, and
negative-copula components; refines broad GSD verb tags with UPOS roles; conservatively attaches
terminal noun suffixes; and orders the expected KRDict part-of-speech group first while retaining
other homographs in source order. The runtime rejects isolated component promotion when a
derivational word part would be omitted and can prefer a close inflected-verb interpretation
after a particle even when punctuation intervenes. Connective auxiliary context likewise
survives intervening punctuation, and richer multi-component promotion cannot discard an
already-supported particle feature. The 400-case language cross-check preserved 90.50%
overall, 93.00% auxiliary, and the required 88.00% multi-lexical result. Dictionary-confirmed
bound roots and adjective-forming suffixes are presented as one descriptive-verb component.

The accepted 200-case v4.9 quick result is 97.46% whole-eojeol OCR, 96.00% target selection,
89.50% functional context, 70.50% exact sentence transcription, 92.00% complete components,
94.50% exact KRDict fidelity, and 83.50% fully correct first popup. Alternative-candidate
recovery is 93.00%, false promotions remain zero, and automated latency is 207.95 ms median /
322.07 ms p95. The remaining categorical failures are eight target, 13 context, and 12 analysis;
the analysis details are four primary lemmas and eight component roles.

The context review contains 39 decisions: 18 incorrect line/sentence reconstructions, six
missed or merged OCR word boundaries, and 15 punctuation or structured-ASCII cases. Its v4.9
audit has 13 active cases, 26 resolved IDs, and no missing decisions. The first-popup review
contains 48 decisions and persists only stable IDs, categorical stages, decisions, and a
categorical summary. Its v4.9 state has 12 active cases and 36 resolved reviewed IDs; after the
newly exposed grammar-role case was classified, the decision history is 20 Kiwi-analysis errors,
12 corpus-oracle defects, eight equivalent learner interpretations, four genuinely ambiguous
cases, and four annotation-convention differences.
The reviewed grammar-role oracle defect is resolved by removing a particle label only when a
single noun-like component accounts for the target's complete punctuation-trimmed surface.
A defined pronoun that already leads by analyzer score is retained instead of being replaced by
a more fragmented determiner-plus-dependent-noun interpretation.
A score-bounded connective-auxiliary promotion experiment was rejected because it produced no
net popup gain and reduced exact KRDict fidelity from 93.50% to 93.00%.
A narrower accepted rule bounds only same-lemma action-versus-auxiliary alternatives while
preserving different-lemma contextual recovery.
An immediate paired-punctuation check can promote a close, dictionary-backed candidate already
present in the original analysis when the same sentence without only those wrappers supports it;
the displayed OCR and sentence context are not altered.
An audit of all eight active role cases against pinned KAIST UPOS, XPOS, and dependency evidence
corrected two review categories without changing an oracle or score. A wider wrapped-adverb
promotion was rejected because it produced byte-identical quick diagnostics and no metric change.
The four remaining primary-lemma cases have one annotation-convention decision and three
equivalent-learner decisions; none supports a general runtime correction.

The context reviewer's geometry-only inspection can select a single stable ID and reports
Unicode-category counts, per-eojeol geometry, confidence, spans, detector/segment provenance, and
text-equality signals without emitting text. A broad recognition-confirmed close-fragment merge
was rejected after it resolved three reviewed cases but introduced nine new failures. A bounded
overlapping-triplet recovery is accepted only when a narrow low-confidence leading sliver is
discarded and combined recognition exactly confirms the high-confidence middle-plus-trailing
surface. A second recovery joins only unusually close fragments isolated between ordinary word
gaps when confidence, character pitch, and combined recognition all agree. Together, the triplet
recovery and separately bounded 2+3 and 3+1 close-pair profiles resolved three cases without
creating a new context, target, or negative-pointer failure.
Two additional profiles recover shallowly overlapping 2+2 terminal or internally isolated 2+3
fragments only when surrounding geometry, character pitch, fragment confidence, and exact
combined recognition agree. They resolved two more context cases without regressions.
The remaining punctuation-wrapped CJK case is a recognizer-coverage limitation rather than safely
reconstructable text; another reviewed reconstruction case is a character-recognition mismatch
rather than a boundary defect.
The nine still-active reconstruction decisions were screened again using structural-only output:
three are recognition-dominated, one requires unsupported CJK coverage, four mix detector
fragmentation and recognition errors without an independently confirmed correction, and one
loses terminal punctuation. A terminal-punctuation retry was rejected after exact-sentence
accuracy fell from 67.00% to 66.50% without a context or popup gain.
The remaining active punctuation case drops two unsupported CJK-only regions. The three active
boundary cases respectively require internal splitting inside recognized regions, have a false
split geometrically indistinguishable from ordinary spaces, or lack independent combined-
recognition confirmation. A narrow low-confidence 1+2 merge experiment was rejected because it
produced byte-identical quick diagnostics and no metric change; global splitting remains
unchanged.

Privacy-safe target geometry review supported two narrow boundary recoveries without changing
the global OCR split threshold. Identical paired slash or dash wrappers can delimit a Hangul word
when every resulting part contains Hangul, while missing mandatory spacing before auxiliary
`했다` is restored only after a multi-syllable `-야` ending. Both retain proportional geometry
and cover single-segment recognition. Aggregate negative activation is now 0.00%; blank, English,
near-miss, punctuation, and all 191 whitespace probes have zero activations. The strict
per-category below-0.5% gate therefore passes. The earlier zero-ink hover-exclusion experiment
remains rejected because it reduced target selection to 85.50%, context to 76.00%, and popup
correctness to 64.00%.
Matched quote and bracket wrappers also recover an adjacent multi-syllable Hangul word, while a
directly attached one-syllable particle remains intact. Terminal `:`, `?`, or `!` punctuation
can recover a following Hangul word only when both sides contain Hangul. These rules recovered
four more target cases without changing the context-failure set or negative-pointer results.
Two of the eight remaining target failures have expected boxes below the 720 px captured
viewport and are corpus-construction defects. The builder now uses the same target-visibility
policy in Qt and Chromium: it shifts an off-screen target into the captured card, removes clipped
words from both rendering and expected geometry, fails closed if the target cannot fit, and
records the later viewport policy in renderer provenance. The locked v4.9 files and accepted
measurements remain unchanged; the correction is validated in v4.11 above.

The v4.9 development lock SHA-256 is
`f23f2388e580a889bd0ef363052ec72ab51c76ad57eacc68d3eca094242be5ab`.
The context-review, first-popup-review, and accepted quick diagnostic SHA-256 values are
`66b6cc918598dcad6d5517cc2db721396533d06fef8153ae52d144f31ff97720`,
`b397d3f1e0bf5eae15c2e438606320fb0ceed98c9f4e3cb7efaa9b07f4874a37`, and
`c6ffcf80053e7ca4ddb8330c8442e9d35fd7f8b57c5eb6c07a5eabc1c18a3f31`.
The accepted aggregate quick report SHA-256 is
`219814acb5946e7c2061135c4c2f52f862589bd696387d0c236471be9b31554c`.
The historical v4.9 complete render run remains unperformed. Current full-run work is governed by
the v4.11 quick blockers above. Thresholds are not frozen; the
untouched v4.2 release split and foreground benchmark remain unevaluated.

## Superseded plain-v1 schema-v4.5 development follow-up

The former development-only corpus is locked under
`F:\bidan-lens-eval-ud218-v4.5\dev`. Its 400-case held-out language tier is 90.50%
overall, 93.00% for auxiliary cases, and 88.00% for multi-lexical cases, with 100% direct
KRDict conformance across 244 independent groups. Multi-lexical accuracy therefore recovers
11 net cases from the v4.2 82.50% result and reaches the exceptional floor. Its remaining
multi-lexical failures are 18 primary lemmas, two grammar roles, two component roles, one
component surface, and one component count.

The local reviewer classified all 35 original v4.2 multi-lexical disagreements without
persisting corpus text: 16 equivalent learner interpretations, eight corpus-oracle defects,
five Kiwi-analysis errors, three annotation-convention differences, and three genuinely
ambiguous cases. After the general corrections, its audit reports 31 active cases, four
resolved stable IDs, and no stale decisions. The implemented corrections normalize the KAIST
derivational-adjective and copula conventions, restore dictionary-backed noun prefixes, and
conservatively attach terminal noun suffixes. They do not introduce sentence- or lemma-specific
overrides.

The context reviewer classified 39 stable IDs without persisting text: 18 incorrect line or
sentence reconstructions, six missed or merged OCR word boundaries, and 15 punctuation or
structured-ASCII cases. The accepted reconstruction changes recover 19 of those IDs. They
retain high-confidence numeric, uppercase, and version context, merge collinear fragments in
reading order, and remove physically overlapping duplicate fragments after edge-punctuation
normalization. The v4.5 oracle also corrects generic GSD component roles from independent UPOS
evidence. No global OCR splitting threshold was changed.

The accepted 200-case v4.5 quick result is 96.03% whole-eojeol OCR, 91.00% target selection,
81.00% functional context, 73.00% complete components, 79.50% exact KRDict fidelity, and
58.00% fully correct first popup, with 221.36 ms median / 332.36 ms p95 automated latency.
The interior Hangul-glyph hover rule lowers aggregate negative activation from the initial
v4.4 3.37% to 0.21%. Blank, English, near-miss, and punctuation probes are at zero; whitespace
remains 2/191, or 1.05%, and therefore fails the strict per-category below-0.5% gate.

Two broader OCR-boundary experiments were rejected during cross-validation. Adaptive visual
splitting removed the whitespace activations but reduced whole-eojeol OCR to 82.66% and popup
correctness to 27.00%. Raw CTC glyph boxes preserved 96.03% OCR but reduced target selection
to 46.00% and popup correctness to 26.00%. Neither experiment remains in the code.
An adaptive hover-gap exclusion also removed the two whitespace activations, but reduced target
selection to 87.00%, context to 77.00%, and popup correctness to 50.50%; it too was removed.

The v4.5 development lock SHA-256 is
`583d6f4cbaa36c9ead71e327196da016274a36c2b234c14c93c0e502e6432228`. The context-review
and accepted quick diagnostic SHA-256 values are
`9b4fe2fd3568dfade1c8a0d24b35e5b5bce8c4f2642834a0051b62314c7b509d` and
`7d1fed02501fd6069c358b3da62c6d809456666447558d9c093d6ed83843cf98`.
The complete 2,000-sample v4.5 render run is deferred because quick popup, functional-context,
and per-category negative-activation gates remain weak. Thresholds are not frozen; the
untouched v4.2 release split and the foreground benchmark remain unevaluated.

## Superseded plain-v1 schema-v4.2 follow-up

The corrected corpora are locked under `F:\bidan-lens-eval-ud218-v4.2`; the release split is
unevaluated. Independent direct KRDict conformance is now separated from analyzer correctness
and passes 243/243 unique groups. After conservative score-bounded multi-component promotion,
the 400-case development language tier is 88.00% overall, 93.50% for auxiliary cases, and
82.50% for multi-lexical cases. The remaining aggregate language failures are 26 primary
lemmas, nine grammar roles, eight component roles, four component surfaces, and one
component-count mismatch. Internal Kiwi search now examines ten analyses while the popup remains
capped at five; primary-lemma failures with no matching navigable candidate decreased from 23
to 17. An isolated-eojeol analysis is used only after dictionary-backed particle recovery and
only when the contextual leader remains undefined and every component in the richer split has
a local definition.

A text- and lemma-independent pairwise linear ranker was trained on 106 non-quick natural
multi-lexical development cases. It regressed five cases in cross-validation and eight held-out
cases without confidence gating. A threshold with zero cross-validation regressions still
produced five held-out recoveries and five regressions, so no learned weights were shipped.
Likewise, treating any final morpheme with a KRDict particle homograph as a particle would
mislabel 119 held-out and 51 quick cases. The implemented particle recovery is therefore limited
to an exact suffix after a fully dictionary-backed noun-component prefix.

The 200-case rendered quick tier is 96.43% whole-eojeol OCR, 74.50% functional context,
60.50% component accuracy, 46.50% fully correct popup, and 2.11% negative activation, with
231.20 ms median / 371.56 ms p95 automated latency. These are provisional development
measurements; a complete v4.2 render run has not been performed and release remains blocked.

The development/release lock SHA-256 values are
`46ced15df84f27bc858f8700c68cc9fa58f36f9407e6610bc3050d0056475567` and
`55babf1b1c53c101016b65b371bdd56b540e36aff6c9edbb93623c9e0acf6aeb`.
The latest aggregate-only language and quick reports are
`report-language-krdict-particle.json` and `report-quick-krdict-particle.json`; their SHA-256
values are `ed856242711e2b4e2d00effb7e7121f94d1435517fb214642185ddb2a60007ae` and
`0dfe94ad7d9be0fa0c9fa7d60cd570ee47c55797f92cdb2fad59a45463c5b898`. The preceding v4.2
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
