# Development status

## Implemented in the repository

- local source-layout Python application and Windows tray shell;
- automatic scanning and hold-key activation modes;
- Paddle-compatible ONNX CPU detector/recognizer with CTC-guided eojeol crops, conservative
  visual-gap refinement, adaptive recognition width, conservative edge-punctuation recovery,
  structured ASCII sentence context, exact reconstructed word geometry, and one
  low-confidence retry;
- immutable OCR lines, glyphs, eojeols and conservative whole-eojeol pointer hit testing
  that requires an interior Hangul-glyph hit;
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
- local stable-ID language, functional-context, and first-popup disagreement review workflows
  whose persisted output contains categorical decisions but no sentence, expected, recognized
  text, definitions, or pixels;
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

The current development corpus is locked under
`F:\bidan-lens-eval-ud218-v4.9\dev`; the untouched v4.2 release corpus remains under
`F:\bidan-lens-eval-ud218-v4.2\release` and has not been evaluated. Earlier roots remain
preserved. The v4.9 rebuild contains 2,000 main, 250 stress, 400 held-out language, and 200
quick cases.

The 400-case v4.9 language tier records 90.50% fully correct first popups, 93.00% auxiliary
correctness, 88.00% multi-lexical correctness, and 100% independent direct KRDict conformance
across 244 unique groups. This meets the multi-lexical exceptional floor after recovering
11 net cases from the 82.50% v4.2 result. The implemented general corrections normalize the
KAIST derivational-adjective convention, keep the KAIST copula out of learner particle labels,
recover dictionary-backed noun prefixes, conservatively attach terminal noun suffixes, expose
adverb/determiner/negative-copula components, and align expected KRDict homograph ordering with
the independently parsed part-of-speech evidence.
All 35 original v4.2 multi-lexical disagreements have stable-ID categorical decisions; the
current audit has 31 active cases, four resolved IDs, and no stale decisions.

The context reviewer classified 39 development disagreements as 18 incorrect line/sentence
reconstructions, six missed or merged OCR word boundaries, and 15 punctuation or structured-
ASCII cases. Its audit has 13 active cases, 26 resolved IDs, and no missing decisions. Based on
that review, same-row OCR fragments are reconstructed in reading order, physically overlapping
duplicate fragments are removed conservatively, and high-confidence numeric, uppercase, and
version identifiers are retained as non-hoverable sentence context even when detected as a
separate same-row region. Extremely narrow single-character fragments contained inside a much
wider eojeol are removed with sentence-span repair. A geometry-only inspection mode exposes IDs
and, optionally for one stable ID, Unicode-category counts, per-eojeol boxes, confidence, spans,
adjacency signals, detector regions, raw segmentation, and recognition equality flags, but no
corpus text. A separate read-only target/probe geometry mode covers stable IDs that fail target
selection and can optionally include the same privacy-safe raw segmentation evidence. A broad
recognition-confirmed close-fragment merge resolved three reviewed context
cases but created nine new failures, so it was rejected and removed. Two much narrower recoveries
handle only a low-confidence one-character sliver overlapping a high-confidence word fragment and
trailing character when combined recognition exactly confirms the latter two fragments, or an
unusually close 2+3-character pair isolated between ordinary word gaps when character pitch,
confidence, and combined recognition all agree. A separately bounded 3+1-character profile
recovers an isolated final syllable only when neighboring gaps, character pitch, very high
per-fragment confidence, and exact combined recognition all agree. Two overlap profiles
additionally recover either a terminal 2+2 pair or an internally isolated
2+3 pair only when shallow overlap, character pitch, surrounding geometry, per-fragment
confidence, and exact combined recognition all agree. The remaining punctuation-
wrapped CJK case is a recognizer-coverage limitation rather than safely reconstructable text,
and one reviewed reconstruction case is a character-recognition mismatch rather than a word-
boundary defect. The corpus oracle maps independent GSD UPOS labels to learner roles instead of
using a generic `word` placeholder.

All nine remaining cases categorized as line/sentence reconstruction have received a second
privacy-safe structural review. Three are recognition-dominated, one requires unsupported CJK
coverage, four combine detector fragmentation with recognition errors without a single safe
reconstruction signal, and one loses terminal punctuation. A recognition-confirmed terminal-
punctuation recovery was rejected because the 200-case quick tier lost one exact-sentence case
without improving functional context or popup correctness. No runtime change from that review is
retained; the accepted metrics below remain unchanged.

The other four active context cases were also re-audited. The sole punctuation/structured case
omits two CJK-only regions that the Korean recognizer does not support. Of the three boundary
cases, one needs two spaces recovered inside high-confidence recognized regions, one false split
has the same visual gap as ordinary spaces and an unrecognized terminal mark, and one candidate
1+2-character merge is not confirmed by combined recognition. A narrowly profiled merge for the
last case was rejected because it produced byte-identical quick diagnostics and no metric change.
No global OCR splitting threshold was changed.

The accepted v4.9 200-case quick tier records 96.35% whole-eojeol OCR, 91.00% target
selection, 84.50% functional context, 92.00% component accuracy, 94.50% exact KRDict
fidelity, and 78.50% fully correct first popups, with 222.04 ms median / 334.04 ms p95
automated latency. This is a 32-point popup gain over v4.2 and a 20.5-point gain over the
accepted v4.5 result while preserving OCR, target selection, and context. The remaining quick
failures are 12 analysis cases (four primary lemmas and eight component roles),
13 context cases,
and 18 target cases. Aggregate negative activation remains 0.21%; blank, English, near-miss, and
punctuation probes have zero
activations. Two of 191 whitespace probes still activate (1.05%), so the required per-category
below-0.5% gate does not pass. A zero-ink hover-exclusion experiment removed only one activation
while reducing target selection to 85.50%, context to 76.00%, and popup correctness to 64.00%;
it was rejected and removed. Target/probe geometry review found that both remaining whitespace
activations come from adjacent Korean words already merged into one raw OCR segment. One retains
internal punctuation, but the other has no model-space or segment boundary, so there is no shared
safe hover-only correction; global OCR splitting remains unchanged.

The first-popup reviewer has 48 stable-ID decisions and no persisted corpus text. Against
v4.9, 12 cases remain active and 36 reviewed IDs are resolved. The complete decision history is
21 Kiwi-analysis errors, 12 corpus-oracle defects, seven equivalent learner interpretations,
four genuinely ambiguous cases, and four annotation-convention differences. Review-supported
runtime changes reject isolated component promotion when it leaves an unrepresented word part
and allow close, dictionary-backed inflected-verb interpretations when they completely represent
the target, including after a particle when punctuation intervenes. Inflected-candidate
promotions are score-bounded, and connective auxiliary context now survives intervening
punctuation. Richer multi-component promotion cannot discard an already-supported particle
feature, and dictionary-confirmed bound roots combine with adjective-forming suffixes into one
learner component. Review-supported oracle corrections preserve all homographs while ordering
the expected part-of-speech group first.
They also remove a particle label when one noun-like component already accounts for the complete
target surface, since that leaves no independently annotated particle surface. A defined pronoun
that already leads by analyzer score is kept intact instead of being replaced by a more
fragmented determiner-plus-dependent-noun analysis.
A score-bounded connective-auxiliary promotion experiment was rejected because it produced no
net popup gain and reduced exact KRDict fidelity from 93.50% to 93.00%.
A narrower rule bounds only same-lemma action-versus-auxiliary alternatives; different-lemma
contextual recovery remains unchanged. It resolves one reviewed role error without reopening the
previously recovered contextual lemma case.
A punctuation-wrapper check can also promote an already-present, dictionary-backed candidate
when analysis of the same sentence without only the immediate paired wrappers supports it and
its original score is within 1.0. Displayed OCR and sentence context remain unchanged.
An eight-case pinned-source audit corrected two categorical decisions without changing expected
text or scores: one broad KAIST noun subtype is an annotation-convention difference, while one
explicit `ADV/mag/advmod` record is a Kiwi role-ranking error. A wider wrapped-adverb promotion
was rejected because it produced byte-identical quick diagnostics and no metric change.

The multi-lexical gate now passes, but rendered popup, functional-context, per-category
negative-activation, and required-stratum gates still block release and foreground evidence.
The complete 2,000-sample v4.9 render evaluation remains deliberately deferred until the quick
language/popup path materially improves. Thresholds are not frozen, and neither the untouched
release split nor the 500-attempt foreground benchmark has been run. See
`docs/RELEASE_BASELINE_2026-08.md` for the measurement breakdown.

## Required before a public v1 release

- publish the validated asset bundle at an immutable project release location and record
  its checksum outside the bundle;
- improve against the locked development corpus until the mandatory floors pass, then
  record the complete release report against the untouched official test split;
- meet the aggregate and every size/punctuation exceptional floor, the false-promotion
  gate, and the primary OCR/fully-correct-popup targets (or explicitly approve a documented
  exceptional release);
- preserve the now-passing 88.00% held-out multi-lexical tier while improving rendered
  functional context, first-popup correctness, and per-category negative activation;
- run the opt-in foreground benchmark with five warmups plus 500 fixed attempts, meeting
  correctness and latency targets with zero safety violations;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
