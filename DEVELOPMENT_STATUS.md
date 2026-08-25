# Development status

## Implemented in the repository

- local source-layout Python application and Windows tray shell;
- automatic scanning and hold-key activation modes;
- Paddle-compatible ONNX CPU detector/recognizer with CTC-guided eojeol crops, conservative
  visual-gap refinement, adaptive recognition width, conservative edge-punctuation recovery,
  paired punctuation-wrapper and mandatory auxiliary-boundary recovery, structured ASCII
  sentence context, exact reconstructed word geometry, and one low-confidence retry;
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
  200-sample quick subset, validated non-overlapping negative probes, and a separate
  250-sample 10 px stress tier;
- an independent v4 oracle for ordered components, contextual roles, verified spacing, and
  exact role-grouped KRDict entry/sense order, with 400 held-out language cases;
- local stable-ID language, functional-context, and first-popup disagreement review workflows
  whose persisted output contains categorical decisions but no sentence, expected, recognized
  text, definitions, or pixels, including fail-closed cross-lock carry-forward only for matching
  active IDs and popup failure stages;
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
`F:\bidan-lens-eval-ud218-v4.12\dev`; the untouched v4.2 release corpus remains under
`F:\bidan-lens-eval-ud218-v4.2\release` and has not been evaluated. Earlier roots remain
preserved. The v4.12 rebuild contains 2,000 main, 250 stress, 400 held-out language, and 200
quick cases, is hash-locked, and passes corpus validation. It uses the current corpus builder and
the `viewport-v3` renderer policy. The intermediate v4.10 card-anchoring experiment is preserved
but rejected because it changed more already-visible geometry than the viewport defect required.

The accepted v4.12 200-case quick tier records 98.96% whole-eojeol OCR, 99.00% target
selection, 93.00% functional context, 73.50% exact sentence transcription, 94.00% component
accuracy, 95.00% exact KRDict fidelity, and 89.00% fully correct first popups, with 210.77 ms
median / 322.92 ms p95 automated latency. Alternative-candidate recovery is 96.50% and false
promotions remain zero. Its remaining failures are eight analysis cases (four primary lemmas and
four component roles), 12 context cases, and two target cases. Aggregate and per-category negative
activation are 0.00%, so the quick popup floor and strict negative gate pass.

The complete v4.12 development evaluation has now run against the current analyzer cleanup.
Its 2,000 main cases record 97.98% whole-eojeol OCR, 97.10% target selection, 86.85% functional
context, 70.00% exact sentence transcription, 88.95% component accuracy, 92.40% exact KRDict
fidelity, 77.45% fully correct first
popups, and 92.75% alternative recovery. False promotions remain zero and automated latency is
225.41 ms median / 338.58 ms p95. Privacy-safe diagnostics contain 58 target, 205 context, and
188 analysis failures; the analysis stages are 70 primary lemmas, 88 component roles, 12 component
surfaces, nine component counts, and nine grammar roles. Three additional records fail only a
negative-pointer category.

The nonblocking 250-case stress tier records 94.19% OCR, 96.00% target selection, 67.20%
functional context, and 63.20% fully correct first popups. The 400-case held-out language tier
records 88.25% overall, 90.00% auxiliary, 86.50% multi-lexical, and 100% direct KRDict
conformance. Aggregate main negative activation is 0.14%; blank, English, and near-miss probes
remain at zero, whitespace is four of 1,931 (0.21%), and punctuation is nine of 1,582 (0.57%).
The correction, dictionary-conformance, and latency gates pass, but the primary, exceptional-floor,
and per-category negative-activation gates fail.

The context reviewer now has a separately scoped full-tier mode so quick and 2,000-case decision
reports cannot be mixed. Full cases can be inspected in a selected batch with one OCR model
initialization and categorized one stable ID at a time, writing each categorical decision
immediately. The complete 222-case review contains 89 non-target OCR transcription errors, 69
punctuation or structured-text cases, 42 missed or merged OCR word boundaries, and 22 incorrect
line/sentence reconstructions. The added transcription category
covers substitutions or omissions outside the correct target when line reconstruction and target
geometry are otherwise intact. The full report contains only its corpus ID, review scope, stable
IDs, categorical decisions, and counts. The current full diagnostics have 205 active context
cases; the fail-closed audit finds 17 resolved reviewed IDs, all 205 active IDs reviewed, and no
missing decisions. The decision report SHA-256 is
`a4c0f528cd6e2b945c8eb0600af8797b0666d96709641551ed5fd3aa8e88ba66`.

Three independently reviewed reconstruction cases contained a one-character eojeol centered
inside a two-character eojeol while exactly repeating its punctuation-normalized suffix. The
accepted cleanup removes only that suffix-specific contained fragment, preserves the protected
unrelated-character regression, and resolves all three stable IDs. On the quick tier it recovers
one context and popup case without changing target selection, component or dictionary accuracy,
false promotions, or negative activation.

Compared with the earlier full report, the five accepted cleanups resolve 21 context IDs without
introducing a new context failure. The preceding cleanup permits a one-pixel overlap of at most 7.5%
of a small line only under the existing exact combined-recognition duplicate profile. It resolves
two full-tier context cases while leaving the quick diagnostics byte-identical. The same profile
now accepts an ASCII digit as the leading artifact only when recognizing the union at 99% or
better omits it and exactly reproduces the following Hangul word. This resolves three additional
full-tier cases without a new failure or quick-tier change.

The first-popup reviewer now has the same separately scoped full-tier mode, repeated stable-ID
batch inspection, structure-only output, and single-ID categorical recording. Full reports use
the `first_popup_analysis_full` kind and cannot be mixed with quick reports. The first 40 decisions
contain 13 Kiwi-analysis errors, 12 equivalent learner interpretations, nine annotation-convention
differences, four corpus-oracle defects, and two genuinely ambiguous cases. Against the current
diagnostics, 37 reviewed IDs remain active, three are resolved, and 151 active analysis IDs remain
unreviewed. The report persists no corpus or analysis text; its SHA-256 is
`e009f6d9b11556f1f5703cd6eb0c0a8ab2b0c20b8bc31ba54cefbfa29f3087bd`.

The second full-tier review batch exposed repeated noun-plus-`화` derivations that Kiwi split into
a noun, derivational noun suffix, and action-verb suffix even though KRDict contains the complete
verb. The accepted analyzer joins only `noun + 화 + 하/되` and only when the exact complete verb is
dictionary-backed. It resolves four main primary-lemma failures, including both reviewed examples,
without adding a failure. Quick diagnostics remain byte-identical, and stress and held-out language
results are unchanged.

Geometry-only review showed that the two former near-miss points were inside real words on the
line adjacent to their targets. Probe construction now tests lower, upper, right, and left
adjacent points in order and emits only a point inside the viewport and outside every oracle
eojeol. The v4.12 rebuild retains 200 near-miss probes and removes both false activations without
changing any substantive quick failure ID or stage.

The v4.11 rebuild incorporated accumulated corpus-oracle and candidate-selection fixes made
after v4.9 was rendered. Consequently, a numeric sample ID can refer to a different independent
source record across those versions. Existing v4.9 categorical review decisions remain valid
historical evidence but must not be transferred to v4.11 by ID without a fresh audit. The v4.11
held-out language tier was not evaluated. The v4.12 tier has now been evaluated only as part of
the complete development run described above.

Fresh v4.12 privacy-safe reviews are complete and contain no corpus text or pixels. The popup
review retains 20 decisions: seven Kiwi-analysis errors, five annotation-convention
differences, seven equivalent learner interpretations, and one genuinely ambiguous case. The
context review retains 16 decisions: seven incorrect line/sentence reconstructions, four missed
or merged word boundaries, and five punctuation or structured-text handling cases. Its current
audit has 13 active cases and three resolved reconstruction IDs. The quick popup audit has eight
active cases and 12 resolved IDs; both audits have no missing or stale IDs. The review-supported analyzer
now keeps a dictionary-defined
whole noun that already leads by score instead of replacing it with a richer but fragmented noun
analysis. This recovered one case without changing OCR, target selection, context, alternative
recovery, false promotions, or negative activation.

The reconstruction review also supported removing a duplicate one-character eojeol when it is
fully contained in a longer eojeol, matches one of that eojeol's characters, and has normal
single-character pitch. This recovered two cases and improved OCR, functional context, exact
transcription, and popup correctness without a target, alternative, promotion, or negative-probe
regression.

A second review-supported rule handles a low-confidence one-character Hangul sliver only when it
touches or slightly overlaps a following Hangul word, is no wider than 90% of that word's character
pitch, and recognizing their union at 99% confidence or better exactly reproduces the following
word. The more specific exact-confirmed triplet recovery runs first so a real final syllable is not
stranded. This resolves 11 further full-tier context cases without a new context failure and
preserves the accepted quick-tier OCR, target, popup, promotion, and negative-pointer results.

A third exact-confirmation rule merges an overlapping pair only when its surfaces share exactly
one boundary syllable, one side is below 80% confidence, the other is at least 95%, and recognizing
their union at 99.8% confidence or better exactly reproduces the deduplicated surface. This
resolves two reviewed suffix-overlap reconstruction cases without a new full-tier context failure
or any quick-tier change.

Reported-speech connectives are now passed into learner-role construction. A following lexical
verb is no longer relabeled as a helping verb merely because the same headword also has an
auxiliary dictionary sense. This resolved one reviewed component-role case and improved component
accuracy, KRDict fidelity, popup correctness, and alternative recovery without an OCR, target,
context, promotion, or negative-probe regression.

The grammatical `-게 되다` construction now prefers a same-lemma, same-boundary, dictionary-backed
helping-verb candidate within a separately bounded 10-point score margin. It runs after isolated
role corroboration so the eojeol alone cannot override sentence-level grammar. This resolves two
main component-role cases, one stress case, and one held-out auxiliary case without a quick-tier,
upstream, alternative, promotion, or negative-probe regression.

For otherwise identical nominal interpretations, a lower-ranked candidate may now be promoted
within a 2.5-point score margin only when KRDict's default homograph order prefers every differing
learner role. This resolved two reviewed noun/pronoun/determiner disagreements and one
proper/common-noun convention difference without changing KRDict fidelity, OCR, target selection,
context, promotions, or negative probes.

When surrounding punctuation or fragmentary context changes only a verb component's learner
role, the isolated eojeol analysis may corroborate an otherwise identical alternative within a
2-point score margin. This resolved two action/helping-versus-descriptive disagreements while
preserving every upstream and safety metric; isolation does not override nominal or adverb roles.

The existing dictionary-backed complete multi-component promotion now uses the same 2-point
score limit as complete inflected-word recovery. This recovered one reviewed lexicalized-verb
versus main-plus-helping-verb interpretation; candidates still require at least two fully defined
components and cannot discard a particle feature.

For inflected targets, a more distant contextual main-plus-helping-verb interpretation may now be
promoted when an isolated analysis independently corroborates the exact component structure. The
isolated and contextual score gaps are bounded separately; every component must be dictionary
backed, a helping verb must be present, no word part may be left unrepresented, and an existing
particle feature cannot be discarded. Exact dictionary base forms are not decomposed. This
resolved four reviewed equivalent-interpretation cases without introducing a quick-tier failure.

The isolated undefined-component fallback now accepts an equal-count dictionary-backed analysis
when it changes the lemma, exposes a particle or verb ending, and its only otherwise unrepresented
word part is the contractible copula. This recovered one bracketed dependent-noun contraction
without weakening the derivational-word-part guard.

### Historical v4.9 reviewed evidence

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

The accepted v4.9 200-case quick tier records 97.46% whole-eojeol OCR, 96.00% target
selection, 89.50% functional context, 70.50% exact sentence transcription, 92.00% component
accuracy, 94.50% exact KRDict fidelity, and 83.50% fully correct first popups, with 207.95 ms
median / 322.07 ms p95 automated latency. Alternative-candidate recovery is 93.00% and false
promotions remain zero. This is a 37-point popup gain over v4.2 and a 25.5-point gain over the
accepted v4.5 result. The remaining quick failures are 12 analysis cases (four primary lemmas
and eight component roles), 13 context cases, and eight target cases.

Privacy-safe target geometry review supported two narrowly bounded OCR corrections. Identical
paired slash or dash characters can recover the word they wrap when every resulting part
contains Hangul, and a missing mandatory boundary before auxiliary `했다` can be restored only
after a multi-syllable `-야` ending. Both rules preserve proportional word geometry and also run
when recognition exposes only one segment. They recovered six target failures without changing
the global OCR splitting threshold. Aggregate negative activation is now 0.00%; all blank,
English, near-miss, punctuation, and 191 whitespace probes have zero activations, so the strict
per-category below-0.5% gate passes. The earlier zero-ink hover-exclusion experiment remains
rejected because it reduced target selection to 85.50%, context to 76.00%, and popup correctness
to 64.00%.
Matched curly/straight quote or bracket wrappers now use the same boundary recovery when the
following word has multiple syllables; a directly attached one-syllable particle remains intact.
A separate rule splits only after terminal `:`, `?`, or `!` punctuation when Hangul occurs on
both sides. These additions recovered four more targets without a context or negative-pointer
regression. Target geometry also exposed two remaining samples whose expected targets are below
the 720 px captured viewport; those are corpus-construction defects, not runtime OCR failures.
The corpus renderers now correct that construction defect by shifting only an otherwise
off-screen target into a 10 px safe band within the captured viewport. Words clipped by the
viewport are omitted from both image and expected geometry, the complete target is checked
against the 1280 by 720 image, and renderer provenance carries a `viewport-v3` policy suffix.
Long target-last text is covered in both Qt and Chromium. This correction is exercised by the
locked v4.11 corpus above; the locked v4.9 corpus and its historical measurements are unchanged.

The first-popup reviewer has 48 stable-ID decisions and no persisted corpus text. Against
v4.9, 12 cases remain active and 36 reviewed IDs are resolved. The complete decision history is
20 Kiwi-analysis errors, 12 corpus-oracle defects, eight equivalent learner interpretations,
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
The four remaining primary-lemma cases were rechecked against the pinned annotation evidence and
local KRDict entries. Their decisions are one annotation-convention difference, three equivalent
learner interpretations, and no review-supported general runtime correction.

The historical v4.9 multi-lexical result passed, but the complete v4.12 development run now shows
that main popup, functional context, required render strata, held-out multi-lexical analysis, and
punctuation activation still block release evidence. The next review target is the 151 unreviewed
privacy-safe main analysis failures: 45 primary lemmas, 78 component roles, 12 component surfaces,
nine component counts, and seven grammar-role cases. General corrections supported by that review
should be cross-validated against the quick, full, stress, and held-out tiers before returning to
the nine punctuation activations. Thresholds are not frozen, and neither the untouched release
split nor the 500-attempt foreground benchmark has been run. See
`docs/RELEASE_BASELINE_2026-08.md` for the measurement breakdown.

## Required before a public v1 release

- publish the validated asset bundle at an immutable project release location and record
  its checksum outside the bundle;
- improve against the locked development corpus until the mandatory floors pass, then
  record the complete release report against the untouched official test split;
- meet the aggregate and every size/punctuation exceptional floor, the false-promotion
  gate, and the primary OCR/fully-correct-popup targets (or explicitly approve a documented
  exceptional release);
- preserve the passing quick gates while improving full first-popup correctness, functional
  context, held-out multi-lexical analysis, required render strata, and punctuation activation;
- run the opt-in foreground benchmark with five warmups plus 500 fixed attempts, meeting
  correctness and latency targets with zero safety violations;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
