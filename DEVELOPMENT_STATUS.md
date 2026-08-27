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
`F:\bidan-lens-eval-ud218-v4.15\dev`; the untouched v4.2 release corpus remains under
`F:\bidan-lens-eval-ud218-v4.2\release` and has not been evaluated. Earlier roots remain
preserved. The v4.15 rebuild contains 2,000 main, 250 stress, 400 held-out language, and 200
quick cases, is hash-locked, and passes corpus validation. It uses the current corpus builder and
the `viewport-v3` renderer policy. The intermediate v4.10 card-anchoring experiment is preserved
but rejected because it changed more already-visible geometry than the viewport defect required.

The accepted v4.15 200-case quick tier records 99.23% whole-eojeol OCR, 99.00% target
selection, 95.00% functional context, 75.50% exact sentence transcription, 94.00% component
accuracy, 95.00% exact KRDict fidelity, and 90.50% fully correct first popups, with 226.82 ms
median / 354.76 ms p95 automated latency. Alternative-candidate recovery is 96.50% and false
promotions remain zero. Its remaining failures are nine analysis cases (five primary lemmas and
four component roles), eight context cases, and two target cases. Aggregate and per-category negative
activation are 0.00%, so the quick popup floor and strict negative gate pass.

The aggregate report SHA-256 is
`ac61efc72a850b6ca8e4c2c1dea7a9d3b51054ed8cc982b8500d556fc2ff780f`; the
privacy-safe diagnostic SHA-256 is
`de2b5c7ae2a86a501e222ece53b8a922529658ad245ce12c6a59164a9faeba3c`.

The complete v4.15 development evaluation has now run against the current OCR and analyzer
cleanup. Its 2,000 main cases record 98.20% whole-eojeol OCR, 97.10% target selection, 88.75%
functional context, 71.70% exact sentence transcription, 92.85% component accuracy, 95.15% exact
KRDict fidelity, 83.25% fully correct first popups, and 94.20% alternative recovery. False
promotions remain zero and the accepted follow-up is 231.84 ms median /
346.59 ms p95. Privacy-safe diagnostics contain 58 target, 167 context, and 110 analysis failures;
the analysis stages are 47 primary lemmas, 53 component roles, four component counts, and six
grammar roles. No component-surface failures remain. The nine stable IDs with negative
activations also fail target selection; one contains both punctuation and whitespace activation.

The nonblocking 250-case stress tier records 94.15% OCR, 96.00% target selection, 67.20%
functional context, 93.60% component accuracy, and 63.20% fully correct first popups. The 400-case
held-out language tier records 92.00% overall, 96.00% auxiliary, 88.00% multi-lexical, and 100%
direct KRDict conformance. Aggregate main negative activation is 0.11%; blank, English, and
near-miss probes remain at zero, whitespace is four of 1,931 (0.21%), and punctuation is six of
1,582 (0.38%). The correction, dictionary-conformance, latency, and aggregate/per-category
negative-activation gates pass, but the primary and exceptional floors fail.

The full aggregate report SHA-256 is
`0f6b833b11865024157264a3e44105ecb82a3a52f3a4707d0123b8daac3de04e`; the
privacy-safe diagnostic SHA-256 is
`02bbceeb9da52d95243ad8326a8694bc1ed35158e706c7124a70196a0833538f`.

The context reviewer now has a separately scoped full-tier mode so quick and 2,000-case decision
reports cannot be mixed. Full cases can be inspected in a selected batch with one OCR model
initialization and categorized one stable ID at a time, writing each categorical decision
immediately. The current 205-decision review contains 88 non-target OCR transcription errors, 68
punctuation or structured-text cases, 42 missed or merged OCR word boundaries, and seven incorrect
line/sentence reconstructions. The added transcription category
covers substitutions or omissions outside the correct target when line reconstruction and target
geometry are otherwise intact. The full report contains only its corpus ID, review scope, stable
IDs, categorical decisions, and counts. The current full diagnostics have 167 active context
cases; the v4.15 fail-closed audit finds every active ID reviewed, with no missing decisions and
38 resolved IDs. Cross-lock carry-forward accepts a prior corpus ID while still requiring the same
review scope and every current stable ID. The decision report SHA-256 is
`34e4bc6e5981dfba10f48b2478884c8def1ffb8c626f3307527d3d34a42def30`.

Three independently reviewed reconstruction cases contained a one-character eojeol centered
inside a two-character eojeol while exactly repeating its punctuation-normalized suffix. The
accepted cleanup removes only that suffix-specific contained fragment, preserves the protected
unrelated-character regression, and resolves all three stable IDs. On the quick tier it recovers
one context and popup case without changing target selection, component or dictionary accuracy,
false promotions, or negative activation.

One remaining reconstruction case consisted of a complete detector fragment followed by a
single-eojeol detector fragment whose exact text and box overlapped the first eojeol on both axes.
The accepted spatial-duplicate cleanup retains both geometry records while remapping the smaller
fragment to the existing sentence span instead of appending it. The exact full diagnostic
comparison removes only `dev-plain-1062`, with no addition or stage change; quick,
stress, language, upstream, promotion, and negative-pointer results remain unchanged.

Two further reviewed reconstruction cases contained a low-confidence one-character box fully
inside a very-high-confidence word of at least three characters. The accepted cleanup removes the
inner box only below 60% confidence, when the containing word is at least 99% confident and the
inner width is no more than 16% of the containing width. The exact full diagnostic comparison
removes only `dev-plain-1526` and `dev-plain-1671`, with no added or
changed record. Functional context and first-popup correctness rise to 87.00% and 81.00%;
exact sentence transcription rises to 70.15%, and quick diagnostics remain byte-identical.

A separate reviewed case contained a below-50%-confidence one-character false recognition nearly
aligned with the leading edge of a 99.9%-confident two-character word. The accepted cleanup
requires leading edges within one pixel, at least 80% vertical overlap, and an artifact no wider
than 25% of the containing word. The exact full diagnostic comparison removes only
`dev-plain-1398`. Functional context and first-popup correctness rise to 87.05% and
81.05%, exact sentence transcription rises to 70.20%, and no other failure record changes.

One final reviewed reconstruction case contained two mutually corroborating artifacts across
overlapping detector fragments: a structured identifier's repeated final digit and a copied
Hangul syllable aligned with the complete following word. The accepted merge removes both only
when the identifier suffix, word prefix, confidence, and geometry all agree. The exact comparison
removes only `dev-plain-1316`; functional context, exact sentence transcription, and
first-popup correctness rise to 87.10%, 70.25%, and 81.10%.

The first word-boundary batch reviews ten stable-target cases where OCR inserted one space inside
an expected word. Two are defensible spacing interpretations and several others lack reliable
spacing evidence. One case has a uniquely isolated pair of one-syllable fragments with zero gap,
wide gaps on both sides, compatible pitch, strong component confidence, and at least 99.99%
combined recognition of their exact concatenation. The accepted profile removes only
`dev-plain-1420`; functional context, exact sentence transcription, and first-popup
correctness rise to 87.15%, 70.30%, and 81.15%.

The second word-boundary batch groups all 41 active cases by component length, overlap, surrounding
gaps, confidence, character pitch, and exact combined recognition. Three reviewed false splits
share a shallow 2+1-syllable overlap, but only two are internal to a line. A separate internal
profile requires clear gaps on both sides, compatible pitch, at least 99.7% confidence for the
two-syllable fragment, and at least 99.97% exact combined recognition. The accepted comparison
removes only `dev-plain-1210` and adds or changes no failure record. Functional context,
exact sentence transcription, and first-popup correctness rise to 87.20%, 70.35%, and 81.20%;
quick diagnostics remain byte-identical.

The third word-boundary batch reviews both active internal 1+4-syllable overlaps. Their candidate
fragments overlap by one pixel while the neighboring boundaries remain positive, character pitch
is compatible, the four-syllable fragment is at least 99.75% confident, and recognizing the union
at 99.75% or better exactly returns all five syllables. The accepted comparison removes only
`dev-plain-0873` and `dev-plain-1421`, with no added or changed failure record.
Whole-eojeol OCR rises to 98.02%, functional context to 87.30%, exact sentence transcription to
70.45%, and first-popup correctness to 81.30%; quick diagnostics remain byte-identical.

The fourth word-boundary batch reviews all four active 1+2-syllable false splits. Three have an
overlapping or touching neighbor and do not support safe isolated recovery. The accepted profile
applies only when both neighboring gaps are wider, the one- and two-syllable fragments are at
least 99.88% and 99.98% confident respectively, character pitch is compatible, and recognizing
the union at 99.99% or better exactly returns the concatenation. It removes only
`dev-plain-1150`, with no added or changed failure record.
Functional context rises to 87.35%, exact sentence transcription to 70.50%, and first-popup
correctness to 81.35%; quick diagnostics remain byte-identical.

The fifth word-boundary batch reviews the two remaining 2+1-syllable false splits. The internal
case already matched the accepted exact-recognition profile but missed its pitch boundary only by
floating-point roundoff, so that comparison now tolerates one nanounit. The line-initial case has
a separate profile requiring a 5.5% to 6% shallow overlap, a following gap of at least 17% of line
height, compatible pitch, at least 99.87% and 97.9% fragment confidence, and at least 99.96% exact
combined recognition. The exact full comparison removes only `dev-plain-0141` and
`dev-plain-0969`, with no addition or stage change. Whole-eojeol OCR rises to
98.03%, functional context to 87.45%, exact sentence transcription to 70.60%, and first-popup
correctness to 81.45%. The quick tier removes only `dev-plain-0141` and otherwise
preserves its failure records. The audit has 193 active context cases, 12 resolved IDs, 35 active
boundary cases, and no missing decisions.

The sixth word-boundary batch reviews four repeated 3+3-syllable missed spaces. At the normal
CTC threshold each pair is one six-syllable word, while a 0.01 space probe produces exactly two
three-syllable crops separated by 28% to 35% of line height. The recovery additionally requires an
all-Hangul surface, at least 99.4% original confidence, compatible pitch, at least 99.3% confidence
for both parts, and exact concatenation back to the original text. Legitimate six-syllable controls
in the same lines remain unsplit. The full diagnostic comparison removes only
`dev-plain-0098`, `dev-plain-1297`, and `dev-plain-1617`
from context; the first two move downstream to primary-lemma failures and the third becomes fully
correct. The fourth case retains a separate 1+1 merge whose 0.005 CTC signal is insufficient for a
safe general rule. No unrelated ID changes. Whole-eojeol OCR rises to 98.06%, functional context
to 87.60%, exact sentence transcription to 70.75%, and first-popup correctness to 81.50%. The
quick tier moves only `dev-plain-0098` from context to primary lemma, raising quick OCR,
context, and exact transcription to 99.10%, 94.00%, and 74.50%. The audit has 190 active context
cases, 15 resolved IDs, 32 active boundary cases, and no missing decisions.

The seventh word-boundary batch reviews three active 2+2-syllable false splits. Two have a
candidate gap of 15% to 24% of line height while the neighboring boundaries are at least ten
percentage points wider. The accepted profile additionally requires pure Hangul two-syllable
fragments, at least 99.6% confidence for both, at least 95% compatible character pitch, and
99.6% exact recognition of their union. A provisional 26% cap also merged a legitimate space in
`dev-plain-0922`; that version was rejected, the final cap is 24%, and the legitimate spacing is
covered by a regression test. The exact full comparison removes only `dev-plain-0734` and
`dev-plain-1995`, with no addition or stage change. `dev-plain-1673` remains active because
it also has a separate bracket-attached boundary without sufficient evidence for recovery.
Functional context rises to 87.70%, exact sentence transcription to 70.80%, and first-popup
correctness to 81.60%; quick diagnostics remain byte-identical. The audit has 188 active context
cases, 17 resolved IDs, 30 active boundary cases, and no missing decisions.

The eighth word-boundary batch groups all 30 active boundary cases by their minimal expected-to-
actual token-length transformation. Two repeated 3+5-to-8 merges have independent CTC evidence:
a 0.02 space probe splits the high-confidence eight-syllable Hangul token into exactly three and
five syllables, the gap is 30% to 33% of line height, character pitch agrees within 3%, both parts
are at least 99.88% confident, and concatenating them exactly reproduces the original token. The
accepted profile removes only `dev-plain-0130` and `dev-plain-0499`, with no addition or stage
change. The 2+4-to-6 group remains unchanged because it mixes a structured-ASCII case, a
defensible compound-spacing interpretation, and a low-confidence proper noun. Whole-eojeol OCR,
functional context, exact sentence transcription, and first-popup correctness rise to 98.08%,
87.80%, 70.90%, and 81.70%. The quick tier removes only `dev-plain-0130`, raising quick OCR,
context, exact transcription, and popup correctness to 99.19%, 94.50%, 75.00%, and 90.00%.
The audit has 186 active context cases, 19 resolved IDs, 28 active boundary cases, and no missing
decisions.

The ninth word-boundary batch reviews both remaining 5-to-3+2 splits. One is a defensible spaced
auxiliary interpretation and remains unchanged. The genuine split has a distinct internal profile:
the fragments are at least 99.97% and 99.98% confident, their gap is 10% to 10.5% of line height,
the preceding boundary is at least 25%, the following boundary is a shallow overlap of at most
5.5%, character pitch agrees within 2%, and recognizing the union at 99.98% or better exactly
returns the concatenation. The exact full comparison removes only `dev-plain-1370`, with no
addition or stage change; `dev-plain-1353` remains active as the spacing control. Functional
context, exact sentence transcription, and first-popup correctness rise to 87.85%, 70.95%, and
81.75%. Quick diagnostics remain byte-identical. The audit has 185 active context cases, 20
resolved IDs, 27 active boundary cases, and no missing decisions.

The tenth word-boundary batch reviews the three repeated 3-to-1+2 false splits as distinct
geometries. The accepted line-initial profile requires fragment confidence of at least 99.92%
and 99.86%, a gap of 36% to 36.5% of line height, a following boundary of 61% to 62.5%, compatible
pitch, and exact union recognition at 99.975% or better. The accepted internal profile requires
fragment confidence of at least 99.99% and 99.93%, a gap of 6% to 6.5%, a preceding boundary of
at least 37%, a following boundary within 0.5% of touching, compatible pitch, and the same exact
union floor. Both profiles additionally reject the merge when recognizing the second fragment
with its following neighbor reaches 90% confidence; their verification crops normalize
subpixel coordinates before integer rounding. The third case remains unchanged because its
one-syllable fragment is only 92.30% confident, both neighboring boundaries overlap, and union
recognition is below the accepted floor. The exact full comparison removes only
`dev-plain-0155` and `dev-plain-1185`, with no new or changed failure record. The quick
comparison removes only `dev-plain-0155`. Full OCR, context, exact transcription, and popup
correctness rise to 98.09%, 87.95%, 71.05%, and 81.85%; quick OCR, context, exact transcription,
and popup correctness rise to 99.23%, 95.00%, 75.50%, and 90.50%. The audit has 183 active
context cases, 22 resolved IDs, 25 active boundary cases, and no missing decisions.

The eleventh word-boundary batch reviews the three repeated 6-to-3+3 false splits. All three
candidate unions exactly reproduce the expected six-syllable Hangul token. The line-initial
profile requires fragment confidence of at least 99.65% and 99.99%, a gap of 26% to 26.5% of
line height, a following boundary of 54% to 55%, pitch agreement within 4%, and exact union
recognition at 99.93% or better. The isolated internal profile requires fragment confidence of
at least 99.81% and 99.68%, a gap of 35% to 36.5%, a preceding boundary of at least 61%, a
following boundary of at least 44%, pitch agreement within 2%, and exact union recognition at
99.83% or better. Both profiles require pure Hangul and reject recovery when either available
adjacent union reaches 99.5% confidence. The exact full comparison removes only
`dev-plain-1272` and `dev-plain-1280`, with no new or changed failure record.
`dev-plain-0475` remains active because independent punctuation, transcription, and additional
split errors remain after its 3+3 component is recovered. Quick correctness and failure records
remain unchanged. Full OCR, context, exact transcription, and popup correctness rise to 98.10%,
88.05%, 71.10%, and 81.95%. The audit has 181 active context cases, 24 resolved IDs, 23 active
boundary cases, and no missing decisions.

The twelfth word-boundary batch reviews two false 5-to-3+2 Hangul splits plus one legitimate
spacing control. The narrow-gap profile requires fragment confidence of at least 99.87% and
99.95%, a gap of 5% to 5.5% of line height, preceding and following boundaries of at least 20%
and 25%, pitch agreement within 11%, and exact union recognition at 99.79% or better. The
isolated-wide profile requires fragment confidence of at least 99.81% and 99.94%, a gap of 36%
to 36.5%, both neighboring boundaries of at least 61%, pitch agreement within 4%, and exact
union recognition at 99.77% or better. Both require pure Hangul and reject recovery when an
available adjacent union reaches the profile-specific 99% or 99.5% ceiling. The 98.38%-confidence
spacing control remains separate. Quick correctness and failure records are unchanged. The exact
full comparison removes only `dev-plain-1129`; `dev-plain-0475` remains active because an
independent one-plus-two split and punctuation defect remain. Full OCR, context, and popup
correctness rise to 98.11%, 88.10%, and 82.00%. The audit has 180 active context cases, 25
resolved IDs, 22 active boundary cases, and no missing decisions.

The thirteenth word-boundary batch reviews two false 6-to-4+2 Hangul splits. Their raw geometry
supports two independent profiles rather than a broad split rule. The positive-gap profile requires
fragment confidence of at least 99.87% and 99.97%, a gap of 22.5% to 23% of line height,
preceding and following boundaries of at least 51% and 45%, pitch agreement within 2%, and exact
union recognition at 99.70% or better. The slight-overlap profile requires fragment confidence
of at least 99.89% and 96.06%, an overlap of 5% to 5.5%, preceding and following boundaries of
at least 36% and 41%, pitch agreement within 15%, and exact union recognition at 99.93% or
better. Both require pure Hangul, normalize subpixel coordinates, and reject recovery when either
available adjacent union reaches 98.5%. Quick correctness and failure records are unchanged. The
exact full comparison removes only `dev-plain-0586`; `dev-plain-0233` remains active because an
independent 3+1 split remains. Full OCR, context, exact transcription, and popup correctness rise
to 98.12%, 88.15%, 71.15%, and 82.05%. The audit has 179 active context cases, 26 resolved IDs,
21 active boundary cases, and no missing decisions.

The fourteenth word-boundary batch revisits the three repeated 6-to-2+4 merges with direct CTC
evidence. A 0.01 space probe returns exact two- and four-character parts across every measured
threshold. The pure-Hangul profile requires a 33% to 35% line-height gap, pitch agreement within
6%, whole-word confidence of at least 65%, both parts at least 84% confident, and one part at
least 99.98% confident. The structured identifier profile requires two Hangul characters followed
by four decimal digits, at least 99.85% whole-word confidence, a 31% to 35% gap, compatible pitch,
and part confidence of at least 99.98% and 99.90%. Both profiles require edge-complete crops and
exact part concatenation. The exact full comparison removes only `dev-plain-1571`,
`dev-plain-1873`, and `dev-plain-1937`, with no addition or stage change. Quick, stress,
language, target, analysis, promotion, and negative-pointer results remain unchanged. Full OCR,
context, exact transcription, and popup correctness rise to 98.14%, 88.30%, 71.30%, and 82.20%.
The audit has 176 active context cases, 29 resolved IDs, 18 active boundary cases, and no missing
decisions.

The fifteenth word-boundary batch reviews the remaining pure-Hangul 4-to-3+1 family as distinct
geometries. The accepted positive-gap profile requires fragment confidence of at least 99.98%
and 99.88%, a gap of 28% to 28.5% of line height, preceding and following boundaries of at least
45% and 56%, pitch agreement within 19%, exact union recognition at 99.96% or better, and no
adjacent union reaching 99%. The accepted shallow-overlap correction requires fragment confidence
of at least 99.85% and 91%, overlap of 5.5% to 6%, neighboring boundaries of at least 62% and
28%, pitch agreement within 42%, and union recognition at 99.95% or better. Its union must remain
four Hangul characters, preserve the first two and final characters, and differ from the fragment
concatenation at exactly one internal character; no adjacent union may reach 98.5%. The exact
full comparison removes only `dev-plain-0233` and `dev-plain-0859`, with no addition or stage
change. Quick, stress, language, target, analysis, promotion, and negative-pointer results remain
unchanged. Full OCR, context, exact transcription, and popup correctness rise to 98.15%, 88.40%,
71.40%, and 82.30%. The audit has 174 active context cases, 31 resolved IDs, 16 active boundary
cases, and no missing decisions. The repeated wide-gap case remains excluded because an adjacent
union reaches 99.89%; another exact union remains excluded because its confidence is only 98.03%.

The sixteenth word-boundary batch separates the remaining pure-Hangul 3-to-1+2 cases by geometry.
The accepted isolated-wide profile requires fragment confidence of at least 83.5% and 99.88%, a
gap of 36% to 36.5% of line height, preceding and following boundaries of at least 77% and 61%,
pitch agreement within 27%, exact union recognition at 99.98% or better, and no adjacent union
reaching 98%. The exact full comparison removes only `dev-plain-0475`, with no addition or stage
change. Quick, stress, language, target, analysis, promotion, and negative-pointer results remain
unchanged. Full OCR, context, and popup correctness rise to 98.16%, 88.45%, and 82.35%; exact
transcription remains 71.40%. The audit has 173 active context cases, 32 resolved IDs, 15 active
boundary cases, and no missing decisions. The overlapping 1+2 case remains excluded because its
one-character fragment is only 92.30% confident, both neighboring boundaries overlap, and exact
union recognition reaches only 99.7795%.

The seventeenth word-boundary batch separates two same-sentence pure-Hangul 4-to-3+1 splits.
The accepted isolated-wide profile requires fragment confidence of at least 99.97% and 99.91%, a
gap of 36% to 36.5% of line height, preceding and following boundaries of at least 51% and 56%,
pitch agreement within 13%, exact union recognition at 99.97% or better, and no adjacent union
reaching 99%. It recovers one exact eojeol in `dev-plain-1115`; the aggregate OCR percentage stays
98.16% after rounding, while the 16 px stratum rises from 98.66% to 98.70% and the ellipsis stratum
rises from 98.02% to 98.06%. Quick diagnostics are byte-identical, full stable-ID diagnostics are
unchanged, and target, context, analysis, stress, language, promotion, and negative-pointer
metrics do not regress. The other occurrence remains excluded because its exact union reaches
only 99.9402% and an adjacent union reaches 99.8913%; a separate punctuation omission also keeps
the sample in the context review.

The eighteenth word-boundary batch distinguishes a line-initial pure-Hangul 4-to-2+2 split from
a separate punctuation-attached 3-to-2+1 split in `dev-plain-1673`. The accepted 2+2 profile
requires fragment confidence of at least 99.98% and 99.99%, a gap of 25.5% to 26% of line height,
a following boundary of at least 46%, pitch agreement within 4%, exact union recognition at
99.99% or better, and no following union reaching 90%. It recovers one exact eojeol; aggregate
OCR remains 98.16% after rounding, while the 12 px stratum rises from 97.79% to 97.83%, terminal
punctuation from 98.23% to 98.26%, and Malgun Gothic from 98.65% to 98.67%. Quick diagnostics are
byte-identical, full stable-ID diagnostics are unchanged, and target, context, analysis, stress,
language, promotion, and negative-pointer metrics do not regress. The 2+1 split remains excluded
because its punctuation-preserving union reaches only 99.5782%, and the sample remains active in
the context review.

The nineteenth word-boundary batch reviews the two remaining seven-syllable pure-Hangul detector
merges as distinct 5+2 and 4+3 profiles. A word-local 0.01 CTC-space probe must return exactly two
edge-complete crops, their gap must be 32% to 34% of line height, character pitch must agree within
3%, both parts must have the reviewed lengths and remain pure Hangul, and concatenating them must
exactly reproduce the original recognition. The 5+2 profile requires one part at 99.97% and the
other at 97.9%; the 4+3 profile requires both parts at 99.99%. The original seven-syllable word
must be at least 96% confident. The exact full comparison removes only `dev-plain-0243` and
`dev-plain-1894`, with no addition or stage change. Full OCR, context, exact transcription, and
popup correctness rise to 98.18%, 88.55%, 71.50%, and 82.45%; target, analysis, component,
dictionary, alternative, stress, language, promotion, and negative-pointer results do not regress.
The 16 px and 24 px OCR strata rise from 98.70% and 98.28% to 98.76% and 98.35%. Quick aggregate
metrics are unchanged and its privacy-safe diagnostics remain byte-identical. The audit has 171
active context cases, 34 resolved IDs, 13 active boundary cases, and no missing decisions.

The twentieth word-boundary batch reviews the three remaining shorter detector merges. Only the
five-syllable 3+2 case has stable word-local evidence: a 0.01 CTC-space probe returns exactly two
edge-complete pure-Hangul crops, their gap is 33% to 34% of line height, pitch agrees within 10%,
both parts are at least 99.92% confident, and their concatenation exactly reproduces an original
word at least 99.9% confident. The exact full comparison removes only `dev-plain-0341`, with no
addition or stage change. Full OCR, context, exact transcription, and popup correctness rise to
98.19%, 88.60%, 71.55%, and 82.50%; the 20 px OCR stratum rises from 99.15% to 99.22%. Quick
aggregate metrics are unchanged and its privacy-safe diagnostics remain byte-identical. Target,
analysis, component, dictionary, alternative, stress, language, promotion, and negative-pointer
results do not regress. The audit has 170 active context cases, 35 resolved IDs, 12 active boundary
cases, and no missing decisions. `dev-plain-0450` remains unchanged because its 1+1 candidate
appears only below the 0.01 CTC threshold, which is unsafe for ordinary two-syllable words;
`dev-plain-0977` remains unchanged because its 2+1 candidate appears only at 0.001 and the crops
overlap instead of exposing a whitespace boundary.

The twenty-first word-boundary batch reviews the remaining 4+1, 1+3, and 1+4 pure-Hangul false
splits as separate profiles. The accepted overlapping 4+1 profile requires fragment confidence of
at least 99.96% and 91.4%, a 4.5% to 5% overlap, preceding and following boundaries of at least
28% and 37% of line height, pitch agreement within 20%, exact union recognition at 99.97%, and no
adjacent union reaching 98%. The accepted isolated 1+4 profile requires fragment confidence of at
least 99.84% and 99.97%, a gap of 35% to 35.5%, surrounding boundaries of at least 54% and 67%,
pitch agreement within 10%, exact union recognition at 99.96%, and no adjacent union reaching
99.8%. Subpixel verification coordinates are normalized for both profiles. The exact full
comparison removes only `dev-plain-0219` and `dev-plain-1437`, with no addition or stage change.
Full OCR, context, exact transcription, and popup correctness rise to 98.20%, 88.70%, 71.65%, and
82.60%; the 16 px and 20 px OCR strata rise from 98.76% and 99.22% to 98.80% and 99.26%. Quick
aggregate metrics are unchanged and its privacy-safe diagnostics remain byte-identical. Target,
analysis, component, dictionary, alternative, stress, language, promotion, negative-pointer, and
latency gates do not regress. The audit has 168 active context cases, 37 resolved IDs, ten active
boundary cases, and no missing decisions. The line-initial 1+3 case `dev-plain-1435` remains
unchanged because its union is only 86.2% confident while the following union reaches 99.4%.

The twenty-second word-boundary batch reviews a terminal structured 3+1 split and a complex
multi-defect layout separately. The accepted terminal profile requires exactly two ASCII decimal
digits and one Hangul syllable in the first fragment followed by one Hangul syllable, fragment
confidence of at least 99.61% and 99.96%, a gap of 35% to 35.5% of line height, a preceding
boundary of at least 62%, pitch agreement within 12%, exact union recognition at 99.92%, and a
preceding adjacent union below 99%. Subpixel verification coordinates are normalized. The exact
full comparison removes only `dev-plain-1277`, with no addition or stage change. Aggregate OCR
remains 98.20%, while context, exact transcription, and popup correctness rise to 88.75%, 71.70%,
and 82.65%. The 20 px stratum rises to 99.29% OCR, 92.40% context, 81.20% exact transcription,
and 87.60% popup correctness. Quick diagnostics remain byte-identical; target, analysis,
component, dictionary, alternative, stress, language, promotion, negative-pointer, and latency
gates do not regress. The audit has 167 active context cases, 38 resolved IDs, nine active
boundary cases, and no missing decisions. `dev-plain-1740` remains unchanged because its complete
multi-fragment union reaches only 97.39%, its punctuation-changing union reaches only 90.13%, and
an exact 99.94% sub-union still does not reproduce the independent word.

The remaining uncharacterized boundary cases do not support another safe recovery. The documented
spacing control `dev-plain-1353` remains separate; `dev-plain-1609` has a 92.30%-confidence
one-syllable fragment, overlapping neighboring boundaries, and only 99.7795% exact union
recognition; and `dev-plain-1838` reaches only 98.8656% for its intended 3+1 union. All nine active
reviewed boundary IDs are now characterized, and no global OCR split threshold is changed.

The next downstream analyzer batch reviews the eight remaining primary-lemma cases categorized as
Kiwi errors. A two-syllable proper-noun leader may yield to an already-present one-syllable
dictionary-backed noun plus exact one-syllable particle only within 3.2 score points and only when
the particle is centrally known or independently present in KRDict. A complete multi-syllable
dictionary-backed inflected predicate is not replaced by a richer split containing only
non-auxiliary verbs; one-syllable bases and main-plus-helping-verb analyses preserve established
behavior. A broader initial predicate guard was rejected because it moved one main failure stage
and reduced held-out multi-lexical accuracy to 87.50%. The accepted rules remove only
`dev-plain-0375`, `dev-plain-0663`, and `dev-plain-1472`, with no addition or stage change.
Component accuracy rises to 92.45%, exact KRDict fidelity to 94.85%, and popup correctness to
82.80%. Quick diagnostics remain byte-identical, stress is unchanged, held-out language remains
92.00% overall / 96.00% auxiliary / 88.00% multi-lexical, and every upstream, promotion,
negative-pointer, and latency gate is preserved.

The following component-role batch narrows two context-sensitive role decisions. A lexical
`hada` candidate is not rewritten as a helping verb solely because it follows a plain connective
ending or the demonstrative-adverb pattern; explicit obligative context remains eligible for the
established helping-verb promotion. When punctuation occurs
immediately after a target and the next non-punctuation token is nominal, an already-present
same-lemma determiner candidate may lead within 4.0 score points. The exact comparison removes
only `dev-plain-1006`, `dev-plain-1496`, `dev-plain-1528`, and `dev-plain-1560`, with no addition
or stage change. Component accuracy rises to 92.65%, exact KRDict fidelity to 95.00%, popup
correctness to 83.00%, and alternative recovery to 94.20%. Quick diagnostics remain
byte-identical; stress and held-out language are unchanged.

The next component-role batch extends the existing dictionary-backed proper-noun-to-ordinary-noun
score window from 2.5 to 3.2 points and reapplies that preference after paired-wrapper context only
when the current leader is still a proper noun. A broader post-wrapper dictionary pass was rejected
because it changed the correct pronoun in `dev-plain-1982` to a determiner. The restricted comparison
removes only `dev-plain-0662` and `dev-plain-1625`, with no addition or stage change. Component
accuracy rises to 92.70% and popup correctness to 83.10%; exact KRDict fidelity remains 95.00% and
alternative recovery remains 94.20%. Quick diagnostics are byte-identical, while stress, held-out
language, upstream, promotion, negative-pointer, and latency gates do not regress.

The following component-role batch preserves an already-present helping-verb reading after an
explicit `-아야만`, `-어야만`, or `-여야만` context instead of letting an isolated action-verb
reading override it. A wrapper-context noun may yield to a same-surface, same-lemma adverb within
6.0 score points only when the adverb is dictionary-backed and both unwrapped context and isolated
analysis independently prefer it. The exact comparison removes only `dev-plain-1399`,
`dev-plain-1512`, and `dev-plain-1607`, with no addition or stage change. Component accuracy rises
to 92.85%, exact KRDict fidelity to 95.15%, and popup correctness to 83.25%; alternative recovery
remains 94.20%. Quick diagnostics are byte-identical, while stress, held-out language, upstream,
promotion, and negative-pointer results are unchanged.

Compared with the earlier full report, the accepted cleanups resolve 54 context IDs without
introducing a new context failure. The preceding cleanup permits a one-pixel overlap of at most 7.5%
of a small line only under the existing exact combined-recognition duplicate profile. It resolves
two full-tier context cases while leaving the quick diagnostics byte-identical. The same profile
now accepts an ASCII digit as the leading artifact only when recognizing the union at 99% or
better omits it and exactly reproduces the following Hangul word. This resolves three additional
full-tier cases without a new failure or quick-tier change.

The first-popup reviewer now has the same separately scoped full-tier mode, repeated stable-ID
batch inspection, structure-only output, batch categorical recording, and matching-only migration
for an oracle-corrected corpus. Full reports use the `first_popup_analysis_full` kind and cannot
be mixed with quick reports. The v4.12 history contains 167 decisions: 73 Kiwi-analysis errors,
32 annotation-convention differences, 34 equivalent learner interpretations, 22 corpus-oracle
defects, and six genuinely ambiguous cases. Its SHA-256 is
`76d089614630f196eb4c003382e2879756bf349fda2b5aaa8eb4e7cbdbb9aed5`.
The v4.13 history then contains 112 active decisions; its SHA-256 is
`a6af7603ec91e6a69e080e05866eb1359bd06974e70d0b16edbd91da62a2fdbc`.
The current v4.15 report contains 120 decisions: 28 Kiwi-analysis errors, 42 annotation-
convention differences, 35 equivalent learner interpretations, eight corpus-oracle defects,
and seven genuinely ambiguous cases. The current evidence leaves 108 reviewed active cases and 12 resolved IDs; two newly
downstream primary-lemma cases remain missing, with no stale decision. Its SHA-256 is
`d6db4974d39f20806866f321aa767bfe927d100b78ed25a90d62b089c66ed8b6`.

The second full-tier review batch exposed repeated noun-plus-`화` derivations that Kiwi split into
a noun, derivational noun suffix, and action-verb suffix even though KRDict contains the complete
verb. The accepted analyzer joins only `noun + 화 + 하/되` and only when the exact complete verb is
dictionary-backed. It resolves four main primary-lemma failures, including both reviewed examples,
without adding a failure. Quick diagnostics remain byte-identical, and stress and held-out language
results are unchanged.

The third full-tier batch supports four additional bounded analyzer corrections. Exact
dictionary-backed `adverb + 하/XSV|XSA` derivations form one learner component and retain a
contextual helping-verb role after a connective. Exact standalone object particles `을` and `를`
can override a false noun analysis, while ambiguous particle/noun surfaces remain unchanged. An
internal `화/XSN` extends a preceding noun only when the complete combined noun is dictionary-backed
and no copula follows. Finally, a zero-length Kiwi insertion is ignored only when it duplicates the
preceding component surface; nonduplicate reported-speech ellipsis remains represented. These
changes recover seven net main popups, raise component accuracy by 0.40 points and KRDict fidelity
by 0.45 points, and improve the held-out auxiliary tier by two cases. Quick diagnostics remain
byte-identical, multi-lexical accuracy remains 86.50%, and no upstream or negative-pointer result
changes. Broader internal-suffix attachment and unconditional zero-length suppression were rejected
because they regressed held-out multi-lexical analysis or changed a correct failure stage.

The fourth full-tier batch reviews 20 component-role cases as nine Kiwi-analysis errors, seven
annotation-convention differences, one corpus-oracle defect, and three genuinely ambiguous
fragments. Accepted corrections treat suffixed reported-speech connectives and nominal `-라도` as
non-auxiliary context, preserve the exact `-기도 하다` helping-verb construction after isolated-
eojeol reranking, and let the existing `-게 되다` rule recognize contracted written forms through
the already-matched `되다` lemma. Paired punctuation may supply a missing role-only candidate only
when unwrapped context preserves the complete lemma and component boundaries and the wrapped
analysis exposed no alternative. Dictionary ordering may promote a supported adverb interpretation
within the existing score margin, but cannot demote an existing adverb to a nominal role. The final
cross-validation removes 12 main failure IDs, adds none, and changes no failure stage. It preserves
byte-identical quick diagnostics, all stress results, held-out multi-lexical accuracy, and every
upstream and negative-pointer metric while recovering six held-out auxiliary cases. A symmetric
adverb/noun reranker was rejected after introducing two main component-role regressions.

The fifth full-tier batch classifies 20 more component-role cases as 12 Kiwi-analysis errors,
four annotation-convention differences, three equivalent learner interpretations, and one
corpus-oracle defect. Paired punctuation no longer hides the existing `-게 되다` cue. When a leading
single-component helping-verb candidate has no auxiliary KRDict sense or connective context, only
its immediate same-lemma, same-boundary, dictionary-backed lexical alternative may replace it,
within a 7.1-point score margin. Dictionary ordering cannot demote an already-leading determiner or
adverb, and a numeric-plus-helping decomposition cannot displace a complete dictionary-backed
verb. The accepted rerun removes five main component-role failures and adds none, raising component
accuracy from 89.95% to 90.20%, KRDict fidelity from 93.35% to 93.50%, and full popup correctness
from 78.40% to 78.65%. Quick diagnostics remain byte-identical; alternative recovery, stress,
held-out language, upstream, promotion, and negative-pointer results are unchanged.

The sixth full-tier batch classifies 20 more component-role cases as 14 Kiwi-analysis errors,
three annotation-convention differences, two corpus-oracle defects, and one equivalent learner
interpretation. Accepted corrections give dictionary-backed dependent-noun readings a separately
bounded 2.7-point margin without ever demoting an already-leading dependent noun; use KRDict's
default predicate role only for identical single-component action/descriptive alternatives within
one point; and recognize descriptive `있다` after a locative particle within a separately bounded
seven-point margin. The established auxiliary-context rule now covers contracted
`-아야/-어야/-여야만 하다`. When all wrapped candidates are learner-identical, paired-punctuation
analysis may synthesize the same-lemma, same-boundary verb-role interpretation supported by the
unwrapped context, and that richer evidence is not undone by isolated-eojeol reranking. The
accepted v7 cross-validation resolves seven component-role failures and one grammar-role failure,
adds no failure or stage change, raises full popup correctness from 78.65% to 79.05%, component
accuracy from 90.20% to 90.55%, KRDict fidelity from 93.50% to 93.65%, and alternative recovery
from 93.10% to 93.15%. Quick diagnostics remain byte-identical, stress is unchanged, and held-out
language improves from 90.25% to 90.50% overall and from 86.50% to 87.00% multi-lexical while
auxiliary accuracy remains 94.00%. A global post-context dictionary-role reapplication was
rejected after adding five main role failures; the retained one-way rule adds none.

The seventh batch completes the remaining 11 component-role reviews: six Kiwi-analysis errors,
three annotation-convention differences, and two corpus-oracle defects. Accepted corrections use
an ordinary-noun alternative when KRDict has no dependent-noun entry; recognize ordinary `식` in
copular and instrumental contexts and an attached compound's terminal noun when dictionary order
supports it; and expose an exclusively auxiliary predicate entry as a helping verb. Isolated-
eojeol evidence may still recover a complete multi-component analysis in ordinary sentence
context, but role-only reranking is limited to punctuation or fragment boundaries. The independent
GSD oracle builder now honors a single-token `ADV` role even when its broad XPOS tag is nominal;
that oracle correction applies on the next development-corpus rebuild rather than mutating the
locked v4.12 samples. The accepted v8 comparison removes exactly six main component-role failures
and adds none. Quick diagnostics remain byte-identical and stress is unchanged; main component
accuracy rises to 90.70% and popup correctness to 79.35%, while held-out language rises to 91.25%
overall and 95.50% auxiliary with multi-lexical unchanged at 87.00%.

The eighth batch completes review of the 24 remaining primary-lemma cases: 13 Kiwi-analysis
errors, nine equivalent learner interpretations, and two annotation-convention differences. The
reviewer can record one category for a validated batch of stable IDs in one model initialization;
the persisted schema remains ID-and-category only. Accepted analyzer rules recover a complete
dictionary-backed inflected predicate without losing tense or other grammar features; preserve a
complete lexical adverb over nominal splitting; require an actual connective before isolated
evidence promotes an auxiliary decomposition; and recognize close adnominal predicates before a
dependent noun. Boundary-scoped isolated predicate evidence excludes single-syllable surfaces and
determiners, while paired-wrapper synthesis requires repeated same-lemma nominal readings. The
independently annotated approximation and extent particle suffixes now recover dictionary-backed
noun stems and retain a dependent-noun role when KRDict supplies one. Two reviewed Kiwi errors—a
bare contracted copula with an inconsistent fallback oracle role and a semantic verb homograph—
remain unchanged because current evidence does not support a safe general runtime rule. The
accepted primary-lemma v2 comparison removes 15 primary-lemma failures, one component-role
failure, and one component-count failure, adds none, and leaves quick diagnostics byte-identical.
Main component accuracy rises to 91.60%, KRDict fidelity to 94.50%, popup correctness to 80.20%,
and alternative recovery to 93.40%. Stress popup correctness is unchanged; held-out language
rises to 91.50% overall and 96.00% auxiliary with multi-lexical unchanged at 87.00%.

The ninth batch classifies all 12 remaining component-surface cases as corpus-oracle defects.
Each pinned KAIST token uses the `pad` demonstrative-adjective tag, which the independent builder
had omitted from descriptive-predicate component mapping; the fallback therefore represented the
whole conjugated form instead of the annotated lexical stem. The v4.13 rebuild maps `pad`
consistently in component roles, dictionary ordering, component lemmas, and primary lemmas. It
removes all 12 main and two stress component-surface failures. An earlier accepted GSD correction
also takes effect in this rebuild: single-token `ADV` records now use their published adverb role,
resolving two main role failures while exposing six main and two stress role disagreements. The
net main comparison removes eight analysis failures and adds no component-surface failure. Quick
diagnostics remain byte-identical; main component accuracy rises to 92.00%, popup correctness to
80.60%, and alternative recovery to 93.90%. Stress popup correctness and held-out language are
unchanged. A matching-only review migration copies only current IDs with unchanged failure stages,
rejects stage changes, and leaves new cases visibly missing rather than discarding the audit.

The tenth batch reviews the six component-role cases exposed by the GSD `ADV` correction. Five
are annotation-convention differences: their published UPOS and `advmod` dependency describe
sentence function, while XPOS, Kiwi, and KRDict consistently identify a lexical noun. No runtime
promotion is supported for those learner-equivalent noun readings. The remaining record is a
corpus-oracle defect: a nominal list item is marked `ADV` but has a `conj` dependency and noun
XPOS. Oracle role refinement now requires `advmod` before a single nominal XPOS token inherits
the adverb role. The v4.14 comparison removes exactly that main component-role failure and adds
none. Quick diagnostics remain byte-identical, stress and held-out language are unchanged, and
main component accuracy rises to 92.10%, popup correctness to 80.65%, and alternative recovery to
94.00%.

The eleventh batch completes the six component-count reviews as two Kiwi-analysis errors, two
annotation-convention differences, and two equivalent learner interpretations. Isolated-eojeol
evidence can no longer promote a decomposition that repeats an identical lexical component. A
separately bounded 3.25-point rule treats a final one-syllable adverb component as a verb ending
only when a same-surface, same-lemma alternative preserves the complete predicate prefix and is
otherwise a dictionary-backed inflected predicate. The accepted comparison resolves exactly two
main component-count failures, adds no failure or stage change, and leaves quick diagnostics
byte-identical. Main component accuracy rises to 92.20%, KRDict fidelity to 94.70%, and popup
correctness to 80.75%; alternative recovery, stress, held-out language, upstream, promotion, and
negative-pointer results are unchanged. The aggregate and diagnostic report SHA-256 values are
`3da04aebe952ea9621e7c3ec828bd6b7a849032b5bc55067d73585a90c670e7d` and
`fcf608311de92707c687e352311f9d74373ce8e06f19e6d4ad56846191f64d22`.

The twelfth batch completes the seven grammar-role reviews as four annotation-convention
differences, two Kiwi-analysis errors, and one corpus-oracle defect. Four KAIST reported-speech
`jcr` tags are learner-equivalent connective endings and do not support a runtime particle
promotion. The oracle now maps an ADP token with particle XPOS to its first annotated particle
morpheme and orders particle homographs first. Wrapper-free sentence context can recover a
dictionary-backed standalone particle or a copular adnominal before a dependent noun under
bounded structural checks. The v4.15 comparison resolves two main grammar-role failures and adds
none: component accuracy rises to 92.35%, KRDict fidelity to 94.75%, popup correctness to 80.85%,
and alternative recovery to 94.10%. Quick accuracy remains unchanged, false promotions stay zero,
and stress, held-out language, upstream, and negative-pointer metrics do not regress. The corrected
particle case remains genuinely ambiguous after OCR omits its adjacent percent sign. The v4.15
corpus lock SHA-256 is
`5c57bdeb06e792960ec8869b0c3914a50170a911f73f1873b25185c011592ba8`;
the aggregate and diagnostic report SHA-256 values are
`f1a3e25371ae46a3b40cff4fc1bbd909e1c1a1a17a415a9277accd5f67662685` and
`0a46fd97070dda3606d9311b39a31b8eb651ab4d223eff5bef5cb600d7662484`.

Geometry-only review of the nine punctuation activations found three single-Hangul OCR boxes
whose width was 1.53 to 1.99 times their height after trailing punctuation was swallowed. Their
valid target points remained in the left interior while punctuation probes occupied the rightmost
21.7% to 23.4% of the abnormally wide box. Hit testing now increases only the right inset from
20% to 25% for a one-glyph Hangul eojeol at least 1.5 times as wide as it is tall. The other box
edges, normal glyphs, and every multi-glyph eojeol retain the existing contract. Full diagnostics
remove exactly those three punctuation-only IDs, add no ID or failure-stage change, and reduce
punctuation activation from nine of 1,582 (0.57%) to six (0.38%). Target selection remains 97.10%,
quick target selection remains 99.00%, and all popup, context, OCR, stress, language, promotion,
and quick negative results are unchanged. The aggregate/per-category negative gate now passes.
The accepted full aggregate and diagnostic SHA-256 values are
`ec0f004dc6c68cca531bbf364d7ef388d848f25264c18bb57fb653bc51aaf5e7` and
`41194b9c0e2b607293c2fa5f0b9394c6091a323766b96cec8aa6192f1f186e85`;
the quick values are
`0cc9b87444d03d1ac217bfa0afafe488328fcabb9a173f1c711eb2f9c21e927f` and
`147ca29fc83a6aeaf6f4b04b1bb69eb9cd23926f27ffb9ca7e9e4d194a867c60`.

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

The separately scoped v4.12 quick reviews are complete and contain no corpus text or pixels. The
quick popup review retains 20 decisions: seven Kiwi-analysis errors, five annotation-convention
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

The held-out multi-lexical review contains 26 stable-ID categorical decisions: three Kiwi-analysis
errors, five annotation-convention differences, 17 equivalent learner interpretations, and one
genuinely ambiguous case. It contains no sentence, expected, analyzed, dictionary, or pixel data.
The accepted analyzer guard prevents a lower-ranked, more fragmented nominal analysis from
displacing an already complete dictionary-backed multi-component leader. It resolves exactly two
reviewed Kiwi errors, leaves 24 active cases with no missing or stale decisions, and raises
multi-lexical accuracy from 87.00% to the required 88.00%. The full main diagnostics and quick
diagnostics remain byte-identical to their accepted baselines, stress and auxiliary results are
unchanged, and overall held-out language rises from 91.50% to 92.00%. The language-review report
SHA-256 is `2534d28a3bf9650f8a61120a6577e7e3397fc5a3e87f1c8b8544772d38913765`.

The complete v4.15 development run now shows that aggregate functional context clears 88%, while
main popup correctness and required render strata still block release evidence. Primary-lemma, component-surface,
component-role, component-count, grammar-role, multi-lexical, and negative-activation review is
complete. Two line/sentence reconstruction cases remain active because their intended punctuation
or English text cannot be inferred from reliable runtime evidence; no speculative rule is
implemented. All nine active reviewed missed-or-merged OCR word-boundary cases are now
characterized, and the remaining evidence is too weak, punctuation-dependent, ambiguous, or
complex for another safe general rule. Five reviewed primary-lemma Kiwi errors likewise lack a
safe complete candidate or bounded general promotion. Nine of the 20 reviewed component-role
Kiwi errors are now resolved without a regression; the next development target is the remaining
11 cases whose target and context are already correct.
Thresholds are not frozen, and neither the untouched release
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
- preserve the passing quick, held-out multi-lexical, and negative-activation gates while
  improving full first-popup correctness, functional context, and required render strata;
- run the opt-in foreground benchmark with five warmups plus 500 fixed attempts, meeting
  correctness and latency targets with zero safety violations;
- complete clean-VM tests on multi-monitor mixed-DPI systems and packaged Windows 10;
- replace the generated tray glyph with reviewed project artwork if desired;

The local asset release candidate is `2026.08.1`, SHA-256
`e623c7b55f2c236ca107baa60f9f0c63c5c5a0ecf8604575047e934fc9b7b8ee`. It is ignored
runtime output, not a published release. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement scope and remaining limitations. The runtime deliberately does not substitute
unverified downloads or cloud services when assets are absent.
