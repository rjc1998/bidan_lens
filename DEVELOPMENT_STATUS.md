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
`local-data\evaluations\bidan-lens-eval-ud218-v4.16\dev`; the untouched v4.2 release corpus
remains under `local-data\evaluations\bidan-lens-eval-ud218-v4.2\release` and has not been
evaluated. Earlier roots remain preserved. The v4.16 rebuild contains 2,000 main, 250 stress,
400 held-out language, and 200 quick cases, is hash-locked, and passes corpus validation. It uses
the current corpus builder and
the `viewport-v3` renderer policy. The intermediate v4.10 card-anchoring experiment is preserved
but rejected because it changed more already-visible geometry than the viewport defect required.

The accepted v4.16 200-case quick tier records 99.37% whole-eojeol OCR, 100.00% target
selection, 96.50% functional context, 76.50% exact sentence transcription, 95.00% component
accuracy, 95.50% exact KRDict fidelity, and 92.50% fully correct first popups, with 216.73 ms
median / 332.93 ms p95 automated latency. Alternative-candidate recovery is 97.50% and false
promotions remain zero. Its remaining failures are eight analysis cases (four primary lemmas and
four component roles) and seven context cases; no target failures remain. Aggregate and per-category
activation are 0.00%, so the quick popup floor and strict negative gate pass.

The aggregate report SHA-256 is
`4a2b6bfff3913b76f73ba4c56e8fa82e08e8fe48e5f59168085824846e24fbe9`; the
privacy-safe diagnostic SHA-256 is
`f6dbca9d8b291ff7190402d8c8808d19acfb50a05bb50c9cf0183423706549a6`.

The complete v4.16 development evaluation has now run against the current OCR and analyzer
cleanup. Its 2,000 main cases record 98.63% whole-eojeol OCR, 99.90% target selection, 91.75%
functional context, 73.40% exact sentence transcription, 93.75% component accuracy, 95.75% exact
KRDict fidelity, 86.85% fully correct first popups, and 97.00% alternative recovery. False
promotions remain zero and the accepted follow-up is 245.07 ms median /
378.22 ms p95. Privacy-safe diagnostics contain 2 target, 163 context, and 98 analysis failures;
the analysis stages are 43 primary lemmas, 45 component roles, four component counts, and six
grammar roles. No component-surface failures remain, and no stable ID has a negative
activation.

The nonblocking 250-case stress tier records 94.36% OCR, 96.00% target selection, 68.40%
functional context, 93.60% component accuracy, and 64.40% fully correct first popups. The 400-case
held-out language tier records 92.00% overall, 96.00% auxiliary, 88.00% multi-lexical, and 100%
direct KRDict conformance. Aggregate main negative activation is 0.00%; blank, English, near-miss,
whitespace, and punctuation probes all remain at zero. The correction, dictionary-conformance,
latency, and aggregate/per-category negative-activation gates pass, but the primary and
exceptional floors fail.

The current privacy-safe diagnostic SHA-256 is
`cbe375fad893ccf1128aed04a1f512e285e9c8d6f1f9ab7fc123ba354064105d`.

The context reviewer now has a separately scoped full-tier mode so quick and 2,000-case decision
reports cannot be mixed. Full cases can be inspected in a selected batch with one OCR model
initialization and categorized one stable ID at a time, writing each categorical decision
immediately. The v4.15 history contains 208 decisions: 90 non-target OCR transcription errors, 68
punctuation or structured-text cases, 43 missed or merged OCR word boundaries, and seven incorrect
line/sentence reconstructions. The added transcription category
covers substitutions or omissions outside the correct target when line reconstruction and target
geometry are otherwise intact. The full report contains only its corpus ID, review scope, stable
IDs, categorical decisions, and counts. Matching-only carry-forward copies reviewed current IDs
without weakening the strict mode and leaves every new ID explicitly missing. The v4.16 migration
retained all 170 active prior decisions and exposed only `dev-plain-1755`. Local review classified
its two inserted non-target spaces as a missed or merged OCR word boundary. The fail-closed v4.16
audit now covers all 163 active cases with no missing decision. It preserves eight resolved IDs:
`dev-plain-0001`, `dev-plain-0257`, `dev-plain-0482`, `dev-plain-0801`, `dev-plain-1281`,
`dev-plain-1601`, `dev-plain-1755`, and `dev-plain-1889`. The retained 171 decisions comprise 90 non-target
transcription errors, 68
punctuation or structured-text cases, 11 word-boundary cases, and two line/sentence
reconstructions. The decision report SHA-256 is
`0505d9d0cc5f6d0a4ef79e85c92c6f86084f355e43eb940c9fb1a070868148c7`.

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
gates do not regress. The audit has 168 active context cases, 38 resolved IDs, nine active
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

The next component-role batch widens the dictionary-preferred predicate-role window from 1.0 to
6.1 points only when an action-verb leader yields to an otherwise identical descriptive-verb
candidate and KRDict's first entry is an adjective. The reverse direction retains the 1.0-point
limit. Isolated analysis cannot restore a higher-scored raw candidate after the contextual
dictionary preference has deliberately promoted a lower-scored leader. The exact comparison
removes only `dev-plain-0996`, with no addition or stage change. Component accuracy rises to
92.90%, exact KRDict fidelity to 95.20%, and popup correctness to 83.30%; alternative recovery
remains 94.20%. Quick diagnostics are byte-identical, while stress, held-out language, upstream,
promotion, and negative-pointer results are unchanged.

The following component-role batch lets an existing same-surface descriptive `있다` candidate
lead by as much as 11.0 score points at a one-sided punctuation or fragment boundary only when
isolated analysis independently prefers the same dictionary-backed descriptive reading. Paired
wrappers and other verb-role pairs retain the 2.0-point isolated limit. The exact comparison removes
only `dev-plain-0280`, `dev-plain-0324`, and `dev-plain-1103`, with no addition or stage
change. Quick diagnostics are byte-identical and quick quality metrics are unchanged. Main
component accuracy rises to 93.00%, exact KRDict fidelity to 95.35%, and popup correctness to
83.45%; alternative recovery remains 94.20%. Stress, held-out language, upstream, promotion, and
negative-pointer results are unchanged.

The next component-role batch promotes an existing dictionary-backed dependent-noun candidate
only when a single-space adnominal clause governs a target that contains both a copula and final
ending. The promotion is capped at 4.3 score points; ordinary non-copular nouns remain unchanged.
The exact comparison removes only `dev-plain-1229`, with no addition or stage change. Quick
diagnostics are byte-identical and quick quality metrics are unchanged. Main component accuracy
rises to 93.05% and popup correctness to 83.50%; exact KRDict fidelity remains 95.35% and
alternative recovery remains 94.20%. Stress, held-out language, upstream, promotion, and
negative-pointer results are unchanged.

The following component-role batch promotes an existing same-surface noun candidate within 4.5
score points only when KRDict orders the noun first and local analysis of the target plus an
adjacent inflected `있다` eojeol independently prefers that candidate. Immediate punctuation may
separate the target from the required single space. Candidate shape and dictionary checks run
before the extra local analysis. The exact comparison removes only `dev-plain-0539`, with no
addition or stage change. Quick diagnostics are byte-identical and quick quality metrics are
unchanged. Main component accuracy rises to 93.10%, exact KRDict fidelity to 95.40%, and popup
correctness to 83.55%; alternative recovery remains 94.20%. Stress, held-out language, upstream,
promotion, and negative-pointer results are unchanged.

The next primary-lemma batch promotes an existing contextual alternative within 3.0 score points
only at a punctuation or fragment boundary and only when its exact candidate signature leads
isolated-eojeol analysis, its lemma differs from the current leader, every component is
dictionary-backed, and no word part is unrepresented. Eligible alternatives are limited to a
longer complete inflected predicate or a single whole-surface number replacing a multi-component
split. The exact comparison removes only `dev-plain-0098`, `dev-plain-1137`, and
`dev-plain-1297`, with no addition or stage change. The quick tier removes only
`dev-plain-0098`; component accuracy and KRDict fidelity rise to 94.50% and 95.50%, and popup
correctness rises to 91.00%. Full component accuracy rises to 93.25%, KRDict fidelity to 95.55%,
and popup correctness to 83.70%; alternative recovery remains 94.20%. Stress, held-out language,
upstream, promotion, and negative-pointer results are unchanged.

Privacy-safe clustering of the 58 remaining main target failures separates 32 wrong-text hits in
matching geometry from 26 no-hit cases, most caused by merged eojeols. The accepted target-boundary
batch probes only an eight-character, at least 99.8%-confidence pure-Hangul eojeol at a 0.04 CTC
space threshold. It splits only when two edge-complete four-character readings have at least
99.96% confidence, 28%-29% of line-height separation, at least 99% pitch agreement, and exactly
recombine to the original. The exact full comparison removes only `dev-plain-0819`, with no
addition or stage change. Quick diagnostics remain byte-identical and every non-latency quick
quality metric remains unchanged; stress, held-out language, promotion, and negative-pointer
results are unchanged.

The second target-boundary batch probes only an 88%-89%-confidence sequence of four Hangul
characters followed by a terminal ellipsis. At the 0.04 CTC-space threshold, it requires two
edge-complete readings of two Hangul and two Hangul plus ellipsis, both at least 99.97% confident,
with 34%-35% of line-height separation, at least 94% pitch agreement, and exact recombination.
The exact full comparison removes only `dev-plain-1698`, with no addition or stage change. Quick
diagnostics remain byte-identical and every non-latency quick quality metric remains unchanged;
stress, held-out language, promotion, and negative-pointer results are unchanged.

Privacy-safe detector-relative probing then groups the 30 matching-geometry wrong-text cases by
raw/core length, punctuation shape, and confidence without persisting text. The accepted
substitution batch covers five exact profiles: bracketed one- and four-Hangul cores, a plain
two-Hangul core, a terminal-punctuation three-Hangul core, and a plain six-Hangul core. Each
replacement preserves Hangul length and requires either two agreeing crop variants or the bounded
high-confidence enhanced retry; paired four-core wrappers must use the attached-particle wrapper
set. Exact quick/full comparisons recover five target readings. Four failures fully resolve and
`dev-plain-0147` moves only from target to component role, with no unrelated addition. The lone
weak two-Hangul retry candidate and nonauthoritative rounded-crop candidates remain rejected.
Stress, held-out language, promotion, and negative-pointer results are unchanged.

Authoritative detector-relative probing of the 16 length-changing cases finds one further exact
split profile. A single default detector segment containing four Hangul at at least 99.98%
confidence is reconsidered at the 0.01 CTC-space threshold. Recovery requires two edge-complete
two-Hangul readings, a one-pixel boundary overlap bounded to 1.0%-1.5% of line height, at least
99% pitch agreement, at least 99.99% confidence for both parts, and exact recombination. The
bounded helper runs in both segmented and single-segment paths. The exact full comparison removes
only `dev-plain-1544`, with no addition or stage change; quick diagnostics remain byte-identical.
Its whitespace negative activation also resolves. At that pass, all other direct whole-segment
matches require the rejected 0.001 threshold or fail exact recombination.

A category/core pass over the 15 remaining length-changing cases isolates one three-Hangul,
decimal-digit, terminal-ellipsis sequence at 99.47% confidence. Two proportional prefix crops
agree on the same three-Hangul reading at 99.03% and 98.92%; normal-threshold CTC independently
returns that reading with ellipsis at 99.58% and the digit at 99.78%, under a bounded overlap and
pitch profile. The recovery restores their order as two eojeols. An intermediate attempt that
discarded the digit is rejected because it merely moved the target failure to context. The
accepted comparison preserves the digit and removes only `dev-plain-0866`, with no addition or
stage change; quick diagnostics remain byte-identical. Stress, held-out language, promotion, and
negative-pointer results are unchanged.

The remaining weak-threshold exact candidate is a 99.99%-confidence two-Hangul segment whose
two one-Hangul CTC readings are edge-complete, separated by 31%-32% of line height, within
84%-85% pitch agreement, at least 99.995% confident, and exactly recombine. Two independent
midpoint crop boundaries reproduce the same ordered pair at at least 99.98% confidence. These
additional signals bound the 0.001-threshold recovery that was previously rejected on CTC
evidence alone. The exact comparison removes only `dev-plain-1303`, with no addition or stage
change; quick diagnostics remain byte-identical. Stress, held-out language, promotion, and
negative-pointer results are unchanged.

A proportional-crop sweep of the three high-confidence prefix merges isolates one additional
profile. A 99.90%-confidence five-Hangul segment spans 5.96-5.97 line heights. Normal-threshold
CTC exposes two edge-complete readings with a 5%-6% overlap and 94%-95% pitch agreement: the
first is the three-Hangul prefix plus terminal punctuation, while the second exactly reproduces
the two-Hangul suffix. Crops at 2.9 and 3.0 line heights independently reproduce the ordered
3+2 reading at at least 99.89% confidence. Recovery uses the tighter prefix boundary and the
CTC-confirmed suffix geometry, preserving their visible gap. The exact comparison removes only
`dev-plain-0290`, with no addition or stage change; quick diagnostics remain byte-identical, and
its punctuation and whitespace negative activations both resolve. Stress, held-out language,
and promotion results are unchanged.

One eight-Hangul merge then exposes an edge-complete 5+punctuation / 3 CTC split with a
25%-27% positive gap and at least 99% pitch agreement. Crops at 3.25 and 3.7 line heights
independently reproduce the same ordered 5+3 reading at at least 99.6% confidence. The recovery
uses the tighter crop-confirmed prefix geometry and the CTC-confirmed suffix geometry. The exact
comparison removes only `dev-plain-1321`, with no addition or stage change; quick diagnostics
remain byte-identical, its punctuation activation resolves, and stress and held-out language
quality are unchanged.

A privacy-safe category pass over the remaining length-changing wrong-text cases isolates one
nine-character Hangul-punctuation-Hangul-punctuation-Hangul merge at 84.4% confidence and
8.51-8.53 line heights. Weak-threshold CTC exposes four edge-complete segments with two positive
gaps and one small overlap; independent prefix, target, and suffix crops reproduce the ordered
3+3+1 Hangul cores while preserving both punctuation characters. The bounded recovery removes
only `dev-plain-1618` at the target stage, with no addition or changed diagnostic. Quick
diagnostics remain byte-identical; stress, held-out language, promotion, and negative-pointer
quality are unchanged.

One remaining single-region fallback has an eight-character
Hangul-punctuation-Hangul-punctuation reading at 65.2% confidence and spans 9.09-9.10 line
heights. At the 0.005 CTC-space threshold, detector-relative margins, a 32%-34% gap, and
98%-99.5% pitch agreement isolate the two eojeols. The first segment and two prefix crops agree
above 99.96%; the second segment preserves the same three-Hangul core under punctuation wrappers,
and two inner crops confirm that core above 99.86%. Recovery preserves the raw punctuation rather
than rewriting the unresolved dash glyph. The exact comparison removes only `dev-plain-0768` at
the target stage, with no addition or changed diagnostic. Exact transcription remains unchanged;
quick diagnostics are byte-identical, and stress, held-out language, promotion, and negative-pointer
quality do not regress.

One remaining punctuation-wrapped merge has a punctuation-five-Hangul-punctuation / four-Hangul
reading at 86.7% confidence and spans 9.78-9.79 line heights. At the 0.001 CTC-space threshold,
three edge-complete segments expose a 1.5%-1.7% overlap, a 28%-29% following gap, and 69%-70%
pitch agreement; the middle punctuation region has a blank recognition. Independent wrapper,
target, and suffix crops reproduce the same five- and four-Hangul cores at the bounded confidence
floors. Recovery preserves the raw punctuation and uses only the confirmed target/suffix geometry.
The exact comparison removes only `dev-plain-0448` at the target stage, with no addition or changed
diagnostic. Exact transcription remains unchanged; quick diagnostics are byte-identical, and stress,
held-out language, promotion, and negative-pointer quality do not regress.

A five-Hangul merge with terminal punctuation remains at 99.17% confidence and spans 6.41-6.42
line heights. At the 0.002 CTC-space threshold, three edge-complete segments expose a 5%-6%
overlap, a 34%-35% following gap, and 68%-69% pitch agreement. The first segment and two target
crops reproduce the three-Hangul target above 99.99%; two wider prefix crops independently agree
on the internal punctuation above 99.99%, while two suffix crops reproduce the two-Hangul suffix
and distinct terminal punctuation above 98.8%. Recovery retains both observed punctuation marks
and their confirmed geometry. The exact comparison moves only `dev-plain-0994` from target to
context and clears its punctuation activation, with no added failure ID or unrelated change. Its
remaining non-target substitution is recorded as a transcription error, and the 168-case context
audit is complete. Quick diagnostics are byte-identical; stress, held-out language, promotion, and
latency gates do not regress.


Privacy-safe target evidence now reports only boolean expected-target matches, numeric target spans,
prefix/suffix Unicode-category summaries, and cross-probe agreement; it emits no target or
recognized text. Grouping the seven length-changing wrong-text cases with that evidence isolates
one 12 px browser case whose target is split across a punctuation-plus-one-Hangul fragment, a
three-Hangul fragment, and a low-confidence punctuation artifact. Both overlaps are 5.16% of line
height and the surrounding gaps are 15.49% and 20.65%. The left-plus-middle and middle-plus-right
crops agree with the corresponding full-union prefix and suffix at 99.77% and 98.83% confidence,
while the full punctuation-Hangul-punctuation union is 99.59% confident. The bounded recovery uses
the raw fragments only as corroborating evidence, replaces the two covered retry-selected words
with the confirmed wrapper, and preserves unrelated enhanced-retry results.

The exact full comparison moves only `dev-plain-1457` from target to context and adds or changes
no other diagnostic. Its separate non-target one-Hangul sliver is retained because the confirming
union is only 57.64% confident, well below the accepted duplicate-cleanup floor; the text-free
review categorizes it as a non-target OCR transcription error. Target selection rises to 97.95%
and alternative recovery to 95.05%, while popup correctness remains 84.40%. Quick diagnostics are
byte-identical; OCR, functional context, components, dictionary fidelity, stress, held-out
language, promotions, negative pointers, and latency gates do not regress. The full context audit
now has 169 active cases, 207 decisions, 38 resolved IDs, and no missing decision.
Privacy-safe regrouping of the six remaining length-changing wrong-text cases isolates one 14 px
desktop multi-line case. Its target spans a 90.45%-confident one-Hangul fragment and an overlapping
48.72%-confident two-Hangul-plus-terminal-punctuation fragment, followed by a touching
25.02%-confident numeric artifact. The fragments overlap by 4.73% of line height, the preceding gap
is 28.40%, and their union is 3.03 line heights wide. The raw union reproduces their concatenation
at 78.84%; enhanced zero-padding and one-pixel-padding crops reproduce the same
three-Hangul-plus-punctuation reading at 95.86% and 99.63%. All seven tested padding variants retain
the target at the same leading span. The bounded recovery requires the exact raw shape, confidence,
overlap, surrounding geometry, width, and agreement from all three union crops before replacing the
two covered retry-selected words.

The exact full comparison moves only `dev-plain-0458` from target to context. Its target surface
and geometry now match, while structure-only review attributes the remaining line difference to
missed or merged OCR word boundaries elsewhere. Target selection rises to 98.00% and alternative
recovery to 95.10%; popup correctness remains 84.40%. Quick diagnostics remain byte-identical, and
OCR, functional context, sentence transcription, components, dictionary fidelity, stress,
held-out language, promotions, negative pointers, and latency gates do not regress. The full
context audit now has 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the five remaining length-changing wrong-text cases isolates one 40 px
browser quote case. Its line-initial six-character reading has punctuation, three Hangul
syllables, mismatched punctuation, and one final Hangul syllable at 59.41% enhanced confidence
and 5.19 line heights wide. The following two three-Hangul words are at least 99.97% confident,
with gaps of 38.30% and 31.70% of line height. At a 0.001 CTC-space threshold, the suspect reading
separates into four visual parts: weak wrapper evidence, the exact three-Hangul target at 99.96%,
the mismatched punctuation at 51.14%, and the final Hangul suffix at 99.88%. Target crops expanded
by 5% and 10% of line height reproduce the exact target at 99.89% and 99.93%; two suffix crops
likewise agree above 99.88%, and both wrapper crops preserve the same three-Hangul interior.

The bounded recovery requires the exact reading shape, enhanced confidence, line-initial position,
two strong three-Hangul neighbors, word widths, surrounding gaps, four-part CTC geometry, all four
part readings, and all six independent crop confirmations. It emits the wrapper-plus-target and
the final suffix as separate words; punctuation trimming then exposes only the confirmed target
to hit testing. The exact full comparison removes only `dev-plain-1768` from the target failures
and adds or changes no diagnostic. Whole-eojeol OCR rises to 98.33%, target selection to 98.05%,
functional context to 89.55%, first-popup correctness to 84.45%, and alternative recovery to
95.15%. Quick diagnostics remain byte-identical; sentence transcription, components, dictionary
fidelity, stress, held-out language, promotions, negative pointers, and all passing gates do not
regress. The full context audit remains complete at 170 active cases, 208 decisions, 38 resolved
IDs, and no missing decision.

Cross-probing the four remaining length-changing wrong-text cases isolates one 18 px desktop
dash/slash case. Its terminal eight-character reading has punctuation, four Hangul syllables,
mismatched punctuation, and a two-Hangul suffix at 54.76% enhanced confidence and 7.13 line
heights wide. Four preceding pure-Hangul words have lengths three, two, five, and five, at least
99.87% confidence, and an exact surrounding width-and-gap profile. Direct recognition of the
original crop instead supplies a correctly paired wrapper at 54.01%. At a 0.001 CTC-space
threshold, the crop separates into that wrapper at 51.82% and the exact suffix at 99.98%, with a
31.94%-of-line-height gap and matching character pitch.

Five line-height-derived target crops reproduce the same four-Hangul wrapper interior at
99.89%-99.98%. Two wrapper-boundary variants preserve the paired wrapper at 52.68% and 74.53%,
and two suffix-boundary variants preserve the suffix at 99.55% and 99.94%. The bounded recovery
requires the exact line-terminal shape, enhanced and direct confidence windows, all four neighbor
lengths, confidences, widths and gaps, the paired raw wrapper, the two-part CTC geometry and
readings, and all nine crop variants. The exact full comparison removes only
`dev-plain-0732` from the target failures and adds or changes no diagnostic. Whole-eojeol OCR
rises to 98.34%, target selection to 98.10%, functional context to 89.60%, first-popup
correctness to 84.50%, and alternative recovery to 95.20%. Quick diagnostics remain
byte-identical; sentence transcription, components, dictionary fidelity, stress, held-out
language, promotions, negative pointers, and all passing gates do not regress. The full context
audit remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing
decision.

Cross-probing the three remaining length-changing wrong-text cases isolates one line-terminal
wrapper-interior case with two strong preceding words. Its nine-character reading has punctuation,
three Hangul syllables, mismatched punctuation, three Hangul syllables, and terminal punctuation
at 68.09% confidence and 8.09 line heights wide. The preceding three- and two-Hangul words are
both above 99.99% confidence, with the same 30.98%-of-line-height gap.

At a 0.001 CTC-space threshold, the crop separates into a paired wrapper at 82.86%, a
single low-confidence ASCII artifact, and the exact three-Hangul-plus-punctuation suffix at
53.16%. Five line-height-derived target crops reproduce the wrapper interior at
99.95%-99.98%. Two wrapper-boundary variants preserve the independently paired wrapper at
79.37%-81.52%, and three suffix-boundary variants preserve the suffix at 98.99%-99.32%.
The bounded recovery requires the exact three-word confidence, width and gap profile, the
three-part CTC geometry and readings, paired wrapper correction, artifact category, and all ten
crop confirmations. The exact full comparison removes only `dev-plain-1728` from the target
failures and adds or changes no diagnostic. Whole-eojeol OCR rises to 98.35%, target selection
to 98.15%, functional context to 89.65%, first-popup correctness to 84.55%, and alternative
recovery to 95.25%. Quick diagnostics remain byte-identical; sentence transcription,
components, dictionary fidelity, stress, held-out language, promotions, negative pointers, and
all passing gates do not regress. The full context audit remains complete at 170 active cases,
208 decisions, 38 resolved IDs, and no missing decision.

The remaining isolated wrapper-interior case is a single detected region whose enhanced
seven-character reading has punctuation, two Hangul syllables, mismatched punctuation, two Hangul
syllables, and terminal punctuation at 80.90% confidence. Its only default CTC segment is 6.46
line heights wide with stable detector margins; direct recognition of that segment preserves all
Hangul and suffix content at 53.58% while disagreeing only on the suspect punctuation.

At a 0.001 CTC-space threshold, the segment separates into an opening mark at 42.13%, the
two-Hangul target plus alternate punctuation at 92.35%, and the exact suffix at 99.38%. Seven
line-height-derived target crops reproduce the target at 99.95%-99.99%. Five detector-margin
wrapper crops reproduce the same independently paired wrapper at 37.65%-62.18%, and five suffix
crops preserve the suffix at 99.08%-99.38%. The bounded single-region recovery requires the exact
region confidence and shape, default segment margins, original segment reading, three-part CTC
geometry and readings, paired wrapper correction, and all 17 crop confirmations. The exact full
comparison removes only `dev-plain-1486` from the target failures and adds no diagnostic.
Whole-eojeol OCR rises to 98.36%, target selection to 98.20%, functional context to 89.70%,
first-popup correctness to 84.60%, and alternative recovery to 95.30%. Its associated whitespace
activation is also removed, lowering aggregate negative activation to 0.04%. Quick diagnostics
remain byte-identical; sentence transcription, components, dictionary fidelity, stress, held-out
language, promotions, all gates, and every other negative pointer do not regress. The full context
audit remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing
decision.

The final length-changing target failure is a central two-syllable target merged with four-Hangul
text on both sides inside a 12-character recognition. The bounded recovery requires the exact
four-word confidence, width, and gap profile and a mismatched embedded wrapper around the target.
A 0.001 CTC-space probe must independently recover the exact four-Hangul prefix, a paired wrapper
reading containing the same two target syllables, its closing punctuation, and the exact
four-Hangul suffix with the reviewed geometry and confidence floors. Five prefix crops, five
target-only crops, six complete-wrapper crops, and five suffix crops must then reproduce their
respective readings. The low-threshold and complete-wrapper readings may choose different quote
styles only when each is internally paired and both preserve the exact target interior.

The exact full comparison removes only `dev-plain-1170` from the target failures and adds no
diagnostic. Whole-eojeol OCR rises to 98.37%, target selection to 98.25%, functional context to
89.75%, exact sentence transcription to 72.10%, first-popup correctness to 84.65%, and
alternative recovery to 95.35%. Quick diagnostics remain byte-identical; components, dictionary
fidelity, stress behavior outside timing, held-out language, promotions, negative pointers, and
all gates do not regress. The full context audit remains complete at 170 active cases, 208
decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the nine equal-length substitutions isolates one three-Hangul target
whose default segment reading at 57.53% exactly matches the independent oracle, while the generic
enhanced retry replaces it with a different three-Hangul reading at 76.57%. Five one-pixel crop
variants reproduce the direct reading at 56.96%-68.46%; wider padding and shifts that disagree are
excluded from the recovery. The bounded profile requires the exact six-segment line shape,
confidence, width, gap, Hangul, single-letter marker, and structured-ASCII category evidence; the
selected retry must differ only at the three-Hangul candidate, and all five accepted crops must
exactly reproduce the direct reading above separate confidence floors. The detector-derived crop
edges are rounded back to their integer CTC coordinates before verification.

The exact full comparison removes only `dev-plain-0555` from the target failures and adds no
diagnostic. Whole-eojeol OCR rises to 98.38%, target selection to 98.30%, functional context to
89.80%, exact sentence transcription to 72.15%, first-popup correctness to 84.70%, and
alternative recovery to 95.40%. Quick diagnostics remain byte-identical; components, dictionary
fidelity, stress behavior outside timing, held-out language, promotions, negative pointers, and
all gates do not regress. The full context audit remains complete at 170 active cases, 208
decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the eight equal-length substitutions isolates one five-Hangul target
whose direct segment reading at 97.73% disagrees with the independent oracle. Extending that crop
through its touching right wrapper fragment produces a different five-Hangul reading at 99.98%
that exactly matches the oracle. Seven base, pad, trim, and shift crops reproduce that reading at
99.48%-99.99%; three enhanced crops independently agree at 99.67%-99.99%. The fail-closed profile
requires the exact 16-raw-segment/12-selected-word line shape, neighboring Hangul lengths,
punctuation and symbol categories, confidence, width, and gap evidence. The candidate and three
wrapper fragments must retain their reviewed zero-gap geometry, and every direct and enhanced crop
must reproduce the same alternative above its individual confidence floor.

The exact full comparison removes only `dev-plain-0345` from the target failures and adds no
diagnostic. Target selection rises to 98.35%, functional context to 89.85%, first-popup correctness
to 84.75%, and alternative recovery to 95.45%; whole-eojeol OCR, exact sentence transcription,
components, and dictionary fidelity remain unchanged. Quick diagnostics remain byte-identical;
stress behavior outside timing, held-out language, promotions, negative pointers, and all gates do
not regress. The replacement retains the original target geometry and conservative 97.73%
confidence. The full context audit remains complete at 170 active cases, 208 decisions, 38 resolved
IDs, and no missing decision.

Privacy-safe regrouping of the seven remaining equal-length substitutions isolates one four-Hangul
target whose leading wrapper is retained by its direct segment while the matching right wrapper is
recognized separately. Nine paired base, pad, trim, and shift crops reproduce a different pure
four-Hangul interior at 84.36%-99.99%; five enhanced crops independently agree at
99.44%-99.99%. The fail-closed profile requires the exact 14-raw-segment/11-selected-word line
shape, neighboring Hangul lengths, wrapper and symbol categories, confidence, width, and gap
evidence. The base paired crop must preserve the observed opening wrapper and supply its matching
close; every direct and enhanced crop must reproduce the same alternative interior above its
individual confidence floor.

The exact full comparison removes only `dev-plain-1043` from the target failures and adds no
diagnostic or stage change. Target selection rises to 98.40%, functional context to 89.90%,
first-popup correctness to 84.80%, and alternative recovery to 95.50%; whole-eojeol OCR, exact
sentence transcription, components, and dictionary fidelity remain unchanged. Quick diagnostics
remain byte-identical; stress behavior outside timing, held-out language, promotions, negative
pointers, and all gates do not regress. The replacement retains the original punctuation-stripped
target geometry and conservative 84.36% confidence. The full context audit remains complete at
170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the six remaining equal-length substitutions isolates one matched-wrapper
four-Hangul target whose direct segment reading at 92.75% disagrees with the independent oracle.
An enhanced reading of the same crop preserves the wrapper pair and produces a different interior
at 99.13% that exactly matches the oracle. Eight direct edge trims reproduce that interior at
89.27%-99.98%, and seven enhanced base, pad, trim, and shift crops agree at 61.89%-99.92%.
The fail-closed profile requires the exact 12-raw-word/12-selected-word line shape, neighboring
Hangul lengths, terminal punctuation, confidence, width, and gap evidence. The enhanced base must
preserve the observed wrapper pair, and every direct and enhanced crop must reproduce the same
alternative interior above its individual confidence floor.

The exact quick and full comparisons remove only `dev-plain-0019` from the target failures and add
no diagnostic or stage change. Quick target selection reaches 100.00%, functional context 96.00%,
first-popup correctness 91.50%, and alternative recovery 97.50%. Full whole-eojeol OCR rises to
98.39%, target selection to 98.45%, functional context to 89.95%, first-popup correctness to
84.85%, and alternative recovery to 95.55%; exact sentence transcription, components, and
dictionary fidelity remain unchanged. Stress behavior outside timing, held-out language,
promotions, negative pointers, and all gates do not regress. The replacement retains the original
punctuation-stripped target geometry and conservative 61.89% confidence. The full context audit
remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the five remaining equal-length substitutions isolates one two-Hangul
target whose direct segment reading at 87.08% disagrees with the independent oracle. An enhanced
reading of the same crop produces a different two-Hangul reading at 93.43% that exactly matches the
oracle. Four direct boundary variants reproduce that reading at 49.42%-59.07%, and eight enhanced
base, pad, trim, and shift crops agree at 93.43%-99.95%. The fail-closed profile requires the exact
six-raw-word/six-selected-word line shape, neighboring Hangul lengths, two terminal punctuation
signals, confidence, width, and gap evidence. Every direct and enhanced crop must reproduce the same
alternative reading above its individual confidence floor.

The exact full comparison removes only `dev-plain-0897` from the target failures and adds no
diagnostic or stage change. Target selection rises to 98.50%, functional context to 90.00%, exact
sentence transcription to 72.20%, first-popup correctness to 84.90%, and alternative recovery to
95.60%; whole-eojeol OCR, components, and dictionary fidelity remain unchanged. Quick diagnostics
remain byte-identical; stress behavior outside timing, held-out language, promotions, negative
pointers, and all gates do not regress. The replacement retains the original target geometry and
conservative 49.42% confidence. The full context audit remains complete at 170 active cases, 208
decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the four remaining equal-length substitutions isolates one terminally
punctuated three-Hangul target whose direct and enhanced retry readings disagree at 48.64% and
50.98%, and both disagree with the independent oracle. A broad boundary grid recovers the oracle
in 113 direct and 156 enhanced crops. Five fixed direct crops and eight enhanced crops reproduce
the same three-Hangul reading at 88.11%-91.90%. The fail-closed profile requires the exact
nine-raw-word/eight-selected-word line shape, neighboring Hangul lengths, two terminal punctuation
signals, one excluded low-confidence ASCII tail, and reviewed confidence, width, and gap evidence.
Every fixed crop must reproduce the same alternative above its individual confidence floor; the
original terminal punctuation is then restored before the existing punctuation split.

The exact full comparison removes only `dev-plain-0550` from the target failures and adds or
changes no diagnostic. Whole-eojeol OCR rises to 98.40%, target selection to 98.55%, functional
context to 90.05%, exact sentence transcription to 72.25%, first-popup correctness to 84.95%, and
alternative recovery to 95.65%; components and dictionary fidelity remain unchanged. Quick
diagnostics remain byte-identical; stress behavior outside timing, held-out language, promotions,
negative pointers, and all gates do not regress. The replacement retains the existing target and
punctuation geometry and conservative 48.64% confidence. The full context audit remains complete
at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

Privacy-safe regrouping of the three remaining equal-length substitutions isolates one terminal
paired-punctuation target whose selected four-Hangul interior disagrees with the independent
oracle. Seven fixed direct boundary crops and seven enhanced crops reproduce the same different
four-Hangul reading at 53.17%-82.34%. The fail-closed profile requires the exact
13-raw-word/12-selected-word line shape, one same-shape enhanced retry, matched wrappers,
neighboring character categories, confidence, width, and gap evidence. Every fixed crop must
reproduce the same alternative above its individual confidence floor before the existing
punctuation splitter exposes the recovered interior.

The exact full comparison removes only `dev-plain-0417` from the target failures and adds or
changes no diagnostic. Target selection rises to 98.60%, functional context to 90.10%,
first-popup correctness to 85.00%, and alternative recovery to 95.70%; whole-eojeol OCR, exact
sentence transcription, components, and dictionary fidelity remain unchanged. Quick diagnostics
remain byte-identical; stress behavior outside timing, held-out language, promotions, negative
pointers, and all gates do not regress. The replacement retains the existing target and
punctuation geometry and conservative 53.17% confidence. The full context audit remains complete
at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

A geometry-only audit of the 28 remaining target failures finds that 24 of the 26 no-hit targets
survive exactly inside one raw segment; the other two have no raw segment occurrence. One isolated
case is a single Hangul glyph fused with terminal punctuation. A broad detector-relative grid
recovers that same already-present glyph in 206 direct and 211 enhanced crops. Seven fixed direct
and seven enhanced trim/shift crops reproduce it at 99.97%-99.99%. The fail-closed profile requires
the exact 12-raw-word/12-selected-word line shape, neighboring character categories, confidence,
width, gap, and one reviewed overlap. Every fixed crop must reproduce the candidate's existing
Hangul glyph above its individual confidence floor before replacing equal-width punctuation
geometry with the consensus crop box.

The exact full comparison removes only `dev-plain-1731` from the target failures and adds or
changes no diagnostic. Target selection rises to 98.65%, functional context to 90.15%,
first-popup correctness to 85.05%, and alternative recovery to 95.75%; whole-eojeol OCR, exact
sentence transcription, components, and dictionary fidelity remain unchanged. Quick diagnostics
remain byte-identical; stress behavior outside timing, held-out language, promotions, negative
pointers, and all gates do not regress. The recovery retains the existing one-Hangul reading at
90.28% confidence while moving its box inside the reviewed crop consensus. The full context audit
remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One remaining no-hit target is an existing single Hangul glyph between matched punctuation
wrappers whose enhanced retry drops only the opening wrapper. A broad detector-relative grid
recovers the same glyph in 167 direct and 175 enhanced crops. Seven fixed direct and seven
enhanced trim/shift crops reproduce it at 99.87%-99.96%. The fail-closed profile requires the
exact ten-raw-word/eight-selected-word line shape, two excluded low-confidence ASCII boundary
artifacts, one same-geometry wrapper-dropping retry, matched wrappers, and reviewed neighboring
character categories, confidence, width, and gap evidence. Every fixed crop must reproduce the
candidate's existing Hangul glyph above its individual confidence floor before replacing the
wrapper-width geometry with the central consensus crop.

The exact full comparison moves only `dev-plain-1578` from the target stage to a newly exposed
primary-lemma analysis failure. Target selection rises to 98.70% and functional context to
90.20%; whole-eojeol OCR, exact sentence transcription, components, dictionary fidelity,
first-popup correctness, and alternative recovery remain unchanged. Quick diagnostics are
byte-identical; stress behavior outside timing, held-out language, promotions, negative pointers,
and all gates do not regress. The recovery retains the existing one-Hangul reading at 99.48%
confidence while moving its box inside the reviewed crop consensus. The full context audit remains
complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One high-confidence raw segment fused a one-Hangul target and its trailing punctuation with the
following two-Hangul eojeol. Four word-local CTC-space thresholds reproduce the same two-part
boundary while the ordinary threshold intentionally remains fused. The fail-closed profile
requires the exact ten-raw-word/ten-selected-word line shape, punctuation order, neighboring
character categories, confidence, width, and gap evidence. A direct and enhanced punctuation-
aware boundary crop, seven direct/enhanced target crops, and eight direct/enhanced following-word
crops must all reproduce their existing readings above individual confidence floors. The
confirmed split keeps punctuation in sentence context, gives only the Hangul glyph hoverable
target geometry, and restores the following eojeol from its independent CTC boundary.

The exact full comparison removes only `dev-plain-0252` from the target failures and adds or
changes no diagnostic. Whole-eojeol OCR rises to 98.41%, target selection to 98.75%, functional
context to 90.25%, exact sentence transcription to 72.30%, first-popup correctness to 85.10%,
and alternative recovery to 95.80%; components and dictionary fidelity remain unchanged. Quick
diagnostics remain byte-identical; stress behavior outside timing, held-out language, promotions,
negative pointers, and all gates do not regress. The recovery exposes the existing target at
99.45% confidence and retains the following eojeol at 99.88%. The full context audit remains
complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One lower-confidence eight-Hangul segment fused the leading three-Hangul target across an
unrecognized ellipsis boundary with a following five-Hangul eojeol. Five low CTC-space thresholds
reproduce the same overlapping two-part boundary, while the next threshold intentionally remains
fused. The fail-closed profile requires the exact eight-raw-word/eight-selected-word line shape,
neighboring character categories, confidence, width, and gap evidence. Seven direct/enhanced
target crops and seven direct/enhanced following-word crops must reproduce the two existing
readings above separate confidence floors. The confirmed split ends target geometry before the
ellipsis probe and begins the following word at its independently confirmed CTC boundary.

The exact full comparison removes only `dev-plain-0361` from the target failures and adds or
changes no diagnostic. Whole-eojeol OCR rises to 98.42%, target selection to 98.80%, functional
context to 90.30%, first-popup correctness to 85.15%, and alternative recovery to 95.85%; exact
sentence transcription, components, and dictionary fidelity remain unchanged. The recovery also
removes that case's punctuation-probe activation, reducing aggregate main negative activation to
0.03% and punctuation activation to two of 1,582 (0.13%). Quick diagnostics remain byte-identical;
stress behavior outside timing, held-out language, promotions, and all gates do not regress. The
recovery exposes the existing target and following eojeol at a conservative 98.81% confidence.
The full context audit remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and
no missing decision.

One high-confidence ten-character segment fused a leading three-Hangul target with a following
six-Hangul eojeol and terminal punctuation. Six low CTC-space thresholds reproduce the same
two-part boundary while the ordinary threshold intentionally remains fused. The fail-closed
profile requires the exact two-raw-word/two-selected-word line shape, terminal punctuation order,
confidence, width, and gap evidence. Seven direct/enhanced target crops and six direct/enhanced
following-word crops must reproduce the two existing readings above separate confidence floors.
The confirmed split ends target geometry before the punctuation probe, begins the following word
at its independently confirmed CTC boundary, and leaves ordinary eojeol construction to exclude
terminal punctuation from hoverable geometry.

The exact full comparison removes only `dev-plain-1428` from the target failures and adds or
changes no diagnostic. Whole-eojeol OCR rises to 98.43%, target selection to 98.85%, functional
context to 90.35%, first-popup correctness to 85.20%, and alternative recovery to 95.90%; exact
sentence transcription, components, and dictionary fidelity remain unchanged. The recovery also
removes that case's punctuation-probe activation, reducing aggregate main negative activation to
0.02% and punctuation activation to one of 1,582 (0.06%). Quick diagnostics remain byte-identical;
stress behavior outside timing, held-out language, promotions, and all gates do not regress. The
recovery exposes the existing target at 99.14% confidence and retains the following eojeol at a
conservative 98.82%. The full context audit remains complete at 170 active cases, 208 decisions,
38 resolved IDs, and no missing decision.

One isolated high-confidence nine-character segment fused a leading three-Hangul target, one
punctuation mark, and a following five-Hangul eojeol. Five low CTC-space thresholds reproduce the
same overlapping three-part boundary, two intermediate thresholds reproduce the target-plus-
punctuation and following-word boundary, and the ordinary threshold remains fused. The fail-closed
profile requires the exact one-word Unicode shape, crop size, confidence, width, and full
multi-threshold signature. Direct and enhanced recognition of the punctuated boundary, seven
target crops, and seven following-word crops must independently reproduce the existing readings
above separate confidence floors. The recovered sentence retains the punctuation, while target
geometry ends before its probe and following-word geometry begins after the whitespace probe.

The exact full comparison removes only `dev-plain-1312` from the target failures and adds or
changes no other diagnostic. Whole-eojeol OCR rises to 98.44%, target selection to 98.90%,
functional context to 90.40%, exact sentence transcription to 72.35%, first-popup correctness to
85.25%, and alternative recovery to 95.95%; components and dictionary fidelity remain unchanged.
Quick diagnostics remain byte-identical; stress, held-out language, promotions, negative
activation, and all gates do not regress. The recovery exposes both existing eojeols at a
conservative 99.25% confidence. The full context audit remains complete at 170 active cases, 208
decisions, 38 resolved IDs, and no missing decision.

One high-confidence seven-character segment contained a punctuation-wrapped single-Hangul target
and a following four-Hangul eojeol. Its five-raw-word/six-selected-word line has a unique geometry,
confidence, Unicode-shape, and gap profile. Fifteen low and ordinary CTC-space thresholds must
reproduce the complete segmentation signature, enhanced recognition must reproduce the full
candidate, and five wrapper, seven target, and seven following-word crops must all reproduce the
existing readings in both direct and enhanced form above separate confidence floors. The
fail-closed recovery changes only the two selected boxes: it moves target geometry inside the
reviewed crop consensus, retains the wrapper punctuation, and begins the following eojeol at its
independently confirmed boundary.

The exact full comparison removes only `dev-plain-1543` from the target failures and adds or
changes no other diagnostic. Target selection rises to 98.95%, functional context to 90.45%,
exact sentence transcription to 72.40%, first-popup correctness to 85.30%, and alternative
recovery to 96.00%; whole-eojeol OCR, components, and dictionary fidelity remain unchanged. Quick
diagnostics remain byte-identical; stress, held-out language, promotions, negative activation,
and all gates do not regress. The recovery exposes the existing target at a conservative 92.72%
confidence and retains the following eojeol at 99.40%. The full context audit remains complete at
170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One isolated high-confidence 12-character mixed segment began with the intact two-Hangul target,
followed by punctuation and a structured ASCII suffix. The fail-closed profile requires the exact
single-word crop geometry, confidence, character-category sequence, ASCII letter/digit counts,
and complete 15-threshold CTC signature. Eleven low thresholds reproduce the same two-part
boundary, the lowest threshold exposes one additional internal structured boundary, and the
ordinary thresholds remain fused. Enhanced full-crop recognition must reproduce the candidate,
while five punctuation-attached target crops and six structured-suffix crops must reproduce both
sides in direct and enhanced form above separate confidence floors. The confirmed split preserves
the punctuation and restores the missing sentence space while exposing only the Hangul prefix as
a hoverable eojeol.

The exact full comparison removes only `dev-plain-1359` from the target failures and adds or
changes no other diagnostic. Target selection rises to 99.00%, functional context to 90.50%,
exact sentence transcription to 72.45%, first-popup correctness to 85.35%, and alternative
recovery to 96.05%; rounded whole-eojeol OCR, components, and dictionary fidelity remain
unchanged. Quick diagnostics remain byte-identical; stress, held-out language, promotions,
negative activation, and all gates do not regress. The recovery exposes the target at a
conservative 97.58% confidence and retains the structured suffix at 74.16%. The full context audit
remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One isolated nine-character punctuation-wrapped segment fused a three-Hangul target to a
four-Hangul following eojeol. The fail-closed profile requires the exact six-word line geometry,
character-category shape, confidence, normalized widths and gaps, and full 15-threshold CTC
signature. Six low thresholds expose three fragments, five middle thresholds expose the same
two-part boundary, and four ordinary thresholds remain fused. Enhanced full-crop recognition must
reproduce the candidate, while seven target and seven following-word crops must independently
reproduce both sides in direct and enhanced form above separate confidence floors. The confirmed
split preserves the punctuation, restores the missing sentence space, and exposes only the
three-Hangul interior as the hover target.

The exact full comparison removes only `dev-plain-1362` from the target failures and adds or
changes no other diagnostic. Whole-eojeol OCR rises to 98.45%, target selection to 99.05%,
functional context to 90.55%, first-popup correctness to 85.40%, and alternative recovery to
96.10%; exact sentence transcription, components, and dictionary fidelity remain unchanged.
Quick diagnostics remain byte-identical; stress, held-out language, promotions, negative
activation, and all gates do not regress. The recovered target and following eojeol both retain
the conservative fused-candidate confidence of 86.29%. The full context audit remains complete at
170 active cases, 208 decisions, 38 resolved IDs, and no missing decision.

One isolated ten-character segment fused an em-dash-prefixed three-Hangul target, an ASCII hyphen,
and a five-Hangul following eojeol. The fail-closed profile requires the exact six-word line
geometry, punctuation code points, character-category shape, confidence, normalized widths and
gaps, and full 15-threshold CTC signature. Two lowest thresholds expose five fragments, the next
exposes four, one exposes two overlapping regions, and the remaining eleven stay fused. Enhanced
full-crop recognition must reproduce the candidate, while seven target and seven following-word
crops must independently reproduce both sides in direct and enhanced form above separate
confidence floors. The confirmed split preserves both punctuation marks, restores the missing
sentence space, and exposes only the three-Hangul target interior.

The exact full comparison removes only `dev-plain-1692` from the target failures and adds or
changes no other diagnostic. Whole-eojeol OCR rises to 98.46%, target selection to 99.10%,
functional context to 90.60%, first-popup correctness to 85.45%, and alternative recovery to
96.15%; exact sentence transcription, components, and dictionary fidelity remain unchanged.
Quick diagnostics remain byte-identical; stress, held-out language, promotions, negative
activation, and all gates do not regress. The target and following eojeol retain the conservative
fused-candidate confidence of 85.18%. The full context audit remains complete at 170 active cases,
208 decisions, 38 resolved IDs, and no missing decision.

One isolated ten-character segment fused a five-Hangul prefix to a hyphen-prefixed three-Hangul
target and trailing em dash. The fail-closed profile requires the exact isolated-line geometry,
punctuation code points, character-category shape, confidence, crop dimensions, default segment,
and full 15-threshold CTC signature. The lowest thresholds expose the observed overlapping
fragments, the middle thresholds repeatedly expose the same prefix boundary, and the ordinary
thresholds remain fused. Direct and enhanced recognition of the interior candidate plus enhanced
full-crop recognition must reproduce the complete candidate. Seven prefix crops and seven
hyphen-plus-target crops must also independently reproduce both sides in direct and enhanced form
above separate confidence floors. The confirmed split preserves both punctuation marks, restores
the missing sentence space, and exposes only the three-Hangul interior as the hover target.

The exact full comparison advances only `dev-plain-0220` from target failure to analysis failure,
clears its whitespace negative activation, and adds no diagnostic ID. Whole-eojeol OCR rises to
98.47%, target selection to 99.15%, and functional context to 90.65%; exact sentence
transcription, components, dictionary fidelity, first-popup correctness, and alternative recovery
remain unchanged. Quick diagnostics remain byte-identical; stress, held-out language, promotions,
and all gates do not regress. The recovered words retain the conservative full-enhanced confidence
of 66.37%. The full context audit remains complete at 170 active cases, 208 decisions, 38 resolved
IDs, and no missing decision.

One terminal eight-character segment fused a three-Hangul prefix with a two-Hangul target while
misreading its surrounding smart quotes as an ASCII hyphen, em dash, and ellipsis. The fail-closed
profile requires the exact five-word line geometry, character-category and punctuation-code-point
shape, confidence, crop dimensions, default segments, and full 15-threshold CTC signature. The
lowest threshold exposes five fragments, the next exposes four, seven middle thresholds expose
the same two-part boundary with an eight-pixel gap, and six ordinary thresholds remain fused.
Direct and enhanced candidate recognition must preserve the original fused reading. Seven prefix
crops independently reproduce the three-Hangul word above 99.94%, eight direct and enhanced
wrapper crops agree on the same smart-quote-wrapped target, and seven target-only crops reproduce
both Hangul characters above 99.95%. The confirmed correction replaces the unstable punctuation
reading with the crop-consensus quotes, restores the missing space, and exposes only the quoted
two-Hangul interior as the hover target.

The exact full comparison removes only `dev-plain-1564` and adds or changes no diagnostic. Whole-
eojeol OCR rises to 98.48%, target selection to 99.20%, functional context to 90.70%, first-popup
correctness to 85.50%, and alternative recovery to 96.20%; exact sentence transcription,
components, and dictionary fidelity remain unchanged. Quick diagnostics remain byte-identical;
stress, held-out language, and promotions do not regress. Removing this case also clears the last
negative activation, so aggregate and every negative category are now 0.00%. The prefix retains
78.02% confidence and the quote-wrapped target conservatively retains 43.34%. The full context
audit remains complete at 170 active cases, 208 decisions, 38 resolved IDs, and no missing
decision.

The next reviewed recovery handles `dev-plain-1931`, whose terminal two-Hangul target
survives inside an ASCII-apostrophe reading fused to a following one-Hangul eojeol while the
default detector also emits an empty trailing segment. The fail-closed profile requires the exact
12-word line geometry and confidence shape, 13 default segments, and separate 15-threshold CTC
signatures for the fused and combined terminal crops. Direct and enhanced candidate recognition
must preserve the original five-character reading. A boundary crop and seven direct/enhanced
wrapper crops must agree on the corrected curly-single-quote wrapper; seven target-only and seven
following-character crops must independently reproduce the two interior Hangul characters and the
following Hangul character at high confidence. The recovered raw boxes retain the observed
three-pixel glyph overlap, while punctuation trimming exposes only the two-Hangul interior as the
target and keeps the following eojeol independently hittable.

The exact full comparison removes only `dev-plain-1931` from target failures and adds or changes
no diagnostic. Target selection rises to 99.25%, functional context to 90.75%, exact sentence
transcription to 72.50%, fully correct first popups to 85.55%, and alternative recovery to
96.25%; whole-eojeol OCR, components, dictionary fidelity, promotions, and every negative category
remain unchanged. Quick diagnostics are byte-identical, while stress and held-out language results
do not regress. The corrected wrapper conservatively retains 56.06% confidence and the following
eojeol 58.95%. The full context audit remains complete at 170 active cases, 208 decisions, 38
resolved IDs, and no missing decision.

Compared with the earlier full report, the accepted cleanups resolve 54 context IDs. The target
recovery above exposes one existing non-target transcription error at context, leaving a net
reduction of 53 active cases. The preceding cleanup permits a one-pixel overlap of at most 7.5%
of a small line only under the existing exact combined-recognition duplicate profile. It resolves
two full-tier context cases while leaving the quick diagnostics byte-identical. The same profile
now accepts an ASCII digit as the leading artifact only when recognizing the union at 99% or
better omits it and exactly reproduces the following Hangul word. This resolves three additional
full-tier cases without a new failure or quick-tier change.

The next reviewed recovery handles `dev-plain-1052`, where a terminal em-dash-wrapped
two-Hangul target and following two-Hangul eojeol are fused after OCR substitutes the inner em dash
with an ASCII hyphen. The detector also emits an empty trailing segment. The fail-closed profile
requires the exact eight-word line geometry and confidence shape, nine default segments, and the
complete 15-threshold CTC signature: seven fragments at the lowest threshold, three at the next,
five two-part wrapper/following separations, and eight fused ordinary thresholds. Direct and
enhanced recognition must preserve the original six-character candidate. Three full-wrapper crops
must independently reproduce the corrected em-dash wrapper in both modes; seven direct/enhanced
boundary crops must recover the inner em dash, while seven target and seven following-eojeol crops
must reproduce their Hangul readings above 99.99%. The recovered CTC boxes expose the wrapped
target and following eojeol independently while leaving both dash glyphs non-hoverable.

The exact full comparison removes only `dev-plain-1052` from target failures and adds or changes
no diagnostic. Whole-eojeol OCR rises to 98.49%, target selection to 99.30%, functional context to
90.80%, exact sentence transcription to 72.55%, fully correct first popups to 85.60%, and
alternative recovery to 96.30%; components, dictionary fidelity, promotions, and every negative
category remain unchanged. Quick diagnostics are byte-identical, while stress and held-out
language results do not regress. The corrected wrapper conservatively retains 50.83% confidence
and the following eojeol 70.57%. The full context audit remains complete at 170 active cases, 208
decisions, 38 resolved IDs, and no missing decision.

One isolated single-segment line fused an em-dash-wrapped four-Hangul target to a seven-Hangul
following eojeol while whole-line recognition substituted curly quotes and ASCII hyphens. The
fail-closed profile requires the exact line geometry, confidence, crop dimensions, default segment,
punctuation code points, character-category shape, and complete 15-threshold CTC signature. Direct
and enhanced recognition of the isolated candidate must agree on its leading em dash, four target
syllables, substituted inner hyphen, and seven following syllables. Seven direct/enhanced boundary
crops must recover the inner em dash; seven target, seven following-eojeol, and seven
target-plus-boundary crops must independently reproduce their readings above separate confidence
floors. The correction pairs the independently confirmed em dashes, restores the missing sentence
space, and uses crop-supported boxes that exclude both punctuation and whitespace from hover
geometry.

The exact full comparison removes only `dev-plain-1152` from target failures and adds or changes
no other main diagnostic stage count. Whole-eojeol OCR rises to 98.50%, target selection to 99.35%,
functional context to 90.85%, exact sentence transcription to 72.60%, fully correct first popups to
85.65%, and alternative recovery to 96.35%; components, dictionary fidelity, promotions, and every
main negative category remain unchanged. Quick diagnostics are byte-identical, while stress and
held-out language results do not regress. The corrected wrapper conservatively retains 51.64%
confidence and the following eojeol 67.44%. The full context audit remains complete at 170 active
cases, 208 decisions, 38 resolved IDs, and no missing decision.

One internal seven-character segment fused a three-Hangul prefix to an em-dash-wrapped
two-Hangul target while full-candidate recognition substituted the opening em dash with an ASCII
hyphen. The fail-closed profile requires the exact six-segment line geometry and confidence shape,
neighboring character categories, crop dimensions, default segments, and complete 15-threshold CTC
signature. The lowest threshold exposes seven fragments, the next exposes four, six thresholds
expose the same prefix and two overlapping wrapper fragments, three expose the prefix and complete
wrapper, and four ordinary thresholds stay fused. The 0.01 direct crop must recover the exact
paired em-dash wrapper while the enhanced crop preserves both boundaries as ASCII hyphens. Seven
prefix, seven target, seven opening-dash, and seven closing-dash crops must independently reproduce
their readings in direct and enhanced form; six direct wrapper crops must retain paired em dashes
while their enhanced readings retain the confirmed opening em dash. The correction restores the
missing space and paired wrapper while keeping both punctuation glyphs outside target geometry.

The exact full comparison removes only `dev-plain-1806` from target failures and adds or changes
no other main diagnostic stage count. Whole-eojeol OCR rises to 98.51%, target selection to 99.40%,
functional context to 90.90%, exact sentence transcription to 72.65%, fully correct first popups to
85.70%, and alternative recovery to 96.40%; components, dictionary fidelity, promotions, and every
main negative category remain unchanged. Quick diagnostics are byte-identical, while stress and
held-out language results do not regress. The corrected prefix conservatively retains 59.69%
confidence and the wrapped target 48.29%. The full context audit remains complete at 170 active
cases, 208 decisions, 38 resolved IDs, and no missing decision.

One internal seven-character segment fused a three-Hangul prefix to a curly-quoted two-Hangul
target while the full candidate substituted its closing quote with an ASCII quote. The fail-closed
profile requires the exact six-segment line geometry, neighboring category and confidence shape,
crop dimensions, default segments, and complete 15-threshold CTC signature. Eleven low thresholds
must expose the same overlapping prefix and wrapper crops while four ordinary thresholds remain
fused. Direct and enhanced candidate, prefix, and wrapper crops must agree; six prefix, seven
wrapper, six target, seven opening-quote, and seven closing-quote variants must independently
reproduce their expected readings above separate confidence floors. The correction restores the
missing space and paired curly wrapper, with proportional inner geometry that excludes both quote
glyphs and the restored whitespace from target hover testing.

The exact full comparison removes only `dev-plain-0822` from target failures and adds or changes no
other diagnostic. Whole-eojeol OCR rises to 98.52%, target selection to 99.45%, functional context
to 90.95%, exact sentence transcription to 72.70%, fully correct first popups to 85.75%, and
alternative recovery to 96.45%; components, dictionary fidelity, promotions, and every negative
category remain unchanged. Quick diagnostics are byte-identical, while stress and held-out
language results do not regress. The corrected prefix conservatively retains 58.61% confidence
and the wrapped target 53.78%.

One internal eight-character segment fused a three-Hangul prefix to a curly-quoted three-Hangul
target while the full candidate retained an ASCII closing quote and the higher-confidence enhanced
retry remained fused. The fail-closed profile requires the exact six-segment line geometry,
neighboring category and confidence shape, raw and selected confidence ranges, crop dimensions,
default segments, and complete 15-threshold CTC signature. Eleven low thresholds must expose the
same prefix and wrapper crops while four ordinary thresholds remain fused. Direct and enhanced
candidate, prefix, and wrapper crops must agree; six prefix, seven wrapper, seven target, seven
opening-quote, and six closing-quote variants must independently reproduce their readings above
separate confidence floors. The correction restores the missing space and paired curly wrapper
while keeping both quotes and whitespace outside target hover geometry.

The exact full comparison removes only `dev-plain-0210` from target failures and adds or changes no
other diagnostic. Whole-eojeol OCR rises to 98.53%, target selection to 99.50%, functional context
to 91.00%, exact sentence transcription to 72.75%, fully correct first popups to 85.80%, and
alternative recovery to 96.50%; components, dictionary fidelity, promotions, and every negative
category remain unchanged. Quick diagnostics are byte-identical, while stress and held-out
language results do not regress. The corrected prefix conservatively retains 60.98% confidence
and the wrapped target 49.94%.

One leading seven-character segment fused a two-Hangul prefix to a curly-single-quoted
three-Hangul target while retaining an ASCII closing quote; its enhanced retry substituted the
opening quote too. The fail-closed profile requires the exact two-segment line geometry, neighboring
category and confidence shape, crop dimensions, default segments, and complete 15-threshold CTC
signature. Ten low thresholds must expose the same prefix and wrapper crops while five ordinary
thresholds remain fused. Direct and enhanced candidate, prefix, and wrapper crops must retain the
same Hangul; six prefix, seven wrapper, seven target, seven opening-quote, and five closing-quote
variants must independently reproduce their readings above separate confidence floors. The
correction restores the missing space and paired curly wrapper while keeping both quotes and
whitespace outside target hover geometry.

The exact full comparison removes only `dev-plain-1711` from target failures and adds or changes no
other diagnostic: adding that one privacy-safe categorical entry back to the new report reproduces
the prior accepted SHA-256 exactly. Whole-eojeol OCR rises to 98.54%, target selection to 99.55%,
functional context to 91.05%, exact sentence transcription to 72.80%, fully correct first popups to
85.85%, and alternative recovery to 96.55%; components, dictionary fidelity, promotions, and every
negative category remain unchanged. Quick diagnostics are byte-identical, while stress and
held-out language results do not regress. The corrected prefix conservatively retains 51.58%
confidence and the wrapped target 47.49%.

One 14 px browser multi-line case ends in a seven-character segment containing a four-Hangul
prefix, an opening curly quote, a one-Hangul target, and an ASCII close. Enhanced recognition
restores the paired curly close. The fail-closed profile requires the exact five-segment line,
candidate geometry, confidence and character-category shape, crop dimensions, default segments,
and complete 15-threshold CTC signature. Six prefix, seven complete-wrapper, seven target, seven
opening-quote, and five closing-quote variants must independently reproduce the reviewed readings
in direct and enhanced recognition above separate confidence floors. The confirmed split restores
the missing space and paired wrapper while keeping quotes and whitespace outside hover geometry.

The exact full comparison removes only `dev-plain-0850` from target failures and adds or changes no
other diagnostic: adding that one privacy-safe categorical entry back to the new report reproduces
the prior accepted SHA-256 exactly. Target selection rises to 99.60%, functional context to 91.10%,
exact sentence transcription to 72.85%, fully correct first popups to 85.90%, and alternative
recovery to 96.60%; rounded whole-eojeol OCR, components, dictionary fidelity, promotions, and every
negative category remain unchanged. Quick diagnostics are byte-identical, while stress and
held-out language results do not regress. The corrected prefix conservatively retains 56.72%
confidence and the wrapped target 48.10%.

One 24 px browser single-line case contains an opening curly quote, a three-Hangul target, an
ASCII close, and a following three-Hangul word in one eight-character segment. Direct and enhanced
recognition preserve that candidate, while bounded crops independently recover the paired curly
wrapper, target, and following word. The fail-closed profile requires the exact four-segment line,
candidate geometry, confidence and character-category shape, crop dimensions, default segments,
and complete 15-threshold CTC signature. Seven target, following-word, paired-wrapper,
opening-quote, and closing-quote variants must independently reproduce the reviewed readings in
direct and enhanced recognition above separate confidence floors. The confirmed split restores
the missing space and paired wrapper while keeping both quotes and whitespace outside target hover
geometry.

The exact full comparison removes only `dev-plain-0502` from target failures and adds or changes no
other diagnostic: adding that one privacy-safe categorical entry back to the new report reproduces
the prior accepted SHA-256 exactly. Whole-eojeol OCR rises to 98.55%, target selection to 99.65%,
functional context to 91.15%, exact sentence transcription to 72.90%, fully correct first popups to
85.95%, and alternative recovery to 96.65%; components, dictionary fidelity, promotions, and every
negative category remain unchanged. Quick diagnostics are byte-identical, while stress and
held-out language results do not regress. The paired wrapper conservatively retains 56.68%
confidence and the following word 57.32%.

One 24 px browser single-line case contains an ASCII opening quote, a two-Hangul target, a curly
close, and one following Hangul glyph in the first of two detector segments. Direct and enhanced
recognition preserve that candidate, while a bounded wrapper crop restores paired curly quotes.
The fail-closed profile requires the exact two-segment line, candidate geometry, confidence and
character-category shape, crop dimensions, default segments, and complete 15-threshold CTC
signature. Seven target, following-word, paired-wrapper, opening-quote, and closing-quote variants
must independently reproduce the reviewed readings in direct and enhanced recognition above
separate confidence floors. The confirmed split restores the missing word boundary and paired
wrapper while keeping both quotes and whitespace outside target hover geometry.

The exact full comparison removes only `dev-plain-0758` from target failures and adds or changes no
other diagnostic: adding that one privacy-safe categorical entry back at its original position
reproduces the prior accepted report byte-for-byte and its SHA-256 exactly. Whole-eojeol OCR rises
to 98.56%, target selection to 99.70%, functional context to 91.20%, exact sentence transcription
to 72.95%, fully correct first popups to 86.00%, and alternative recovery to 96.70%; components,
dictionary fidelity, promotions, and every negative category remain unchanged. Quick diagnostics
are byte-identical, while stress and held-out language results do not regress. The paired wrapper
and following glyph conservatively retain 50.94% confidence.

One 14 px browser multi-line case contains an opening curly quote, a four-Hangul target, a
substituted ASCII close, and a four-Hangul following word in the fourth of six detector segments.
Direct recognition preserves the mismatched close while enhanced recognition restores the paired
curly close. The fail-closed profile requires the exact six-segment line, candidate geometry,
confidence and character-category shape, crop dimensions, default segments, and complete
15-threshold CTC signature. Fixed CTC crops plus seven target, following-word, paired-wrapper,
opening-quote, and closing-quote variants must independently reproduce the reviewed readings in
direct and enhanced recognition above separate confidence floors. The confirmed split restores
the missing word boundary and paired wrapper while keeping both quotes and whitespace outside
target hover geometry.

The exact full comparison removes only `dev-plain-1298` from target failures and adds or changes no
other diagnostic: reinserting that one 262-character privacy-safe record at its original position
reproduces the prior report byte-for-byte and its SHA-256 exactly. Whole-eojeol OCR rises to 98.57%,
target selection to 99.75%, functional context to 91.25%, exact sentence transcription to 73.00%,
fully correct first popups to 86.05%, and alternative recovery to 96.75%; components, dictionary
fidelity, promotions, and every negative category remain unchanged. Quick diagnostics are
byte-identical, while stress and held-out language results do not regress. The paired wrapper
conservatively retains 45.72% confidence and the following word 61.04%.

One 32 px desktop multi-line case contains a seven-character candidate in the third of three
overlapping detector segments: an opening curly quote, a one-Hangul target, a substituted ASCII
close, a three-character structured following word, and a misplaced terminal curly close. Direct
and enhanced recognition preserve that low-confidence candidate. The fail-closed profile requires
the exact three-segment line, candidate geometry, confidence and character-category shape, crop
dimensions, default segments, and complete 15-threshold CTC signature. The four lowest thresholds
expose the same overlapping wrapper/following split while the remaining eleven stay fused. Fixed
wrapper and following crops plus seven wrapper, target, following-word, opening-quote, and
closing-quote variants must independently reproduce the reviewed readings in direct and enhanced
recognition above separate confidence floors. The confirmed split restores the paired curly
wrapper and missing word boundary, removes the misplaced terminal close, and keeps both quotes
and whitespace outside target hover geometry.

The exact full comparison advances only `dev-plain-1391` from target failure to the
`grammar_roles` analysis stage and adds or removes no ID. Replacing that one 275-character
privacy-safe record with its prior 262-character form reproduces the previous report byte-for-byte
and its SHA-256 exactly. Whole-eojeol OCR rises to 98.58%, target selection to 99.80%, functional
context to 91.30%, and exact sentence transcription to 73.05%; components, dictionary fidelity,
fully correct first popups, alternative recovery, promotions, and every negative category remain
unchanged. Quick diagnostics are byte-identical, while stress and held-out language results do not
regress. The paired wrapper and structured following word conservatively retain 49.52% confidence.

One 16 px desktop single-line ellipsis case splits a three-Hangul target into a two-Hangul
fragment and a one-Hangul-plus-ellipsis fragment. Their confidence is at least 99.97% and 99.88%,
their internal gap is 36% to 36.5% of line height, and the surrounding gaps are at least 61% and
56% of line height. The punctuation-inclusive union reproduces both fragments at 99.98%, while
direct and enhanced crops that exclude only the ellipsis independently reproduce the same
three-Hangul core above 99.98%. The bounded merge requires that exact text shape, confidence and
isolated geometry plus all three crop confirmations; it preserves the ellipsis in sentence
context while excluding it from hover geometry.

The exact full comparison advances only `dev-plain-1755` from target failure to context and adds
or removes no ID. Target selection rises to 99.85% and alternative recovery to 96.80%; OCR,
functional context, exact transcription, components, dictionary fidelity, fully correct first
popups, promotions, every negative category, stress, and held-out language do not regress. The
quick diagnostic remains byte-identical at SHA-256
`0639ae2e1a7c55a6cbb2d125cb0082aa50bd67c7d49109ff773f51c903f9d47b`, and the new full
privacy-safe diagnostic SHA-256 is
`674e98bd3d2be173882fcfc65fa42da5f7e7956b0a8ba74f31ab6feed392b7af`.

One 24 px browser single-line case splits a one-Hangul target into an overlapping punctuation
artifact and a compatibility-Jamo-plus-punctuation reading. A bounded recovery requires the
exact privacy-safe 11-segment line profile, the nine-word selection mapping, crop dimensions,
character-category shapes, confidence bands, and normalized geometry. Direct and enhanced
recognition of both the two-segment union and its punctuation-trimmed core must independently
agree above 99.90% and 99.76% confidence. The recovered word retains the terminal punctuation in
sentence context while hover construction excludes it from the target.

The exact full comparison advances only `dev-plain-1190` from target failure to analysis and adds
or removes no ID. Whole-eojeol OCR rises to 98.59%, target selection to 99.90%, functional
context to 91.35%, exact sentence transcription to 73.10%, and alternative recovery to 96.85%;
components, dictionary fidelity, fully correct first popups, promotions, every negative category,
stress, and held-out language do not regress. Automated latency is 236.10 ms median / 362.14 ms
p95. The quick diagnostic remains byte-identical at the SHA-256 above, and the new full
privacy-safe diagnostic SHA-256 is
`31e96e444725dbb9757d30298aa84e5b4bc9766e73c0be3a3f9e40266e4dcf31`.

The newly exposed `dev-plain-1190` component-role case has a complete same-lemma determiner
candidate before a following noun, but it trails a nominal reading by 5.894 score points. The
prenominal promotion ceiling extends from 4.0 to 5.9 only when the candidate is a single bare
component whose complete surface matches the target and KRDict orders an explicit determiner
entry first for that analysis. Particle-bearing and unbacked candidates retain the 4.0 ceiling.
A broader dictionary-backed rule was rejected because it regressed the two particle-bearing
pronoun cases `dev-plain-0909` and `dev-plain-1575`.

The accepted exact comparison removes only `dev-plain-1190` from the component-role failures,
with no addition or stage change. Component accuracy rises to 93.30%, exact KRDict fidelity to
95.60%, and fully correct first popups to 86.10%; OCR, target selection, context, exact
transcription, alternative recovery, quick metrics, stress, held-out language, promotions, and
negative probes do not regress. The full diagnostic SHA-256 is
`d8a9de3d89932601835f214ddd8d7b9c180e72370d316c54d5116f7d13c93e33`.

The next categorical review classifies `dev-plain-0147` and `dev-plain-1391` as Kiwi-analysis
errors, `dev-plain-0220` as an annotation-convention difference, and `dev-plain-1578` as an
equivalent learner interpretation. Removing the synthetic brackets around `dev-plain-0147`
makes Kiwi's complete dependent-noun reading lead, while the wrapped reading trails a complete
noun by 3.076 score points. The accepted wrapper-context promotion therefore extends the
dictionary-backed margin from 1.0 to 3.1 only for a complete noun-to-dependent-noun alternative
that exactly matches the unwrapped contextual leader. The exact full comparison removes
`dev-plain-0147` and the already reviewed bracketed month-unit convention case
`dev-plain-0759`, with no addition or stage change. The quick comparison removes only
`dev-plain-0147`. `dev-plain-1391` has no complete particle candidate to promote, so no
speculative synthesis is added; the other two reviewed cases already provide learner-useful
analyses.

The first-popup reviewer now has the same separately scoped full-tier mode, repeated stable-ID
batch inspection, structure-only output, batch categorical recording, and matching-only migration
for an oracle-corrected corpus. Full reports use the `first_popup_analysis_full` kind and cannot
be mixed with quick reports. The v4.12 history contains 167 decisions: 73 Kiwi-analysis errors,
32 annotation-convention differences, 34 equivalent learner interpretations, 22 corpus-oracle
defects, and six genuinely ambiguous cases. Its SHA-256 is
`76d089614630f196eb4c003382e2879756bf349fda2b5aaa8eb4e7cbdbb9aed5`.
The v4.13 history then contains 112 active decisions; its SHA-256 is
`a6af7603ec91e6a69e080e05866eb1359bd06974e70d0b16edbd91da62a2fdbc`.
The v4.15 history contains 127 decisions: 25 Kiwi-analysis errors, 45 annotation-convention
differences, 38 equivalent learner interpretations, ten corpus-oracle defects, and nine genuinely
ambiguous cases. The current v4.16 matching migration contains all 98 active decisions with no
missing, resolved, or stale ID: two Kiwi-analysis errors, 42 annotation-convention differences,
37 equivalent learner interpretations, eight corpus-oracle defects, and nine genuinely ambiguous
cases. Its SHA-256 is
`144f1390db59d0cdbe823698b96096d01d4258788940aca6aa38e2d2d25634f2`.
The closing audit reclassifies eight provisional Kiwi errors: two truncated-context ambiguities,
two source-tag convention differences, two learner-equivalent noun/proper-noun readings, and two
corpus-oracle defects involving a wrapped standalone particle and contracted-copula fallback. The
other two inspected cases remain Kiwi errors; no speculative analyzer promotion was accepted
because independent contextual and isolated evidence did not support one.

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

One wrapped one-syllable target ranked as a complete noun even though removing only its immediate
wrappers and analyzing the eojeol in isolation independently selected the same existing,
dictionary-backed inflected-predicate candidate. The accepted later-stage promotion requires both
signals, a complete predicate, a complete nominal leader with a different lemma, and a score gap
of at most 4.9. The exact full comparison removes only `dev-plain-0398`, adds or changes no
failure record, and leaves quick diagnostics byte-identical. Main component accuracy rises to
93.55%, exact KRDict fidelity to 95.65%, and first-popup correctness to 86.25%; stress and held-out
language are unchanged, false promotions and negative activations remain zero, and the current
privacy-safe diagnostic SHA-256 is
`62ea67d206b954cda83bec98d9671510fae08422b331a4f3e38c699e4b90f09e`.

One reviewed quoted target followed an explicit object particle but ranked a fragmented
adverb-plus-`hada` reading ahead of the existing complete action verb. The accepted correction
requires the object-particle tag, two dictionary-backed leader components with the exact
adverb-plus-`hada` shape, a single dictionary-backed action-verb alternative reconstructing the
same surface, preserved learner features, no unrepresented word part, and a score gap of at most
6.0. That contextual signature survives immediate-wrapper analysis and prevents the later
isolated-form role tie-breaker from overriding the object evidence. An unrelated object-taking
predicate retains its existing isolated complete-form recovery. The exact full comparison removes
only `dev-plain-0310`, adds or changes no failure record, and leaves quick diagnostics
byte-identical. Main component accuracy rises to 93.60%, exact KRDict fidelity to 95.70%, and
first-popup correctness to 86.30%; false promotions and negative activations remain zero. The
accepted aggregate and privacy-safe diagnostic SHA-256 values are
`b84c079afb7892892f91e09539b2d01f64fcef8279750791edfcea379a0002db` and
`4c243cd584b8df858343b6811f8671a691c8b0534bed401abbaa153b5cb9b389a`.

One reviewed counter was tagged as an ordinary noun because Kiwi attached the separately written
counting form to the preceding word across a real whitespace boundary. The accepted contextual
role correction requires an exact counting eojeol or counting-determiner tag, a gap of at most
three whitespace or punctuation characters, one particle-bearing noun component, and an existing
KRDict dependent-noun entry. It changes only the learner-facing component role and preserves the
dictionary entry order. The focused negative boundaries exclude non-counting determiners and bare
nouns without an attached particle. The exact full comparison removes only `dev-plain-1469`, adds
or changes no failure record, and leaves quick diagnostics byte-identical. Main component accuracy
rises to 93.65%, first-popup correctness to 86.35%, and alternative recovery to 96.90%; exact
KRDict fidelity remains 95.70%, and false promotions and negative activations remain zero. The
accepted aggregate and privacy-safe diagnostic SHA-256 values are
`41ae8e67e80ac4e0d8340e48f4160e13dff663c21121e6afc0141c8ee5d98c3d` and
`bf5986b00330fce9a2023d3bbbaf601192a69bf7d8d8478593f7fb6f7318559f`.

The v4.16 corpus rebuild corrects the two closing-audit oracle defects without mutating prior
evidence. Standalone particles annotated as conjunctions now receive a particle-first oracle group,
and standalone contracted copulas receive a whole-surface `linking word` component backed by
`이다`. Runtime analysis promotes a lower-ranked standalone particle only between separately
wrapped neighboring phrases and within the existing score margin. It recognizes only Kiwi's exact
zero-length `이/VCP` plus target-covering `ETM` contracted-copula shape; a positive-length
copula is a tested negative boundary. The exact v4.15-to-v4.16 diagnostic comparison removes only
`dev-plain-1391` and `dev-plain-1926`, adds or changes no failure record, and leaves quick
diagnostics byte-identical. The v4.16 lock SHA-256 is
`1c5661f511a49c4931214c614b812aedf298edb746e95c951113d9829158aa62`; the accepted full aggregate
report SHA-256 is `70fec242d3ee7f04f743fbb17f8494c98a1c703d16b99ca6397c23ff6efc5943`.

The next bounded OCR batch recovers the two false spaces in `dev-plain-1755`: `2 + 1` and `2 + 2`
pure-Hangul pairs at a narrowly measured relative gap. The profiles require high-confidence part
readings, matching character pitch, and substantially wider neighboring whitespace. Exact-union
recognition must reproduce the merged word at 99.95% or better; neighboring crops still veto the
merge unless their recognized text explicitly retains a separating space. Focused negative tests
cover tighter neighboring whitespace, a strong no-space competitor, and the part-confidence
boundary. The exact 2,000-case comparison removes only `dev-plain-1755|context|`, adds or changes
no diagnostic, and reduces the full diagnostic count from 363 to 362. The affected 16 px,
ellipsis, Malgun Gothic, desktop, and single-line strata improve without any stress, held-out
language, target, analysis, dictionary, promotion, or negative-category regression. The accepted
full report and diagnostic SHA-256 values are the current values above; the quick diagnostic
remains byte-identical.

The following small-text OCR batch adds a production-recognizer-only binarized retry for
two-to-five-syllable pure-Hangul words below 99.8% confidence on detector lines no taller than
14.1 px. Autocontrasted 3x crops are thresholded at 216 and recognized independently with
bilinear, bicubic, and Lanczos resampling. All three readings must agree on a same-length
alternative, and their conservative minimum confidence must be at least 94% and exceed the
original reading. A broader 16 px candidate without the 94% floor was rejected after adding
`dev-plain-0538`, `dev-plain-1729`, and `dev-plain_stress-0157`; it also lowered the 14 px
stratum. The retained boundaries exclude all three regressions.

The exact accepted comparison removes five main context diagnostics
(`dev-plain-0001`, `dev-plain-0801`, `dev-plain-1281`, `dev-plain-1601`, and
`dev-plain-1889`) plus three stress context diagnostics (`dev-plain_stress-0089`,
`dev-plain_stress-0117`, and `dev-plain_stress-0185`), adding or changing none. The full
diagnostic count falls from 362 to 354. The 12 px stratum rises from 98.10% to 98.27% OCR,
84.00% to 86.00% context, 61.60% to 63.20% exact transcription, and 80.80% to 82.80% popup
correctness. Bracket and natural-punctuation strata improve, stress OCR/context/popup results
rise to 94.36% / 68.40% / 64.40%, and held-out language and every safety gate remain unchanged.
The accepted full report and diagnostic SHA-256 values are the current values above.

A bounded follow-up extends the same retry only for exactly two-syllable pure-Hangul words on
detector lines above 14.1 px and no taller than 15.9 px. A broader 17.7 px trial retained the 94%
confidence floor but recovered one main context case while adding six, so it was rejected. The
retained two-tier boundary removes only `dev-plain-0257|context|`, adds or changes no diagnostic,
and reduces the full diagnostic count from 354 to 353. Aggregate functional context and first-popup
correctness rise to 91.70% and 86.80%. The 12 px context/popup stratum rises to 86.40% / 83.20%,
and natural punctuation rises to 87.60% / 83.20%. Quick diagnostics remain byte-identical;
stress, held-out language, target, analysis, dictionary, promotion, and negative-category results
are unchanged. The accepted full report and diagnostic SHA-256 values are the current values above.

The next bounded context recovery handles a high-confidence Hangul prefix fused directly to an
opening straight or curly double quote when a later high-confidence eojeol on the same detector line
contains the matching closer. Both sides of the opening boundary must be Hangul, the quoted start
must contain at least two syllables, and the closer may have at most one attached Hangul syllable.
The 95% confidence floor applies independently to the opening and closing segments, so a weak
punctuation recognition cannot trigger the split.

The exact full comparison removes only `dev-plain-0482|context|`, adds or changes no diagnostic,
and reduces the full diagnostic count from 353 to 352. Whole-eojeol OCR, functional context, exact
sentence transcription, and first-popup correctness rise to 98.63%, 91.75%, 73.40%, and 86.85%.
The 14 px OCR/context/transcription/popup stratum rises to 97.11% / 84.80% / 58.00% / 80.40%, and
the ellipsis stratum rises to 98.73% / 92.80% / 54.40% / 87.60%. Quick diagnostics remain
byte-identical; stress, held-out language, target, analysis, dictionary, promotion, and every
negative category are unchanged. The accepted aggregate and privacy-safe diagnostic SHA-256 values
are `a1a9e0ebc86103eff900ffefecfbbb8813445f934f06d1e01b73d420bccaaebc` and
`cbe375fad893ccf1128aed04a1f512e285e9c8d6f1f9ab7fc123ba354064105d`.

The complete v4.16 development run now shows that aggregate functional context clears 88%, while
main popup correctness and required render strata still block release evidence. Every active popup
case is classified. The only confirmed Kiwi-analysis errors are `dev-plain-0068` and
`dev-plain-0990`; neither has a safe complete candidate or bounded general promotion. The strict
context matching migration retained 170 prior active decisions and exposed one new boundary case;
the recorded decision remains in the ledger, and the latest full audit retains it with seven
additional resolved transcription decisions, leaving 163 active cases with complete review
coverage. Two historical line/sentence
reconstruction cases require punctuation or English text that cannot be inferred from reliable runtime evidence, and the
remaining reviewed boundary cases are too weak, punctuation-dependent, ambiguous, or complex for
another safe general rule. The geometry-clustering, substitution, direct-retry, wrapper-fragment,
and boundary recoveries leave two main target failures, both equal-length wrong-text hits in
matching geometry. A 140-variant direct/enhanced crop sweep produced no independent exact reading
for either substitution. The next development target is required-render and OCR work without
speculative analyzer rules. Thresholds are not frozen, and neither the untouched release split nor
the 500-attempt foreground benchmark has been run. See `docs/RELEASE_BASELINE_2026-08.md` for the
measurement breakdown.

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
