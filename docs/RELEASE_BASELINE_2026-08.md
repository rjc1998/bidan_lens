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

## Plain-v1 schema-v4.15 development follow-up

The current development-only corpus is locked under
`F:\bidan-lens-eval-ud218-v4.15\dev`. It contains 2,000 main, 250 stress, 400 held-out
language, and 200 quick cases and records the `viewport-v3` policy for both renderers. The policy
shifts only targets outside the 1280 by 720 image into a 10 px safe band and removes clipped words
from rendering and expected geometry together. The intermediate v4.10 card-anchoring build is
preserved as rejected evidence because its correction was broader than the viewport defect.

The v4.15 quick tier records 99.23% whole-eojeol OCR, 99.00% target selection, 95.00%
functional context, 75.50% exact sentence transcription, 94.00% component accuracy, 95.00%
exact KRDict fidelity, 90.50% fully correct first popups, 96.50% alternative recovery, and zero
false promotions. The accepted current rerun is 222.70 ms median / 331.76 ms p95. There
are nine analysis, eight context, and two target failures. Aggregate and every negative category are
0.00%, including all 200 near-miss probes, so the quick popup floor and strict negative-activation
gate pass.

The v4.15 lock SHA-256 is
`5c57bdeb06e792960ec8869b0c3914a50170a911f73f1873b25185c011592ba8`.
The aggregate quick report and privacy-safe diagnostic SHA-256 values are
`9e5911fe5caad334129475d6116e6cf3fdd1d585afc73c7d34dd04ab47c90355` and
`de2b5c7ae2a86a501e222ece53b8a922529658ad245ce12c6a59164a9faeba3c`.
Accumulated candidate-builder changes mean v4.9 decisions cannot be mapped to v4.15 by numeric ID
without a fresh review audit.

The complete v4.15 development run against the current OCR and analyzer cleanup records 98.10%
whole-eojeol OCR, 97.10% target selection, 88.05% functional context, 71.10% exact sentence
transcription, 92.35% component accuracy, 94.75% exact KRDict fidelity, 81.95% fully correct first
popups, 94.10% alternative recovery, and zero false promotions across 2,000 main cases. The
accepted rerun is 226.03 ms median / 336.66 ms p95. The privacy-safe stage totals are 58 target,
181 context, and 122 analysis failures. The analysis failures comprise 50 primary lemmas, 62
component roles, four component counts, and six grammar roles; no component-surface failures
remain.

The nonblocking 250-case stress tier records 94.19% OCR, 96.00% target selection, 67.20%
functional context, 93.60% component accuracy, and 63.20% fully correct first popups. The 400-case
held-out language tier is 92.00% overall, 96.00% for auxiliary cases, and the required 88.00%
for multi-lexical cases; direct KRDict conformance remains 100%. Aggregate main negative activation is
0.11%. Blank, English, and near-miss remain at zero, whitespace is four of 1,931 (0.21%), and
punctuation is six of 1,582 (0.38%). The correction, dictionary-conformance, latency, and strict
aggregate/per-category negative gates pass; the primary and exceptional floors do not.

The full aggregate report and privacy-safe diagnostic SHA-256 values are
`c0aa6c3cca7807b0745032bcbf107d7ec78b971bd6622984a02a5de5d1c9dbbe` and
`01d491452b2a241722407e48195d388fc71c249fa0dbd0bf5432666beb5488ac`.

The context reviewer now assigns full-tier reports the distinct `functional_context_full` kind and
supports repeated-ID batch inspection and single-ID categorical recording without scanning every
main image. The current 205-decision review is 88 non-target OCR transcription errors, 68 punctuation
or structured-text cases, 42 missed or merged OCR word boundaries, and seven incorrect line/sentence
reconstructions. The current full diagnostics have 181 active context cases; the v4.15 fail-closed
audit finds every active ID reviewed with no missing decisions and 24 resolved IDs. Cross-lock
carry-forward accepts a prior corpus ID while still requiring the same review scope and every
current stable ID.
The additional
transcription category distinguishes non-target character
substitution or omission from sentence reconstruction and target-span defects. Three reconstruction
cases share a one-character eojeol centered inside a two-character eojeol and repeating its
normalized suffix. The accepted contained-suffix cleanup resolves all three and preserves the
protected unrelated-character regression. The decision report persists no corpus text, recognized
text, definitions, or pixels. Its SHA-256 is
`34e4bc6e5981dfba10f48b2478884c8def1ffb8c626f3307527d3d34a42def30`.
The current full run resolves 45 context IDs from the earlier report without adding a new context
failure: three from the contained-suffix cleanup, 11 from the confirmed leading-sliver cleanup,
two from the exact-confirmed suffix-overlap cleanup, and two from permitting one-pixel overlap up
to 7.5% of a small line under the same exact combined-recognition duplicate profile. That profile
now also accepts an ASCII digit as the leading artifact only when recognizing the union at 99% or
better omits it and exactly reproduces the following Hangul word. This resolves three additional
full-tier cases without a new failure and leaves the quick diagnostics byte-identical.

The popup reviewer now supports a separately scoped `first_popup_analysis_full` report, repeated
stable-ID batch inspection, structure-only output, validated same-category batch recording, and a
matching-only migration mode for corrected corpora. The v4.12 history has 167 decisions: 73
Kiwi-analysis errors, 32 annotation-convention differences, 34 equivalent learner interpretations,
22 corpus-oracle defects, and six genuinely ambiguous cases. Its SHA-256 is
`76d089614630f196eb4c003382e2879756bf349fda2b5aaa8eb4e7cbdbb9aed5`.
The v4.13 history then has 112 active decisions and SHA-256
`a6af7603ec91e6a69e080e05866eb1359bd06974e70d0b16edbd91da62a2fdbc`.
The current v4.15 report contains 120 decisions: 28 Kiwi-analysis errors, 42 annotation-
convention differences, 35 equivalent learner interpretations, eight oracle defects, and seven
genuinely ambiguous cases. Its fail-closed audit finds all 120 active cases reviewed with no
missing, resolved, or stale ID. Its SHA-256 is
`d6db4974d39f20806866f321aa767bfe927d100b78ed25a90d62b089c66ed8b6`.

The second full-tier batch supports one bounded morphology correction. When Kiwi emits
`noun + 화/XSN + 하/되/XSV`, the analyzer now forms a single action-verb component only if KRDict
contains the exact complete `-화하다` or `-화되다` lemma. This resolves four main primary-lemma
failures and adds none. The accepted quick diagnostic remains byte-identical, while stress and
held-out language results are unchanged.

The third full-tier review batch supports four more narrowly bounded corrections. Exact
dictionary-backed `adverb + 하/XSV|XSA` derivations form one learner component and retain a
contextual helping-verb role after a connective. Exact standalone object particles `을` and `를`
can override a false noun analysis, while ambiguous particle/noun surfaces remain unchanged. An
internal `화/XSN` extends a preceding noun only when the combined noun is dictionary-backed and no
copula follows. A zero-length Kiwi insertion is ignored only when it duplicates the preceding
component surface, preserving nonduplicate reported-speech ellipsis. The accepted changes recover
seven net main popups, improve component accuracy by 0.40 points and KRDict fidelity by 0.45 points,
and recover two held-out auxiliary cases. Quick diagnostics remain byte-identical, multi-lexical
accuracy remains 86.50%, and no upstream or negative-pointer metric changes. Broader internal-
suffix attachment and unconditional zero-length suppression were rejected after held-out or
failure-stage regressions.

The fourth full-tier batch classifies 20 component-role cases as nine Kiwi-analysis errors, seven
annotation-convention differences, one corpus-oracle defect, and three genuinely ambiguous
fragments. Accepted changes recognize suffixed reported-speech connectives and nominal `-라도` as
non-auxiliary context, preserve exact `-기도 하다` helping-verb evidence after isolated-eojeol
reranking, and recognize contracted `-게 되다` forms through the already-matched `되다` lemma. A
paired wrapper may supply a missing role-only candidate only when unwrapped context preserves the
complete lemma and boundaries and the wrapped analysis has no alternative. Dictionary order may
promote an adverb within the existing score margin but cannot demote an existing adverb. The final
full rerun removes 12 main failure IDs, adds none, and changes no stage. Quick diagnostics remain
byte-identical, stress is unchanged, held-out auxiliary accuracy gains six cases, and held-out
multi-lexical accuracy remains 86.50%. A symmetric adverb/noun experiment was rejected after it
introduced two main role regressions.

The fifth full-tier batch classifies 20 more component-role cases as 12 Kiwi-analysis errors,
four annotation-convention differences, three equivalent learner interpretations, and one oracle
defect. Paired punctuation no longer masks the existing `-게 되다` cue. A leading single-component
helping-verb reading without an auxiliary KRDict sense or connective context may yield only to its
immediate same-lemma, same-boundary, dictionary-backed lexical alternative within 7.1 points.
Dictionary order cannot demote an already-leading determiner or adverb, and numeric-plus-helping
decompositions cannot displace a complete dictionary-backed verb. The final rerun removes five
main role failures, adds none, and raises component accuracy to 90.20%, KRDict fidelity to 93.50%,
and popup correctness to 78.65%. Quick diagnostics remain byte-identical, while main and stress
alternative recovery, all stress results, held-out language, upstream, promotion, and negative
metrics remain unchanged.

The sixth full-tier batch classifies 20 more component-role cases as 14 Kiwi-analysis errors,
three annotation-convention differences, two oracle defects, and one equivalent learner
interpretation. The accepted corrections add separately bounded, one-way dependent-noun
promotion; identical-boundary dictionary-backed action/descriptive disambiguation; locative
descriptive `있다`; contracted `-아야/-어야/-여야만 하다` auxiliary context; and synthesis from
unwrapped paired-punctuation context only when every wrapped candidate is learner-identical.
The final v7 cross-validation removes seven main component-role failures and one grammar-role
failure, adds none, and improves component accuracy to 90.55%, KRDict fidelity to 93.65%, popup
correctness to 79.05%, and alternative recovery to 93.15%. Quick diagnostics remain byte-identical,
stress remains unchanged, and held-out language improves to 90.50% overall and 87.00%
multi-lexical while auxiliary accuracy remains 94.00%. A broader post-context dictionary-role
reapplication was rejected after adding five main role failures. At that point the privacy-safe
review target was 61 cases: 11 component roles, 24 primary lemmas, 12 component surfaces, seven
component counts, and seven grammar-role cases.

The seventh batch completes the remaining 11 component-role reviews as six Kiwi-analysis errors,
three annotation-convention differences, and two oracle defects. Bounded accepted rules reject an
unsupported dependent-noun reading, recognize dictionary-supported ordinary `식` and attached
compound-terminal nouns in their reviewed contexts, and expose a predicate with only auxiliary
KRDict entries as helping. Isolated-eojeol evidence still supports complete multi-component
recovery in an ordinary sentence, while role-only reranking requires a punctuation or fragment
boundary. The independent GSD oracle builder also honors single-token `ADV` over a broad nominal
XPOS role on the next corpus rebuild. The v8 cross-validation removes exactly six main role
failures, adds none, and raises main component accuracy to 90.70% and popup correctness to 79.35%.
Quick diagnostics remain byte-identical, stress remains unchanged, and held-out language rises to
91.25% overall and 95.50% auxiliary while multi-lexical remains 87.00%. Component-role review is
complete; at that point the next review target was the 24 unreviewed primary-lemma cases.

The eighth batch completes review of those 24 primary-lemma cases as 13 Kiwi-analysis errors,
nine equivalent learner interpretations, and two annotation-convention differences. The reviewer
can now record one validated category for a repeated set of stable IDs in one model initialization,
while persisting only IDs, stages, and categories. Accepted analyzer rules recover complete
dictionary-backed predicates without dropping tense or other grammar features, preserve a complete
lexical adverb over a nominal split, require a connective before promoting an isolated auxiliary
decomposition, and recognize a close adnominal predicate before a dependent noun. Boundary-scoped
isolated-predicate recovery excludes single-syllable surfaces and determiners, and paired-wrapper
synthesis requires repeated same-lemma nominal readings. Independently supported approximation and
extent suffixes recover their dictionary-backed noun stems while retaining a dependent-noun role
when KRDict supplies one. Two reviewed Kiwi errors remain unchanged because a contracted-copula
fallback has an inconsistent oracle role and a semantic verb homograph has no safe general runtime
signal. The accepted primary-lemma v2 comparison removes 15 primary-lemma failures, one
component-role failure, and one component-count failure, and adds none. Quick diagnostics remain
byte-identical and stress popup correctness is unchanged. Main component accuracy rises to 91.60%,
KRDict fidelity to 94.50%, popup correctness to 80.20%, and alternative recovery to 93.40%; held-out
language rises to 91.50% overall and 96.00% auxiliary while multi-lexical remains 87.00%.
Primary-lemma and component-role review is complete. The immediate target is the 12 unreviewed
component-surface cases, followed by six component-count and seven grammar-role cases.

The ninth batch classifies all 12 component-surface cases as corpus-oracle defects. Their pinned
KAIST records use the `pad` demonstrative-adjective tag, which the builder had omitted from
descriptive-predicate component mapping and therefore replaced with a whole-form `conjugated`
fallback. The v4.13 builder maps `pad` consistently for roles, dictionary ordering, component
lemmas, and primary lemmas. The rebuild removes all 12 main and two stress component-surface
failures. It also activates the previously accepted GSD single-token `ADV` correction, resolving
two main role failures while exposing six main and two stress role disagreements. The exact main
delta is 14 removed analysis failures and six newly exposed role failures, with no surface failure
remaining. Quick diagnostics are byte-identical; main component accuracy rises to 92.00%, popup
correctness to 80.60%, and alternative recovery to 93.90%. Stress popup correctness and held-out
language remain unchanged. At that point the next target was the six newly exposed component-role
cases, followed by six component-count and seven grammar-role cases.

The tenth batch reviews the six component-role cases exposed by the GSD `ADV` correction. Five
are annotation-convention differences: published UPOS and `advmod` describe sentence function,
while XPOS, Kiwi, and every local KRDict headword entry identify a lexical noun. Those useful noun
readings remain unchanged. The remaining case is an oracle defect whose nominal list item has
`ADV` UPOS but `conj` dependency and noun XPOS. The v4.14 oracle therefore requires `advmod`
before a single nominal XPOS component inherits the adverb role. The exact comparison removes that
one main component-role failure and adds none. Quick diagnostics are byte-identical, stress and
held-out language remain unchanged, and main component accuracy rises to 92.10%, popup correctness
to 80.65%, and alternative recovery to 94.00%. Component-role review is complete; the next target
is six component-count cases followed by seven grammar-role cases.

The eleventh batch completes those six component-count reviews as two Kiwi-analysis errors, two
annotation-convention differences, and two equivalent learner interpretations. Isolated-eojeol
evidence no longer promotes a decomposition containing a repeated lexical component. A separate
3.25-point rule promotes a final one-syllable verb-ending interpretation only when it removes one
terminal adverb component while preserving the same surface, primary lemma, complete predicate
prefix, and dictionary-backed inflected analysis. The exact full comparison removes two
component-count failures, adds no failure or stage change, and leaves quick diagnostics
byte-identical. Main component accuracy rises to 92.20%, KRDict fidelity to 94.70%, and popup
correctness to 80.75%; alternative recovery, stress, held-out language, upstream, promotion, and
negative-pointer metrics remain unchanged. Component-count review is complete; the next target is
the seven unreviewed grammar-role cases.

The twelfth batch completes the seven grammar-role reviews as four annotation-convention
differences, two Kiwi-analysis errors, and one corpus-oracle defect. The four published KAIST
`jcr` reported-speech tags are learner-equivalent connective endings and do not support a
runtime particle promotion. The oracle now maps an ADP token with particle XPOS to its first
annotated particle morpheme and orders particle homographs first. Under bounded structural checks,
wrapper-free sentence context can recover a dictionary-backed standalone particle or a copular
adnominal before a dependent noun. The exact v4.15 comparison removes two main grammar-role
failures and adds none. Component accuracy rises to 92.35%, KRDict fidelity to 94.75%, popup
correctness to 80.85%, and alternative recovery to 94.10%; quick accuracy is unchanged, false
promotions stay zero, and stress, held-out language, upstream, and negative-pointer metrics do not
regress. The corrected particle case remains genuinely ambiguous after OCR omits its adjacent
percent sign. Grammar-role review is complete; at that point the next development target was the
nine full-tier punctuation activations.

Geometry-only review finds three punctuation activations in single-Hangul OCR boxes whose width is
1.53 to 1.99 times their height after trailing punctuation is swallowed. Their valid target points
remain in the left interior while the punctuation probes occupy the rightmost 21.7% to 23.4% of the
abnormally wide box. Hit testing now uses a 25% right inset, instead of 20%, only for a one-glyph
Hangul eojeol at least 1.5 times as wide as it is tall. Other edges, normal glyphs, and multi-glyph
eojeols are unchanged. The exact full diagnostic comparison removes only the three
punctuation-only IDs, with no addition or stage change. Punctuation activation falls from nine of
1,582 (0.57%) to six (0.38%), while target selection remains 97.10% and quick target selection
remains 99.00%. Every OCR, context, popup, stress, language, promotion, and quick negative result
is unchanged, so the aggregate/per-category negative gate now passes. The next target is the
held-out multi-lexical tier.

The thirteenth analyzer batch classifies all 26 v4.15 multi-lexical disagreements as three
Kiwi-analysis errors, five annotation-convention differences, 17 equivalent learner
interpretations, and one genuinely ambiguous case. Its ID-only report contains no corpus text,
expected or analyzed values, dictionary data, or pixels. Review found that the existing
multi-component promotion could replace an already complete dictionary-backed two-part leader
with a lower-ranked three-part fragmentation. The accepted guard limits that promotion to an
incomplete leader. It resolves exactly two reviewed Kiwi errors, leaves 24 active cases with no
missing or stale decisions, and raises multi-lexical accuracy from 87.00% to 88.00% and overall
held-out language from 91.50% to 92.00%. Auxiliary stays at 96.00%, stress is unchanged, and the
full main and quick diagnostics remain byte-identical to their accepted baselines. The language
review SHA-256 is
`2534d28a3bf9650f8a61120a6577e7e3397fc5a3e87f1c8b8544772d38913765`.
At that point the next target was the 205 active full-tier context disagreements, followed by
analysis failures whose target and context were already correct.

The fourteenth batch reviews the seven active line/sentence reconstruction cases with geometry-only
and segmentation-only diagnostics. One case contains an entire one-eojeol detector line whose
exact text overlaps an existing same-row eojeol by at least 80% on both axes. The accepted cleanup
keeps the duplicate geometry but remaps it to the existing sentence span instead of appending
duplicate text. The exact full diagnostic comparison removes only `dev-plain-1062`, with
no added or changed failure record. Functional context and first-popup correctness each rise by
0.05 points to 86.90% and 80.90%; quick diagnostics are byte-identical, and stress, language,
upstream, promotion, and every negative-pointer result remain unchanged. Six reviewed
reconstruction cases and 204 total context disagreements remain.

The fifteenth batch identifies two more reviewed reconstruction cases with the same independent
geometry profile: a below-60%-confidence one-character box fully contained in a word of at least
three characters that is at least 99% confident, with the inner width no more than 16% of the
containing width. The accepted cleanup removes only `dev-plain-1526` and
`dev-plain-1671` from the full diagnostics. No failure record is added or changed;
quick diagnostics remain byte-identical. Functional context rises to 87.00%, exact sentence
transcription to 70.15%, and first-popup correctness to 81.00%. The full context audit remains
complete with 202 active cases, three resolved IDs, no missing decisions, and four reviewed
line/sentence reconstruction cases remaining.

The sixteenth batch reviews a below-50%-confidence one-character false recognition nearly aligned
with the leading edge of a 99.9%-confident two-character word. The accepted cleanup additionally
requires leading edges within one pixel, at least 80% vertical overlap, and an artifact no wider
than 25% of the word. The exact full comparison removes only `dev-plain-1398` with
no added or changed record. Functional context rises to 87.05%, exact sentence transcription to
70.20%, and first-popup correctness to 81.05%; quick diagnostics remain byte-identical. The
fail-closed audit has 201 active cases, four resolved IDs, no missing decisions, and three reviewed
line/sentence reconstruction cases remaining.

The seventeenth batch resolves one paired structured-overlap artifact. The left detector fragment
ends with an identifier followed by one copied Hangul syllable; the overlapping right fragment
starts with the identifier's repeated final digit and the complete word beginning with that
syllable. The accepted merge requires exact suffix/prefix agreement, at least 99% confidence for
both Hangul boxes, leading edges within one pixel, and at least 80% vertical overlap. The exact
full comparison removes only `dev-plain-1316`. OCR rounds to 98.00%, functional
context rises to 87.10%, exact sentence transcription to 70.25%, and first-popup correctness to
81.10%; quick diagnostics remain byte-identical. The audit has 200 active cases, five resolved
IDs, and no missing decisions. The two remaining reconstruction cases lack independent runtime
evidence for their intended punctuation or English text and are left unchanged.

The eighteenth batch begins the 42-case missed-or-merged boundary review. Ten stable-target cases
insert exactly one space inside an expected word; two are defensible spacing interpretations, and
the wider component-length profiles also fit legitimate Korean word boundaries. One case has a
distinct isolated 1+1-syllable profile: zero measured gap, gaps of at least 50% and 44% of line
height on either side, compatible character pitch, strong component confidence, and at least
99.99% combined recognition of the exact concatenation. The accepted recovery removes only
`dev-plain-1420` from full diagnostics, with no added or changed record. Functional
context rises to 87.15%, exact sentence transcription to 70.30%, and first-popup correctness to
81.15%; quick diagnostics remain byte-identical. The audit has 199 active context cases, six
resolved IDs, 41 active boundary cases, and no missing decisions.

The nineteenth batch groups all 41 active boundary cases by component length and privacy-safe
segmentation signals. Three reviewed false splits have a shallow 2+1-syllable overlap; the
accepted internal profile additionally requires clear gaps on both sides, compatible pitch,
at least 99.7% confidence for the two-syllable fragment, and at least 99.97% exact combined
recognition. It removes only `dev-plain-1210`, with no added or changed diagnostic record.
Functional context rises to 87.20%, exact sentence transcription to 70.35%, and first-popup
correctness to 81.20%; quick diagnostics remain byte-identical. The audit has 198 active context
cases, seven resolved IDs, 40 active boundary cases, and no missing decisions.

The twentieth batch reviews both remaining internal 1+4-syllable overlap cases. The accepted
profile requires a one-pixel overlap between positive neighboring gaps, compatible pitch, at
least 99.75% confidence for the four-syllable fragment, and at least 99.75% exact combined
recognition. It removes only `dev-plain-0873` and `dev-plain-1421`, with no added
or changed diagnostic record. Whole-eojeol OCR rises to 98.02%, functional context to 87.30%,
exact sentence transcription to 70.45%, and first-popup correctness to 81.30%; quick diagnostics
remain byte-identical. The audit has 196 active context cases, nine resolved IDs, 38 active
boundary cases, and no missing decisions.

The twenty-first batch reviews all four active 1+2-syllable false splits. Three have an overlapping
or touching neighbor and remain unchanged. The accepted isolated profile requires wider gaps on
both sides, at least 99.88% and 99.98% confidence for the one- and two-syllable fragments
respectively, compatible pitch, and at least 99.99% exact combined recognition. It removes only
`dev-plain-1150`, with no added or changed
diagnostic record. Functional context rises to 87.35%, exact sentence transcription to 70.50%,
and first-popup correctness to 81.35%; quick diagnostics remain byte-identical. The audit has
195 active context cases, ten resolved IDs, 37 active boundary cases, and no missing decisions.

The twenty-second batch reviews the two remaining 2+1-syllable false splits. The internal case
already matched the accepted exact-recognition profile but missed its pitch threshold only through
floating-point roundoff, so the comparison now tolerates one nanounit. The line-initial case uses
a separate profile requiring a 5.5% to 6% shallow overlap, a following gap of at least 17% of line
height, compatible pitch, at least 99.87% and 97.9% fragment confidence, and at least 99.96% exact
combined recognition. The full diagnostic comparison removes only `dev-plain-0141` and
`dev-plain-0969`, with no addition or stage change. Whole-eojeol OCR rises to
98.03%, functional context to 87.45%, exact sentence transcription to 70.60%, and first-popup
correctness to 81.45%. Quick diagnostics remove only `dev-plain-0141`; the quick
context and popup metrics rise to 93.50% and 89.50%. The audit has 193 active context cases, 12
resolved IDs, 35 active boundary cases, and no missing decisions.

The twenty-third batch reviews four repeated 3+3-syllable missed spaces. A 0.01 CTC-space probe,
used only on an all-Hangul six-syllable word already recognized at 99.4% or better, must produce
exactly two three-syllable crops separated by 28% to 35% of line height. Their pitch must be
compatible, each part must be at least 99.3% confident, and concatenating them must exactly
reproduce the original text. Legitimate six-syllable controls in the same lines remain unsplit.
The full comparison removes only `dev-plain-0098`, `dev-plain-1297`,
and `dev-plain-1617` from context. The first two move to primary-lemma failures;
the third becomes fully correct. The fourth case retains a distinct 1+1 merge whose 0.005 CTC
signal is too weak for a general rule. No unrelated ID changes. Full OCR, context, exact
transcription, and popup correctness rise to 98.06%, 87.60%, 70.75%, and 81.50%. The quick tier
moves only `dev-plain-0098` from context to primary lemma, raising OCR, context, and exact
transcription to 99.10%, 94.00%, and 74.50%. The audit has 190 active context cases, 15 resolved
IDs, 32 active boundary cases, and no missing decisions.

The twenty-fourth batch reviews three active 2+2-syllable false splits. The accepted relative-gap
profile requires pure Hangul two-syllable fragments, at least 99.6% confidence for both, a gap of
15% to 24% of line height, neighboring boundaries at least ten percentage points wider, at least
95% compatible character pitch, and 99.6% exact recognition of their union. A provisional 26%
cap regressed the legitimate space in `dev-plain-0922`; that candidate was rejected and the
final spacing is regression-tested. The full comparison removes only `dev-plain-0734` and
`dev-plain-1995`, with no addition or stage change. `dev-plain-1673` remains active because
its separate bracket-attached boundary lacks independent recovery evidence. Functional context,
exact transcription, and popup correctness rise to 87.70%, 70.80%, and 81.60%. Quick diagnostics
remain byte-identical. The audit has 188 active context cases, 17 resolved IDs, 30 active boundary
cases, and no missing decisions.

The twenty-fifth batch groups all 30 active boundary cases by their minimal expected-to-actual
token-length transformation. Two repeated 3+5-to-8 merges have a distinct exact-confirmation
profile: a 0.02 CTC-space probe returns exactly three- and five-syllable Hangul parts separated
by 30% to 33% of line height, their pitch agrees within 3%, both parts are at least 99.88%
confident, and concatenating them exactly reproduces the original token. The full comparison
removes only `dev-plain-0130` and `dev-plain-0499`, with no addition or stage change. The
2+4-to-6 group remains unchanged because its cases mix structured ASCII, defensible compound
spacing, and a low-confidence proper noun. Full OCR, context, exact transcription, and popup
correctness rise to 98.08%, 87.80%, 70.90%, and 81.70%. The quick tier removes only
`dev-plain-0130` and rises to 99.19% OCR, 94.50% context, 75.00% exact transcription, and
90.00% popup correctness. The audit has 186 active context cases, 19 resolved IDs, 28 active
boundary cases, and no missing decisions.

The twenty-sixth batch reviews both remaining 5-to-3+2 splits. The defensible spaced auxiliary
interpretation remains unchanged. The accepted genuine-split profile requires fragment confidence
of at least 99.97% and 99.98%, a gap of 10% to 10.5% of line height, a preceding boundary of at
least 25%, a following shallow overlap of at most 5.5%, pitch agreement within 2%, and exact union
recognition at 99.98% or better. The full comparison removes only `dev-plain-1370`, with no
addition or stage change; `dev-plain-1353` remains as the spacing control. Functional context,
exact transcription, and popup correctness rise to 87.85%, 70.95%, and 81.75%. Quick diagnostics
remain byte-identical. The audit has 185 active context cases, 20 resolved IDs, 27 active boundary
cases, and no missing decisions.

The twenty-seventh batch reviews the three repeated 3-to-1+2 false splits as separate geometries.
The line-initial profile requires fragment confidence of at least 99.92% and 99.86%, a gap of
36% to 36.5% of line height, a following boundary of 61% to 62.5%, compatible pitch, and exact
union recognition at 99.975% or better. The internal touching-following profile requires fragment
confidence of at least 99.99% and 99.93%, a gap of 6% to 6.5%, a preceding boundary of at least
37%, a following boundary within 0.5% of touching, compatible pitch, and the same union floor.
Both reject recovery when the second fragment joined to its following neighbor recognizes at 90%
confidence or better; their verification crops normalize subpixel coordinates before integer
rounding. The low-confidence, overlapping control remains unchanged. The full diagnostic
comparison removes only `dev-plain-0155` and `dev-plain-1185`, with no new or changed record;
the quick comparison removes only `dev-plain-0155`. Full OCR, context, exact transcription, and
popup correctness rise to 98.09%, 87.95%, 71.05%, and 81.85%. Quick OCR, context, exact
transcription, and popup correctness rise to 99.23%, 95.00%, 75.50%, and 90.50%. The audit has
183 active context cases, 22 resolved IDs, 25 active boundary cases, and no missing decisions.

The twenty-eighth batch reviews the three repeated 6-to-3+3 false splits. Every candidate union
exactly reproduces the expected six-syllable Hangul token. The line-initial profile requires
fragment confidence of at least 99.65% and 99.99%, a gap of 26% to 26.5% of line height, a
following boundary of 54% to 55%, pitch agreement within 4%, and exact union recognition at
99.93% or better. The isolated internal profile requires fragment confidence of at least 99.81%
and 99.68%, a gap of 35% to 36.5%, a preceding boundary of at least 61%, a following boundary
of at least 44%, pitch agreement within 2%, and exact union recognition at 99.83% or better.
Both require pure Hangul and reject recovery when either available adjacent union reaches 99.5%
confidence. The full diagnostic comparison removes only `dev-plain-1272` and
`dev-plain-1280`, with no new or changed record. `dev-plain-0475` remains active because
separate punctuation, transcription, and split errors remain. Quick correctness and failure
records are unchanged. Full OCR, context, exact transcription, and popup correctness rise to
98.10%, 88.05%, 71.10%, and 81.95%. The audit has 181 active context cases, 24 resolved IDs,
23 active boundary cases, and no missing decisions.

The v4.12 corpus rebuild itself was limited to negative-probe construction. Geometry-only review
showed that the two v4.11 near-miss failures pointed inside real eojeols on adjacent lines. The builder now selects
the first lower, upper, right, or left adjacent point that is inside the viewport and outside
every oracle eojeol. The remaining substantive failures preserve their v4.11 IDs and stages.

The separately scoped v4.12 quick audits are complete. Their fail-closed carry-forward workflow copies only current
stable IDs from a prior categorical report, requires every current ID, additionally requires
exact popup failure stages, and refuses to overwrite an existing report. The quick popup audit has
20 decisions: seven Kiwi-analysis errors, five annotation-convention differences, seven
equivalent learner interpretations, and one genuinely ambiguous case. The context report retains
16 decisions: seven incorrect line/sentence reconstructions, four missed or merged OCR word
boundaries, and five punctuation or structured-text cases. Its current audit has 13 active cases
and three resolved reconstruction IDs. Neither report persists sentence text, recognized text,
definitions, or pixels, and both audits have no missing or stale IDs.

The current quick popup audit has eight active cases and 12 resolved IDs. Reported-speech
connectives are now passed into learner-role construction, preventing a following lexical verb
from being relabeled as a helping verb solely because its headword also has an auxiliary sense.
This improved component accuracy, KRDict fidelity, popup correctness, and alternative recovery
without changing OCR, target selection, context, promotions, or negative probes.

The grammatical `-게 되다` construction now prefers a same-lemma, same-boundary, dictionary-backed
helping-verb candidate within a separate 10-point score margin. It runs after isolated-role
corroboration so an eojeol analyzed alone cannot override sentence-level grammar. This resolves
two main component-role cases, one stress case, and one held-out auxiliary case without changing
the accepted quick failure IDs or any upstream, alternative, promotion, or negative metric.

For candidates with identical lemmas and component boundaries, a nominal-role alternative may
be promoted within a 2.5-point score margin only when KRDict's default homograph order prefers
each differing role. This resolved two noun/pronoun/determiner cases and one proper/common-noun
case without changing KRDict fidelity, OCR, target selection, context, promotions, or negative
probes.

For otherwise identical verb-role candidates, an isolated eojeol analysis may corroborate a
lower-ranked interpretation within a 2-point contextual score margin. The rule is limited to
action, descriptive, and helping roles. It resolved two additional component-role cases without
changing OCR, target selection, context, alternatives, promotions, or negative probes.

The complete multi-component promotion margin is now 2 points, matching complete inflected-word
recovery. It resolved one lexicalized-verb versus main-plus-helping-verb interpretation while
retaining the requirements for multiple dictionary-backed components and particle preservation.

A separately bounded rule for inflected targets now uses isolated analysis to corroborate a
contextually more distant main-plus-helping-verb structure. It requires multiple fully
dictionary-backed components, a helping verb, complete word-part representation, particle
preservation, and a target that is not already an exact dictionary base form. It resolved four
equivalent-interpretation cases with no quick-tier regression.

An isolated fallback may now recover an equal-count dictionary-backed lemma when it exposes a
particle or verb ending and its only unrepresented word part is the contractible copula. This
resolved one dependent-noun contraction while retaining the general derivational-word-part guard.

A review-supported runtime rule removes a duplicate one-character eojeol only when it is fully
contained in a longer eojeol, matches one of that eojeol's characters, and has normal
single-character pitch. It resolved two context cases without introducing a failure and accounts
for the accepted metric changes above. The popup-review and context-review SHA-256 values are
`97176c303e1fd53671f2ce24cd5cf939066287c7744c4cbbcde7656384abb1be` and
`d82596f8f014bcbbb740db3b531632cfa0707da5ccff91a52c0e775ee4ae5d41`.

An additional exact-confirmation rule removes a low-confidence one-character Hangul sliver only
when it touches or slightly overlaps a following Hangul word, is no wider than 90% of that word's
character pitch, and a recognition of their union at 99% confidence or better exactly reproduces
the following word. The specific exact-confirmed triplet recovery runs first to preserve a real
final syllable. This resolves 11 more full-tier context cases with no new context failure while
preserving the accepted quick-tier metrics.

A separate exact-confirmation rule merges a slightly overlapping pair only when its surfaces
share exactly one boundary syllable, one side is below 80% confidence, the other is at least 95%,
and recognizing their union at 99.8% confidence or better exactly reproduces the deduplicated
surface. It resolves two reviewed full-tier reconstruction cases without a new context failure or
any quick-tier metric change.

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
