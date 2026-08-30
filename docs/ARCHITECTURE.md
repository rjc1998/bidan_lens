# Architecture

BiDan Lens keeps the interactive shell separate from language processing so every
stage can be tested and replaced independently.

```text
pointer + in-memory screen crop
              |
              v
Paddle detector -> CTC-guided eojeol crops -> Paddle Korean recognizer -> OCR document
                                                  |
                                        whole-eojeol hit test
                                                  |
                              sentence + exact target character span
                                                  |
                              Kiwi candidate morphology analyses
                                                  |
                              KRDict SQLite exact lemma lookup
                                                  |
                        ranked learner-friendly popup candidates
```

## Boundaries

- `ocr` owns pixel preprocessing, local ONNX inference, line text, and geometry.
- `pipeline.hit_test` selects a complete whitespace-delimited eojeol only when the pointer
  is inside a conservative inner region and over Hangul glyph geometry. Punctuation, ASCII
  context, eojeol edges, and blank space do not create targets. A single-Hangul box whose width
  is at least 1.5 times its height uses a slightly larger right inset to reject swallowed
  trailing punctuation without changing its left or vertical target area.
- `analysis` receives the full OCR line plus the selected span. Raw morphology tags are
  internal and are translated into ordered lexical components with surface, lemma, and a
  beginner-facing contextual role. Auxiliary tags prioritize KRDict helping-verb/helping-
    adjective groups while retaining other homographs in source order. The adapter examines ten
    Kiwi analyses while exposing at most five popup candidates. A richer multi-component analysis
    may lead only when every component has a local definition, its adjusted Kiwi score is within
    2 points of the leader, it does not displace a contextual auxiliary, and the leader is not
    already a complete dictionary-backed multi-component analysis. A complete dictionary-backed
    multi-syllable inflected predicate is likewise not replaced by a split containing only
    non-auxiliary verbs; one-syllable predicates and main-plus-helping-verb analyses retain the
    established ambiguity handling. A two-syllable proper-noun leader may yield to an
    already-present one-syllable noun plus one-syllable particle analysis only within 3.2 score
    points, with dictionary support for the noun, a centrally known or exact KRDict particle, and
    exact morpheme-boundary agreement. When the first Kiwi
    analysis has no local definition, a known particle suffix may produce a promoted candidate
    only when its remaining stem has a local dictionary entry; an already segmented sequence of
    dictionary-backed nouns may also recover a missing particle feature when its remaining suffix
    is a known particle or an exact KRDict particle entry. This does not change the candidate's
    lemma or definitions. If that particle recovery still leaves an undefined leader, an
    isolated-eojeol analysis may lead only when it supplies more lexical components, every
    component has a local definition, and no derivational word part is left unrepresented.
    A close inflected-verb alternative may outrank a noun homograph after a particle; intervening
    punctuation is ignored for that context check. A plain connective ending and the
    demonstrative-adverb pattern do not alone turn lexical `hada` into a helping verb; explicit
    obligative context retains its dedicated promotion. When immediate punctuation
    separates a target from the following nominal, a close same-lemma determiner interpretation
    may lead without changing the displayed context. Dictionary-backed noun prefixes can be restored to the
    following lexical component, and a terminal noun suffix can extend that component only in
    conservative end/particle contexts; plural `들`, between-noun suffixes, and copular
    contexts are not rewritten.
    Adverbs, conjunctive adverbs, determiners, and negative copulas are exposed as conservative
    learner components with matching role-first dictionary lookup.
- `dictionary` compiles a versioned source export into a read-only runtime database.
  Exact headwords take precedence over aliases; aliases remain a fallback.
  `DictionarySourceAdapter` is the seam for future sources.
- `gui` owns display and input only. It never interprets Korean itself.

The detector returns line regions at up to 1280 px input resolution. The recognizer uses
its dynamic-width CTC space probabilities to identify candidate eojeol crops, tightens
those crops against foreground pixels, and splits a missed CTC boundary only at an unusually
wide completely blank visual gap. It recognizes the resulting crops independently and
rebuilds the containing line with per-character and per-eojeol geometry. High-confidence
numeric tokens and uppercase abbreviations remain in sentence context, as do complete
`K-YYYY/vN` identifiers at a conservative lower confidence threshold. Adjacent `K` and
`-YYYY/vN` fragments may be rejoined. Collinear detector fragments are reconstructed in
left-to-right reading order, and physically overlapping duplicate text is removed only after
edge-punctuation normalization. When every eojeol in a same-row detector fragment exactly matches
an existing eojeol with at least 80% horizontal and vertical overlap, its geometry is retained but
remapped to the existing sentence span instead of appending duplicate text. These context tokens
never become hover targets.
An unrelated one-character fragment can also be removed with sentence-span repair when its box is
fully contained within a word of at least three characters, its confidence is below 60%, the
containing word is at least 99% confident, its width is no more than 16% of the containing word,
and their vertical overlap is at least 80%.
A one-character leading artifact over a two-character word has a separate stricter profile: its
confidence must be below 50%, the word must be at least 99.9% confident, their leading edges must
be within one pixel, the artifact must be no wider than 25% of the word, and vertical overlap must
be at least 80%.
Overlapping detector fragments can also remove a paired structured-text artifact only when the
left fragment ends with an identifier followed by one copied Hangul syllable, the right fragment
starts with the identifier's repeated final digit and a complete word beginning with that
syllable, both Hangul boxes are at least 99% confident, and their leading-edge and vertical
geometry agree. Neither artifact is removed independently.
At line end, a structured false split consisting of two ASCII decimal digits and one Hangul
syllable followed by one Hangul syllable has a separate recovery profile. Fragment confidence
must reach 99.61% and 99.96%, the gap must be 35% to 35.5% of line height, the preceding boundary
at least 62%, and pitch agreement within 12%. Subpixel verification coordinates are normalized;
the union must exactly reproduce the concatenation at 99.92% confidence or better, and a preceding
adjacent union at 99% preserves the original pair.
An interior pair of one-syllable Hangul fragments can be rejoined only when their measured gap is
effectively zero, gaps on both sides are wide, character pitch agrees, each fragment has strong
confidence, and recognizing their union at 99.99% or better exactly returns their concatenation.
An internal two-plus-one-syllable pair has a separate shallow-overlap profile: both surrounding
gaps must clearly separate it from neighboring words, character pitch must agree, the
two-syllable fragment must be at least 99.7% confident, and recognizing the union at 99.97% or
better must exactly return the concatenation. Its pitch-boundary comparison tolerates only
floating-point roundoff. A line-initial two-plus-one pair has a separate narrower profile:
overlap must be 5.5% to 6% of line height, the following gap at least 17%, character pitch
compatible, fragment confidence at least 99.87% and 97.9%, and exact combined recognition at
least 99.96%.
An internal one-plus-four-syllable pair has a separate profile for a one-pixel overlap between
otherwise positive neighboring gaps. It additionally requires compatible pitch, at least 99.75%
confidence for the four-syllable fragment, and exact combined recognition at 99.75% or better.
An isolated internal one-plus-four pair has a distinct positive-gap profile requiring fragment
confidence of at least 99.84% and 99.97%, a gap of 35% to 35.5% of line height, surrounding
boundaries of at least 54% and 67%, pitch agreement within 10%, and exact combined recognition at
99.96% or better. Subpixel verification coordinates are normalized, and either adjacent union at
99.8% preserves the original pair.
An internal one-plus-two-syllable close pair can be rejoined only when both neighboring gaps are
wider, the one- and two-syllable fragments are at least 99.88% and 99.98% confident respectively,
character pitch is compatible, and exact combined recognition reaches 99.99%.
An isolated-wide pure-Hangul one-plus-two pair has a separately reviewed profile requiring
fragment confidence of at least 83.5% and 99.88%, a gap of 36% to 36.5% of line height,
preceding and following boundaries of at least 77% and 61%, pitch agreement within 27%, and exact
combined recognition at 99.98% or better. Subpixel verification coordinates are normalized, and
the pair is preserved when either available adjacent union reaches 98% confidence.
A line-initial one-plus-two-syllable pair has a separate profile: fragment confidence must reach
99.92% and 99.86%, their gap must be 36% to 36.5% of line height, the following boundary must be
61% to 62.5%, character pitch must be compatible, and exact combined recognition must reach
99.975%. An internal touching-following variant instead requires fragment confidence of 99.99%
and 99.93%, a gap of 6% to 6.5%, a preceding boundary of at least 37%, a following boundary
within 0.5% of touching, and compatible pitch. It uses the same combined-recognition floor.
Both variants reject recovery when recognizing the two-syllable fragment with its following
neighbor reaches 90% confidence. Only these two verification paths normalize subpixel crop
coordinates before integer rounding.
A line-initial three-plus-three-syllable pair has a pure-Hangul merge profile requiring fragment
confidence of at least 99.65% and 99.99%, a gap of 26% to 26.5% of line height, a following
boundary of 54% to 55%, pitch agreement within 4%, and exact combined recognition at 99.93% or
better. An isolated internal three-plus-three pair instead requires fragment confidence of at
least 99.81% and 99.68%, a gap of 35% to 36.5%, a preceding boundary of at least 61%, a
following boundary of at least 44%, pitch agreement within 2%, and exact combined recognition
at 99.83% or better. Both variants normalize subpixel verification coordinates and reject the
candidate when either available adjacent union reaches 99.5% confidence.
Two internal three-plus-two-syllable pure-Hangul profiles recover independently reviewed false
splits. The narrow-gap profile requires fragment confidence of 99.87% and 99.95%, a gap of 5%
to 5.5% of line height, preceding and following boundaries of at least 20% and 25%, pitch
agreement within 11%, and exact combined recognition at 99.79% or better. The isolated-wide
profile requires fragment confidence of 99.81% and 99.94%, a gap of 36% to 36.5%, both
neighboring boundaries of at least 61%, pitch agreement within 4%, and exact combined recognition
at 99.77% or better. Both normalize subpixel verification coordinates and reject the candidate
when an available adjacent union reaches 99% for the narrow profile or 99.5% for the wide one.
Two internal four-plus-two-syllable pure-Hangul profiles recover independently reviewed false
splits. The positive-gap profile requires fragment confidence of 99.87% and 99.97%, a gap of
22.5% to 23% of line height, preceding and following boundaries of at least 51% and 45%, pitch
agreement within 2%, and exact combined recognition at 99.70% or better. The slight-overlap
profile requires fragment confidence of 99.89% and 96.06%, an overlap of 5% to 5.5%, preceding
and following boundaries of at least 36% and 41%, pitch agreement within 15%, and exact combined
recognition at 99.93% or better. Both normalize subpixel verification coordinates and reject the
candidate when either available adjacent union reaches 98.5% confidence.
Two internal three-plus-one-syllable pure-Hangul profiles cover independently reviewed false
splits. The positive-gap profile requires high fragment confidence, a gap of 28% to 28.5% of
line height, wide neighboring boundaries, compatible pitch, exact union recognition at 99.96%
or better, and no adjacent union at 99% or better. The shallow-overlap correction permits the
union to repair exactly one internal character only when the four-character union remains pure
Hangul, preserves the first two characters and final fragment, reaches 99.95% confidence, and
both adjacent unions remain below 98.5%. Its fragment confidence, overlap, isolation, and pitch
checks are separate from the positive-gap profile.
An isolated-wide internal three-plus-one profile requires fragment confidence of at least 99.97%
and 99.91%, a gap of 36% to 36.5% of line height, preceding and following boundaries of at least
51% and 56%, pitch agreement within 13%, and exact union recognition at 99.97% or better. It
normalizes subpixel verification coordinates and preserves the pair when either available
adjacent union reaches 99% confidence.
An internal overlapping four-plus-one profile requires fragment confidence of at least 99.96%
and 91.4%, an overlap of 4.5% to 5% of line height, preceding and following boundaries of at least
28% and 37%, pitch agreement within 20%, and exact combined recognition at 99.97% or better.
Subpixel verification coordinates are normalized, and either adjacent union at 98% preserves the
original pair.
An internal three-plus-two-syllable pair has a separate narrow-gap profile: fragment confidence
must reach 99.97% and 99.98%, the candidate gap must be 10% to 10.5% of line height, the preceding
boundary at least 25%, and the following boundary a shallow overlap of at most 5.5%. Character
pitch must agree within 2%, and exact combined recognition must reach 99.98%.
An internal two-plus-two-syllable pair has a separate relative-gap profile: both fragments must
be pure Hangul and at least 99.6% confident, their gap must be 15% to 24% of line height, each
available neighboring boundary must be at least ten percentage points wider, character pitch
must agree within 5%, and recognizing the union at 99.6% or better must exactly return their
concatenation. A following word is required, which prevents this profile from acting at line end.
A line-initial two-plus-two profile requires pure-Hangul fragments at least 99.98% and 99.99%
confident, a gap of 25.5% to 26% of line height, a following boundary of at least 46%, and pitch
agreement within 4%. Subpixel verification coordinates are normalized, exact union recognition
must reach 99.99%, and the pair is preserved when its second fragment combined with the following
word reaches 90% confidence.
A five-syllable all-Hangul word can be split into three- and two-syllable eojeols only when a
0.01 word-local CTC-space probe returns exactly two edge-complete crops. Their gap must be 33% to
34% of line height, pitch must agree within 10%, the original word must be at least 99.9%
confident, both parts must be at least 99.92% confident, and concatenating the recognized parts
must exactly reproduce the original word.
A six-syllable all-Hangul word can be split into two three-syllable eojeols only when the normal
CTC threshold retains one word but a 0.01 space probe returns exactly two crops. Their gap must be
28% to 35% of line height, pitch must be compatible, the original and parts must meet separate
high-confidence floors, and concatenating the two recognized parts must exactly reproduce the
original word.
A six-character word has separate two-plus-four CTC split profiles for pure Hangul and a
two-Hangul-plus-four-digit structured identifier. Both require a 0.01 space probe with exactly
two edge-complete crops, a narrow profile-specific gap, compatible character pitch, exact part
types and lengths, separate whole/part confidence floors, and concatenation that exactly
reproduces the original word. The pure-Hangul profile permits a lower detector-capped whole-word
confidence only when one part is near-certain; the structured identifier profile retains high
confidence floors for the word and both parts.
A seven-syllable all-Hangul word has separate five-plus-two and four-plus-three CTC split
profiles. A 0.01 word-local space probe must return exactly two edge-complete crops separated by
32% to 34% of line height, character pitch must agree within 3%, both recognized parts must have
the profile lengths, and their concatenation must exactly reproduce the original recognition. The
original word must be at least 96% confident. The five-plus-two profile requires one part at
99.97% and the other at 97.9%; the four-plus-three profile requires both parts at 99.99%.
An eight-syllable all-Hangul word has a distinct three-plus-five profile. A 0.02 CTC-space probe
must return exactly three- and five-syllable crops separated by 30% to 33% of line height, their
character pitch must agree within 3%, the original and both parts must meet separate
high-confidence floors, and concatenating the parts must exactly reproduce the original word.
Matched slash, dash, quote, or bracket wrappers can restore adjacent boundaries only when every
resulting part contains Hangul; quote and bracket wrappers preserve a directly attached
one-syllable particle. Terminal `:`, `?`, or `!` punctuation can delimit a following Hangul word
only when Hangul occurs on both sides. A missing mandatory space before auxiliary `했다` is
restored only after a multi-syllable `-야` ending. These recoveries retain proportional
per-character geometry and apply to the line fallback as well as segmented recognition; none
changes the global visual-gap threshold.
A separately reviewed central-wrapper profile can split a two-syllable target from a merged word
only under an exact four-word neighbor-confidence, width, gap, and character-category profile. A
word-local 0.001 CTC-space probe must recover the exact four-Hangul prefix, a paired wrapper reading
with the same two target syllables, its closing punctuation, and the exact four-Hangul suffix under
separate geometry and confidence floors. Five prefix, five target, six complete-wrapper, and five
suffix crops must independently reproduce their respective readings. The low-threshold wrapper and
complete-wrapper crops may disagree on quote style only when each wrapper is internally paired and
both preserve the exact two-syllable interior. The recovered wrapper uses its observed full geometry,
so learner-facing punctuation removal retains exact inner target glyph boxes. All unmatched profile,
geometry, recognition, pairing, or crop evidence leaves the original OCR word unchanged.
A separately reviewed direct-retry profile can retain a lower-confidence three-Hangul default
segment only when a higher-confidence enhanced retry changes all three characters under an exact
six-segment line-confidence, width, gap, and category profile. The neighboring evidence includes a
four-Hangul word, a single uppercase marker, one structured ASCII identifier, and one- and
two-Hangul trailing words. Five separately bounded one-pixel pad, trim, and shift crops must exactly
reproduce the direct reading above individual confidence floors. Detector-relative segment edges are
rounded to their original integer CTC coordinates to avoid subpixel arithmetic changing a crop. If
any line, retry, geometry, category, confidence, or crop-consensus signal differs, the normal
higher-confidence retry remains selected. The recovered word retains the original segment geometry
and uses the minimum direct/crop confidence.
A separately reviewed right-wrapper recalibration profile can replace one five-Hangul segment only
under an exact 16-raw-segment/12-selected-word line profile. The candidate must sit between three
reviewed punctuation or symbol fragments with the expected zero-gap geometry, while the surrounding
Hangul words reproduce exact length, confidence, width, and gap evidence. Recognizing the candidate
through its touching right fragment must produce a different pure five-Hangul reading, and seven
direct base/pad/trim/shift crops plus three enhanced crops must all reproduce that alternative above
individual confidence floors. Any profile, category, geometry, confidence, or crop disagreement
keeps the original segment. A confirmed replacement retains the candidate's original geometry and
uses the minimum candidate/crop confidence.
A separately reviewed paired-wrapper recalibration profile can replace one four-Hangul interior
only under an exact 14-raw-segment/11-selected-word line profile. The candidate must retain its
leading wrapper while the adjacent segment supplies the matching right wrapper; surrounding Hangul
lengths, category, confidence, width, and gap evidence must match the reviewed profile. The base
paired crop must preserve the candidate's opening wrapper and supply its matching close; nine
direct base/pad/trim/shift crops and five enhanced crops must all reproduce one different pure
four-Hangul interior above individual confidence floors. Any profile,
geometry, category, wrapper, confidence, or crop disagreement keeps the original segment. A
confirmed replacement retains the original punctuation-stripped interior geometry and uses the
minimum accepted crop confidence.
A separately reviewed enhanced-wrapper recalibration profile can replace one four-Hangul interior
only under an exact 12-raw-word/12-selected-word line profile with reviewed neighboring Hangul
lengths, terminal punctuation, confidence, width, and gap evidence. The enhanced base crop must
preserve the candidate's matched wrapper pair while producing a different pure four-Hangul
interior. Eight direct edge crops and seven enhanced base/pad/trim/shift crops must all reproduce
that alternative above individual confidence floors. Any profile, geometry, category, wrapper,
confidence, or crop disagreement keeps the original segment. A confirmed replacement retains the
original punctuation-stripped interior geometry and uses the minimum accepted crop confidence.
A separately reviewed enhanced two-Hangul recalibration profile can replace one reading only under
an exact six-raw-word/six-selected-word line profile with reviewed neighboring Hangul lengths, two
terminal punctuation signals, confidence, width, and gap evidence. The enhanced base crop must
produce a different pure two-Hangul reading, and four direct boundary crops plus eight enhanced
base/pad/trim/shift crops must all reproduce it above individual confidence floors. Any profile,
geometry, category, confidence, or crop disagreement keeps the original segment. A confirmed
replacement retains the original target geometry and uses the minimum accepted crop confidence.
A separately reviewed terminal-three recalibration profile can replace one three-Hangul interior
only under an exact nine-raw-word/eight-selected-word line profile with reviewed neighboring
Hangul lengths, two terminal punctuation signals, an excluded low-confidence ASCII tail,
confidence, width, and gap evidence. The direct reading and enhanced retry must disagree while
retaining the same terminal punctuation. Five direct boundary crops and eight enhanced crops must
all reproduce one different pure three-Hangul reading above individual confidence floors. Any
profile, geometry, category, punctuation, confidence, or crop disagreement keeps the selected
retry. A confirmed replacement restores the original punctuation, retains the existing geometry,
and uses the minimum accepted raw, retry, or crop confidence.
A separately reviewed terminal paired-wrapper recalibration profile can replace one four-Hangul
interior only under an exact 13-raw-word/12-selected-word line profile. The selected line must
omit one punctuation-only segment and contain exactly one same-geometry, same-shape enhanced retry;
the final candidate must contain a matched wrapper pair around the four-Hangul interior. Reviewed
neighboring character categories, confidence, width, and gap evidence must also match. Seven direct
boundary crops and seven enhanced crops must all reproduce one different pure four-Hangul reading
above individual confidence floors. Any profile, retry, geometry, category, wrapper, confidence,
or crop disagreement keeps the selected reading. A confirmed replacement restores the wrappers,
retains the existing candidate geometry, and uses the minimum candidate or crop confidence; the
ordinary punctuation splitter then exposes the corrected interior without making punctuation a
hover target.
A separately reviewed punctuation-trimmed single-glyph profile can replace one
Hangul-plus-punctuation candidate only under an exact 12-raw-word/12-selected-word line profile.
The candidate must have the reviewed one-pixel overlap and surrounding Hangul lengths, confidence,
width, and gap evidence. Seven direct and seven enhanced detector-relative trim/shift crops must
all reproduce the candidate's existing Hangul glyph above individual confidence floors. This rule
does not infer different text: it removes the punctuation and replaces equal-width character
geometry with the central consensus crop. Any profile, geometry, category, confidence, or crop
disagreement keeps the ordinary punctuation split.
A separately reviewed wrapped single-glyph geometry profile can replace one matched-wrapper-width
candidate only under an exact ten-raw-word/eight-selected-word line profile. The selected retry
must retain the candidate box and existing Hangul glyph while dropping only the opening wrapper;
the raw reading must retain both matched wrappers. Reviewed neighboring character categories,
confidence, width, and gap evidence include two excluded low-confidence ASCII boundary artifacts.
Seven direct and seven enhanced detector-relative trim/shift crops must all reproduce the existing
glyph above individual confidence floors. A confirmed recovery removes the remaining wrapper and
uses the central consensus crop box; any profile, geometry, category, wrapper, confidence, or crop
disagreement keeps the selected retry unchanged.
A separately reviewed leading punctuated-single split can separate one Hangul target and its
trailing punctuation from a following two-Hangul eojeol only under an exact ten-raw-word/
ten-selected-word line profile. Four low CTC-space thresholds must expose identical subword
boundaries while the ordinary candidate remains fused at the higher threshold. One punctuation-
aware boundary crop, seven target crops, and eight following-word crops must reproduce the exact
existing parts in both direct and enhanced recognition above individual confidence floors.
The punctuation remains in sentence context but outside hoverable target geometry; the following
Hangul box begins at its independently confirmed CTC boundary. Any profile, boundary, geometry,
category, confidence, or crop disagreement keeps the fused candidate unchanged.

A separately reviewed low-confidence three-plus-five split can separate a leading three-Hangul
target from a following five-Hangul eojeol across an unrecognized ellipsis only under an exact
eight-raw-word/eight-selected-word line profile. Five low CTC-space thresholds must expose the
same overlapping subword boundaries while the next threshold remains fused. Seven target crops
and seven following-word crops must reproduce the exact existing parts in both direct and enhanced
recognition above their respective confidence floors. Target geometry ends before the punctuation
probe and the following Hangul box begins at its independently confirmed CTC boundary. Any profile,
boundary, geometry, category, confidence, or crop disagreement keeps the fused candidate unchanged.

A separately reviewed leading three-plus-six punctuated split can separate a three-Hangul target
from a following six-Hangul eojeol and terminal punctuation only under an exact
two-raw-word/two-selected-word line profile. Six low CTC-space thresholds must expose the same
subword boundaries while the ordinary threshold remains fused. Seven target crops and six
following-word crops must reproduce the exact existing parts in both direct and enhanced
recognition above their respective confidence floors. Target geometry ends before the punctuation
probe, the following word begins at its confirmed CTC boundary, and ordinary eojeol construction
excludes terminal punctuation from hoverable geometry. Any profile, boundary, geometry, category,
confidence, or crop disagreement keeps the fused candidate unchanged.

A separately reviewed isolated dash-wrapper split can recover a four-Hangul target and a
seven-Hangul following eojeol only under an exact single-segment whole-line profile. The direct
and enhanced whole-line readings, candidate crop, line geometry, confidence, crop dimensions,
character categories, punctuation code points, default segment, and full 15-threshold CTC
signature must all match. Seven direct/enhanced crops independently confirm each of the target,
inner em-dash boundary, following eojeol, and target-plus-boundary readings above separate
confidence floors. The reconstructed paired wrapper uses only the independently agreed leading
and inner em dashes; crop-supported boxes keep both punctuation marks and the restored whitespace
outside hoverable geometry. Any profile, boundary, geometry, category, confidence, segmentation,
or crop disagreement keeps the whole-line reading unchanged.

A separately reviewed internal dash-wrapper split can recover a three-Hangul prefix and paired
em-dash-wrapped two-Hangul target only under an exact six-segment line profile. The neighboring
word shapes, confidence, geometry, crop dimensions, default segments, and full 15-threshold CTC
signature must match. Direct and enhanced candidate readings may disagree only on the closing
dash, while the low-threshold prefix and wrapper crops, six wrapper variants, and seven variants
for each prefix, target, opening dash, and closing dash must reproduce the accepted readings above
separate confidence floors. Crop-supported boxes restore the missing word boundary and exclude
both punctuation marks from hoverable target geometry. Any profile, boundary, geometry, category,
confidence, segmentation, or crop disagreement keeps the original candidate unchanged.

A separately reviewed internal paired-wrapper split can recover a three-Hangul prefix and a
curly-quoted two-Hangul target only under a different exact six-segment line profile. The full
candidate may substitute only its closing quote, while direct and enhanced candidate, prefix, and
wrapper crops must agree on the underlying Hangul. The complete 15-threshold CTC signature must
expose the same overlapping prefix and wrapper at every low threshold and remain fused at ordinary
thresholds. Multiple direct and enhanced prefix, wrapper, target, opening-quote, and closing-quote
crops must reproduce the reviewed readings above separate confidence floors. The reconstructed
paired wrapper restores the missing sentence space, and its proportional inner box keeps both
quotes and whitespace outside hoverable target geometry. Any profile, geometry, category,
confidence, segmentation, or crop disagreement keeps the original candidate unchanged.

A separate three-syllable internal paired-wrapper profile applies only to its own exact six-segment
line fingerprint. The selected retry and raw candidate must agree on a three-Hangul prefix, opening
curly quote, three-Hangul target, and substituted ASCII close while retaining their reviewed
confidence ranges. The complete 15-threshold signature, low-threshold prefix and wrapper crops,
and direct/enhanced candidate readings must match. Multiple prefix, wrapper, target, opening-quote,
and closing-quote variants then independently confirm the restored paired wrapper. Detector-relative
boxes restore the missing sentence space and exclude both quotes and whitespace from hoverable
target geometry. Any profile, retry, geometry, category, confidence, segmentation, or crop
disagreement keeps the fused candidate unchanged.

A separate leading three-syllable paired-wrapper profile applies only to its exact two-segment,
large-serif line fingerprint. The selected and raw candidate must agree on a two-Hangul prefix,
opening curly quote, three-Hangul target, and substituted ASCII close; the enhanced candidate may
normalize only the opening quote while retaining the same prefix, target, and close. Ten low CTC
thresholds must expose identical prefix and wrapper crops while five ordinary thresholds remain
fused. Direct and enhanced prefix, complete-wrapper, target, opening-quote, and closing-quote crop
variants must independently reproduce the reviewed readings above separate confidence floors.
Detector-relative boxes restore the missing sentence space, and proportional wrapper geometry
keeps both quote glyphs and whitespace outside hoverable target geometry. Any profile, retry,
geometry, category, confidence, segmentation, or crop disagreement keeps the fused candidate
unchanged.

Matched opening/closing quote signals and a strong trailing ellipsis signal may restore edge
punctuation that CTC otherwise leaves blank; these operations do not change the selected
Korean surface. A line-level recognition path remains as a fallback when segmentation is
unavailable or produces no usable Korean.

`PipelineCoordinator` uses a capacity-one latest-value queue. When OCR is slower than
pointer movement, stale frames are replaced rather than accumulating latency.

## Privacy and logging

Captures are PIL-owned in-memory objects and are not written to disk. There is no
telemetry or remote recognition code. Network access exists only in the user-initiated
asset setup operation. Production diagnostics must log timing, model versions, and
exception types only—never screenshots or recognized text.

The opt-in release latency recorder carries only a monotonic request timestamp through
`PopupResult`. It measures from immediately before screen capture through the Qt popup
event flush, discards warm-up samples, and writes aggregate JSON only on normal exit.

On supported Windows versions the popup requests `WDA_EXCLUDEFROMCAPTURE`. If Windows
rejects that request, the shell hides and flushes the popup before the next capture.

## Version-one language boundary

The selected unit is one complete eojeol. Within it, the popup may show multiple ordered
lexical definitions and a contextual auxiliary explanation; for example, `먹어 버리다`
identifies `버리다` as a helping verb rather than presenting only its lexical “throw away”
use. A small provenance-backed offline map may display verified spacing such as
`갔다오다` -> `갔다 오다`; Kiwi-only spacing guesses are never shown. These explanations do
not generate a combined translation. Full-line context remains available for later
multi-eojeol construction grouping, so `먹고` and `싶어요` remain separate hover targets.
