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
- `pipeline.hit_test` selects a complete whitespace-delimited eojeol. Punctuation and
  blank space do not create targets.
- `analysis` receives the full OCR line plus the selected span. Raw morphology tags are
  internal and are translated into ordered lexical components with surface, lemma, and a
  beginner-facing contextual role. Auxiliary tags prioritize KRDict helping-verb/helping-
    adjective groups while retaining other homographs in source order. The adapter examines ten
    Kiwi analyses while exposing at most five popup candidates. A richer multi-component analysis
    may lead only when every component has a local definition, its adjusted Kiwi score is within
    1.0 of the leader, and it does not displace a contextual auxiliary. When the first Kiwi
    analysis has no local definition, a known particle suffix may produce a promoted candidate
    only when its remaining stem has a local dictionary entry; an already segmented,
    dictionary-backed noun may also recover a missing particle feature without changing its
    lemma or definitions.
- `dictionary` compiles a versioned source export into a read-only runtime database.
  Exact headwords take precedence over aliases; aliases remain a fallback.
  `DictionarySourceAdapter` is the seam for future sources.
- `gui` owns display and input only. It never interprets Korean itself.

The detector returns line regions at up to 1280 px input resolution. The recognizer uses
its dynamic-width CTC space probabilities to identify candidate eojeol crops, tightens
those crops against foreground pixels, and splits a missed CTC boundary only at an unusually
wide completely blank visual gap. It recognizes the resulting crops independently and
rebuilds the containing line with per-character and per-eojeol geometry. High-confidence
structured ASCII identifiers remain in sentence context but never become hover targets.
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
