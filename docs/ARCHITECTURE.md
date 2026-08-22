# Architecture

BiDan Lens keeps the interactive shell separate from language processing so every
stage can be tested and replaced independently.

```text
pointer + in-memory screen crop
              |
              v
Paddle detector -> Paddle Korean recognizer -> OCR document
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
  internal and are translated into beginner-facing explanations.
- `dictionary` compiles a versioned source export into a read-only runtime database.
  `DictionarySourceAdapter` is the seam for future sources.
- `gui` owns display and input only. It never interprets Korean itself.

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

The selected unit is one complete eojeol. Full-line context is retained for correct
morphology and future multi-eojeol grammar construction support. Thus `먹고 싶어요`
produces separate `먹고` and `싶어요` hover results in v1, without losing the context
needed to recognize the construction in a later release.
