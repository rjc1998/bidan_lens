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

## What this does not close

Synthetic results do not replace the required licensed real website/subtitle/game corpus.
The timings also exclude screen capture, queueing, pointer hit testing, popup rendering, and
cold model startup, so they do not by themselves close the pointer-to-popup latency gate.
Clean Windows 10 VM, mixed-DPI multi-monitor, packaged-build, false-correction, and complete
license-payload tests remain release blockers until separately recorded.
