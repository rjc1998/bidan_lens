# Benchmark corpus contract

Release benchmarks are intentionally data-free in Git until licensed or synthetic
fixtures are curated. Store corpora outside the source tree and record only aggregate
results. Never use private user screenshots.

- `clean/`: 500 website and standard-font samples
- `subtitles/`: 300 subtitle/common image-background samples
- `complex/`: 200 game/comic/stylized samples
- `morphology.jsonl`: 300 sentences with target spans, expected lemmas and learner labels

Each OCR sample needs a sidecar JSON file containing expected line text and eojeol boxes.
The release report records machine details, cold/warm state, model and dictionary bundle
versions, whole-eojeol accuracy, median latency, and p95 latency.

`release_baseline.py` is the deterministic synthetic precursor to that locked corpus. It
uses only in-memory generated images and prints aggregate results; see
`docs/RELEASE_BASELINE_2026-08.md` for the current production-asset baseline and limits.

## Locked release evaluator

Keep the real corpus outside the repository. Its root must contain `corpus.lock.json`,
hash-locked license evidence, the three OCR category directories, and
`morphology.jsonl`. The lock has this shape:

```json
{
  "schema_version": 1,
  "corpus_id": "stable-public-or-private-id",
  "license_evidence": ["LICENSES/source.txt"],
  "files": {
    "LICENSES/source.txt": "sha256",
    "clean/0001.json": "sha256",
    "clean/0001.png": "sha256",
    "morphology.jsonl": "sha256"
  }
}
```

Each OCR annotation names an image relative to that annotation and records image-pixel
coordinates. Text is read for comparison but never included in program output:

```json
{
  "image": "0001.png",
  "lines": [
    {
      "text": "expected full line",
      "box": [10, 20, 400, 60],
      "eojeols": [{"text": "expected", "box": [10, 20, 150, 60]}]
    }
  ]
}
```

Each `morphology.jsonl` line contains `sentence`, `target_span`, `expected_lemma`,
`expected_labels`, and an optional `expected_interpreted_surface`. Run the production
assets against the locked corpus with:

```powershell
$env:PYTHONPATH = "src;."
python benchmarks/locked_corpus.py assets/runtime/onnx D:\bidan-lens-corpus
```

The runner fails on changed/unlocked files and incorrect release sample counts. During
corpus development, `--allow-incomplete` permits a partial run but always reports it as
ineligible for release. Output is aggregate JSON only; do not redirect exceptions or
debug output into a corpus containing private captures.
