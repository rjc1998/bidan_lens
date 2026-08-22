# Benchmark corpus contract

Release corpora remain outside Git. The repository contains automation, schemas, and
tests only; restricted images and downloaded corpora must never be committed. Normal
application use still keeps screenshots in memory. Corpus-building commands are an
explicit developer workflow and never capture a user's screen.

The release corpus contains:

- `clean/`: 500 controlled website-like and standard-font samples;
- `subtitles/`: 300 known-caption/common-background samples;
- `complex/`: 200 independently annotated game, comic, or scene-text samples;
- `morphology.jsonl`: 300 held-out, independently annotated morphology cases.

Automation removes the need to transcribe individual samples, but it does not turn
synthetic rendering into evidence about every real application. Reports therefore retain
the annotation oracle and source count for each category, and publish 95% Wilson confidence
intervals alongside point accuracy.

## Source manifest

Create `sources.json` before building samples. Every text corpus, font, background set,
and published OCR dataset used by a sample needs an entry and hash-locked license evidence.
An example using held-out Universal Dependencies text and an open font is:

```json
{
  "schema_version": 1,
  "sources": [
    {
      "id": "ud-korean-kaist-2.18",
      "license_id": "CC-BY-SA-4.0",
      "license_evidence": "LICENSES/ud-korean-kaist.txt",
      "annotation_basis": "published converted-manual lemma and XPOS annotations",
      "allowed_oracles": [
        "known-render",
        "published-annotation-independent-map"
      ]
    },
    {
      "id": "noto-sans-kr",
      "license_id": "OFL-1.1",
      "license_evidence": "LICENSES/noto-sans-kr.txt",
      "annotation_basis": "rendering asset",
      "allowed_oracles": ["known-render"]
    }
  ]
}
```

`allowed_oracles` is an explicit trust boundary. A sample is rejected when it references
an unknown source, an unapproved oracle, a training split, duplicate source identity, or
unlocked evidence. Font and background source IDs are recorded as supporting provenance.

## Automated construction

`benchmarks.corpus_builder` prints aggregate construction statistics only. Use a fixed
seed and select source examples before running the production model.

Known rendering accepts JSON Lines containing `text`, `source_sample_id`, and optional
`source_split`. It creates exact eojeol boxes from the text being drawn. Use multiple
licensed Korean fonts. Subtitle runs may use a directory of licensed backgrounds; if none
is supplied, deterministic generated backgrounds are development evidence only.

```powershell
$env:PYTHONPATH = "src;."
python -m benchmarks.corpus_builder render-ocr D:\corpus-input\ud-test.jsonl `
  D:\bidan-lens-corpus clean --source-id ud-korean-kaist-2.18 `
  --font-source-id noto-sans-kr --font D:\fonts\NotoSansKR-Regular.ttf `
  --font D:\fonts\NotoSansKR-Bold.ttf --count 500 --seed 20260822

python -m benchmarks.corpus_builder render-ocr D:\corpus-input\captions.jsonl `
  D:\bidan-lens-corpus subtitles --source-id licensed-captions `
  --font-source-id noto-sans-kr --font D:\fonts\NotoSansKR-Bold.ttf `
  --backgrounds D:\corpus-input\licensed-backgrounds `
  --background-source-id licensed-backgrounds --count 300 --seed 20260822
```

The AI Hub importer accepts its documented COCO-style structure: `images` records with
`id` and `file_name`, plus `annotations` records with `image_id`, `text`, and
`bbox: [x, y, width, height]`. It deterministically imports only positive-area,
single-eojeol Hangul annotations and records aggregate rejection reasons. Dataset access
and acceptance of its terms remain manual.

```powershell
python -m benchmarks.corpus_builder import-aihub D:\aihub\labels D:\aihub\images `
  D:\bidan-lens-corpus complex --source-id aihub-korean-wild-v1 `
  --source-split test --count 200 --seed 20260822
```

Morphology import reads held-out CoNLL-U files. Expected lemmas come from published lemma
and XPOS fields; learner labels come from a small mapping in the builder that is separate
from production Kiwi code. Only cases with a supported particle, tense, honorific,
speech-level, or connective label are selected. The builder never runs Kiwi while
creating expected answers.

```powershell
python -m benchmarks.corpus_builder import-ud D:\bidan-lens-corpus `
  D:\ud\ko_kaist-ud-test.conllu D:\ud\ko_ksl-ud-test.conllu `
  --source-id ud-korean-held-out --source-split test --count 300 --seed 20260822
```

Do not discard analyzer-disagreement cases after selection. A later expert audit can focus
on those cases, but model output must never decide which samples remain in the release set.

## Locking and evaluation

Create the version-two lock after all files and license evidence are present:

```powershell
python -m benchmarks.corpus_builder lock D:\bidan-lens-corpus `
  --corpus-id bidan-lens-v1-eval-2026-08
python -m benchmarks.corpus_builder validate D:\bidan-lens-corpus
```

The generated lock records `sources.json`, all corpus files, and every license-evidence
SHA-256. `--allow-incomplete` supports development runs but never creates release-complete
sample counts.

Run production assets against the locked corpus with:

```powershell
python benchmarks/locked_corpus.py assets/runtime/onnx D:\bidan-lens-corpus
```

The evaluator emits aggregate JSON only. OCR output includes whole-eojeol accuracy,
missing-eojeol rate, latency, and its confidence interval. Morphology output distinguishes
the first result from a correct lemma with a definition anywhere among navigable
alternatives. Contextual best-sense selection and pedagogical wording are not certified by
this automated corpus and must not be claimed as measured outcomes.

`release_baseline.py` remains the deterministic in-memory synthetic precursor. See
`docs/RELEASE_BASELINE_2026-08.md` for the current production-asset baseline and limits.
