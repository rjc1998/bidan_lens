# Runtime asset bundles

OCR models and dictionary data are release artifacts, not Git source. A bundle is a ZIP
with `manifest.json` at its root and all listed files stored at the listed relative paths.

Required runtime files:

```text
manifest.json
ocr.json
models/korean_detection.onnx
models/korean_recognition.onnx
models/korean_characters.txt
dictionary.sqlite3
licenses/PaddleOCR-APACHE-2.0.txt
licenses/KRDict-CC-BY-SA-2.0-KR.html
licenses/KRDict-ATTRIBUTION.txt
provenance/release-assets.lock.json
```

Example configuration:

```json
{
  "detection_model": "models/korean_detection.onnx",
  "recognition_model": "models/korean_recognition.onnx",
  "character_dictionary": "models/korean_characters.txt"
}
```

Example manifest:

```json
{
  "schema_version": 1,
  "bundle_version": "2026.08.1",
  "minimum_app_version": "0.1.0",
  "source_url": "the exact upstream/release provenance URL",
  "license": "combined asset license notice",
  "files": [
    {
      "path": "dictionary.sqlite3",
      "sha256": "64 lowercase hexadecimal characters",
      "size": 123456
    }
  ]
}
```

Every file must be listed. Installation occurs in a temporary sibling directory, checks
paths, byte counts and SHA-256 digests, then atomically activates the version. Failed or
partial downloads never replace the current bundle.

## Building the dictionary

Fetch and verify the exact locked official sources, extract the Paddle models, and build
the current English KRDict export:

```powershell
bidan-lens-release-assets fetch assets/runtime/upstream
bidan-lens-release-assets verify sources assets/runtime/upstream
bidan-lens-release-assets extract-models assets/runtime/upstream assets/runtime/paddle
```

Then run:

```powershell
bidan-lens-build-dictionary krdict.json dictionary.sqlite3 --source-version 2026-07
```

Review source attribution and redistribution terms before publishing a bundle. The
builder keeps normalized headwords, homograph and vocabulary metadata, ordered English
senses, source/version metadata, and an alias table reserved for later sources.

After selecting and validating the exact Paddle Korean ONNX detector/recognizer, build
the distributable bundle from local inputs:

```powershell
bidan-lens-build-bundle `
  --detection-model korean_detection.onnx `
  --recognition-model korean_recognition.onnx `
  --characters korean_characters.txt `
  --dictionary dictionary.sqlite3 `
  --paddle-license PaddleOCR-APACHE-2.0.txt `
  --krdict-license KRDict-CC-BY-SA-2.0-KR.html `
  --krdict-attribution assets/KRDICT_ATTRIBUTION.txt `
  --provenance assets/release-assets.lock.json `
  --output bidan-lens-assets-2026.08.1.zip `
  --bundle-version 2026.08.1 `
  --source-url "immutable provenance URL" `
  --license-notice "complete model and data license summary"
```

Record the printed bundle SHA-256 beside the published release asset.

The exact Paddle-to-ONNX toolchain and expected output hashes are locked in
`assets/release-assets.lock.json`. On Windows, conversion currently requires the locked
Paddle nightly in an isolated environment because released Paddle/Paddle2ONNX wheels do
not load together reliably. Never substitute an unrecorded converter output in a release.

## Publishing

Set `BIDAN_LENS_ASSET_URL` in a packaged launcher or provide the release URL in first-run
setup. Users may download through the setup dialog or import the identical ZIP offline.
Do not publish an unverified moving URL; immutable release assets and an externally
recorded bundle digest are preferred.
