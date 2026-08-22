from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundleInput:
    archive_path: str
    source_path: Path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build_bundle(
    destination: Path,
    files: tuple[BundleInput, ...],
    *,
    bundle_version: str,
    minimum_app_version: str,
    source_url: str,
    license_notice: str,
) -> str:
    missing = [str(item.source_path) for item in files if not item.source_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bundle inputs: {', '.join(missing)}")
    manifest_files = [
        {
            "path": item.archive_path,
            "sha256": _digest(item.source_path),
            "size": item.source_path.stat().st_size,
        }
        for item in files
    ]
    manifest = {
        "schema_version": 1,
        "bundle_version": bundle_version,
        "minimum_app_version": minimum_app_version,
        "source_url": source_url,
        "license": license_notice,
        "files": manifest_files,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for item in files:
                archive.write(item.source_path, item.archive_path)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _digest(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified BiDan Lens asset bundle")
    parser.add_argument("--detection-model", required=True, type=Path)
    parser.add_argument("--recognition-model", required=True, type=Path)
    parser.add_argument("--characters", required=True, type=Path)
    parser.add_argument("--dictionary", required=True, type=Path)
    parser.add_argument("--paddle-license", required=True, type=Path)
    parser.add_argument("--krdict-license", required=True, type=Path)
    parser.add_argument("--krdict-attribution", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--minimum-app-version", default="0.1.0")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-notice", required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="bidan-bundle-") as directory:
        config = Path(directory) / "ocr.json"
        config.write_text(
            json.dumps(
                {
                    "detection_model": "models/korean_detection.onnx",
                    "recognition_model": "models/korean_recognition.onnx",
                    "character_dictionary": "models/korean_characters.txt",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        files = (
            BundleInput("ocr.json", config),
            BundleInput("models/korean_detection.onnx", arguments.detection_model),
            BundleInput("models/korean_recognition.onnx", arguments.recognition_model),
            BundleInput("models/korean_characters.txt", arguments.characters),
            BundleInput("dictionary.sqlite3", arguments.dictionary),
            BundleInput("licenses/PaddleOCR-APACHE-2.0.txt", arguments.paddle_license),
            BundleInput("licenses/KRDict-CC-BY-SA-2.0-KR.html", arguments.krdict_license),
            BundleInput("licenses/KRDict-ATTRIBUTION.txt", arguments.krdict_attribution),
            BundleInput("provenance/release-assets.lock.json", arguments.provenance),
        )
        digest = build_bundle(
            arguments.output,
            files,
            bundle_version=arguments.bundle_version,
            minimum_app_version=arguments.minimum_app_version,
            source_url=arguments.source_url,
            license_notice=arguments.license_notice,
        )
    print(f"Built {arguments.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
