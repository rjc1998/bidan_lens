import hashlib
import json
import zipfile

import pytest

from bidan_lens.assets import AssetError, AssetManager
from bidan_lens.bundle_builder import BundleInput, build_bundle


def make_bundle(
    path,
    payload=b"database",
    digest=None,
    asset_path="dictionary.sqlite3",
    minimum_app_version="0.1.0",
):
    digest = digest or hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "bundle_version": "2026.1",
        "minimum_app_version": minimum_app_version,
        "source_url": "https://example.invalid/source",
        "license": "test-only",
        "files": [{"path": asset_path, "sha256": digest, "size": len(payload)}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(asset_path, payload)


def test_bundle_install_is_verified_and_activated(tmp_path) -> None:
    bundle = tmp_path / "bundle.zip"
    make_bundle(bundle)
    manager = AssetManager(tmp_path / "assets")
    installed = manager.install_bundle(bundle)
    assert (installed / "dictionary.sqlite3").read_bytes() == b"database"
    assert manager.current_dir() == installed


def test_bad_checksum_is_rejected_without_activation(tmp_path) -> None:
    bundle = tmp_path / "bad.zip"
    make_bundle(bundle, digest="0" * 64)
    manager = AssetManager(tmp_path / "assets")
    with pytest.raises(AssetError, match="checksum"):
        manager.install_bundle(bundle)
    assert manager.current_dir() is None


def test_zip_traversal_is_rejected(tmp_path) -> None:
    bundle = tmp_path / "unsafe.zip"
    make_bundle(bundle, asset_path="../outside.txt")
    with pytest.raises(AssetError, match="unsafe"):
        AssetManager(tmp_path / "assets").install_bundle(bundle)
    assert not (tmp_path / "outside.txt").exists()


def test_bundle_requiring_newer_app_is_rejected(tmp_path) -> None:
    bundle = tmp_path / "future.zip"
    make_bundle(bundle, minimum_app_version="99.0.0")
    with pytest.raises(AssetError, match="requires BiDan Lens"):
        AssetManager(tmp_path / "assets").install_bundle(bundle)


def test_release_bundle_builder_round_trip(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model")
    bundle = tmp_path / "release.zip"
    digest = build_bundle(
        bundle,
        (BundleInput("models/model.onnx", model),),
        bundle_version="2026.2",
        minimum_app_version="0.1.0",
        source_url="https://example.invalid/release",
        license_notice="test fixture",
    )
    assert len(digest) == 64
    installed = AssetManager(tmp_path / "installed").install_bundle(bundle)
    assert (installed / "models" / "model.onnx").read_bytes() == b"fake model"
