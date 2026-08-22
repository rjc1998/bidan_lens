import hashlib
import io
import json
import tarfile

import pytest

from bidan_lens.release_assets import (
    ReleaseAssetError,
    _safe_tar_members,
    load_lock,
    verify_file,
)


def test_load_lock_and_verify_file(tmp_path) -> None:
    payload = tmp_path / "model.onnx"
    payload.write_bytes(b"model")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    assert load_lock(lock_path)["schema_version"] == 1
    verify_file(payload, hashlib.sha256(b"model").hexdigest())
    with pytest.raises(ReleaseAssetError, match="checksum"):
        verify_file(payload, "0" * 64)


def test_model_tar_rejects_parent_traversal(tmp_path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    with tarfile.open(archive_path) as archive, pytest.raises(ReleaseAssetError, match="unsafe"):
        _safe_tar_members(archive)
