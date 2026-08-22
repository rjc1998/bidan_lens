from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from bidan_lens import __version__


class AssetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AssetFile:
    path: str
    sha256: str
    size: int | None = None


@dataclass(frozen=True, slots=True)
class AssetManifest:
    schema_version: int
    bundle_version: str
    minimum_app_version: str
    source_url: str
    license: str
    files: tuple[AssetFile, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AssetManifest:
        if int(value.get("schema_version", 0)) != 1:
            raise AssetError("unsupported asset manifest schema")
        files = tuple(
            AssetFile(str(item["path"]), str(item["sha256"]).lower(), item.get("size"))
            for item in value.get("files", [])
        )
        if not files:
            raise AssetError("asset manifest contains no files")
        minimum_version = str(value.get("minimum_app_version", "0"))
        if _version_parts(minimum_version) > _version_parts(__version__):
            raise AssetError(f"asset bundle requires BiDan Lens {minimum_version} or newer")
        return cls(
            1,
            str(value["bundle_version"]),
            minimum_version,
            str(value.get("source_url", "")),
            str(value.get("license", "")),
            files,
        )


def _version_parts(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in re.findall(r"\d+", value)[:3])
    return parts + (0,) * (3 - len(parts))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AssetManager:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def current_file(self) -> Path:
        return self.root / "current.json"

    def current_dir(self) -> Path | None:
        try:
            value = json.loads(self.current_file.read_text(encoding="utf-8"))
            candidate = self.root / value["bundle_version"]
            return candidate if candidate.is_dir() else None
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None

    def download_and_install(self, url: str, expected_sha256: str | None = None) -> Path:
        """Explicitly download an asset bundle. Never called by background scanning."""
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.root, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BiDan-Lens/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if expected_sha256 and sha256_file(temporary) != expected_sha256.lower():
                raise AssetError("downloaded bundle checksum does not match")
            return self.install_bundle(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def install_bundle(self, bundle: Path) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle) as archive:
            try:
                manifest = AssetManifest.from_dict(json.loads(archive.read("manifest.json")))
            except (KeyError, json.JSONDecodeError) as error:
                raise AssetError("invalid or missing manifest.json") from error
            with tempfile.TemporaryDirectory(dir=self.root, prefix="install-") as directory:
                staging = Path(directory)
                for item in manifest.files:
                    relative = PurePosixPath(item.path)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise AssetError(f"unsafe asset path: {item.path}")
                    try:
                        payload = archive.read(item.path)
                    except KeyError as error:
                        raise AssetError(f"bundle is missing {item.path}") from error
                    if hashlib.sha256(payload).hexdigest() != item.sha256:
                        raise AssetError(f"checksum mismatch for {item.path}")
                    if item.size is not None and len(payload) != item.size:
                        raise AssetError(f"size mismatch for {item.path}")
                    destination = staging.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                (staging / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": manifest.schema_version,
                            "bundle_version": manifest.bundle_version,
                            "minimum_app_version": manifest.minimum_app_version,
                            "source_url": manifest.source_url,
                            "license": manifest.license,
                            "files": [asdict(item) for item in manifest.files],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                destination = self.root / manifest.bundle_version
                if destination.exists():
                    self._verify_install(destination, manifest)
                else:
                    os.replace(staging, destination)
                pointer = self.current_file.with_suffix(".tmp")
                pointer.write_text(
                    json.dumps({"bundle_version": manifest.bundle_version}), encoding="utf-8"
                )
                os.replace(pointer, self.current_file)
                return destination

    @staticmethod
    def _verify_install(directory: Path, manifest: AssetManifest) -> None:
        for item in manifest.files:
            path = directory.joinpath(*PurePosixPath(item.path).parts)
            if not path.is_file() or sha256_file(path) != item.sha256:
                raise AssetError(f"installed asset is invalid: {item.path}")
