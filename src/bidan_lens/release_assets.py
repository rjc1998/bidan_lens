from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


class ReleaseAssetError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ReleaseAssetError("unsupported release asset lock schema")
    return value


def verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ReleaseAssetError(f"missing release asset: {path}")
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ReleaseAssetError(f"checksum mismatch for {path.name}")


def fetch_sources(lock: dict[str, Any], destination: Path) -> None:
    """Explicitly fetch hash-locked upstream release inputs."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in lock["sources"]:
        target = destination / source["filename"]
        if target.exists():
            verify_file(target, source["sha256"])
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "BiDan-Lens-release-tool/0.1"}
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            verify_file(temporary, source["sha256"])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _safe_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ReleaseAssetError(f"unsafe model archive member: {member.name}")
        members.append(member)
    return members


def extract_models(lock: dict[str, Any], sources: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in lock["sources"]:
        if source["kind"] != "paddle-model":
            continue
        archive_path = sources / source["filename"]
        verify_file(archive_path, source["sha256"])
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination, members=_safe_tar_members(archive), filter="data")


def write_characters(configuration: Path, destination: Path, expected_count: int) -> None:
    try:
        import yaml
    except ImportError as error:
        raise ReleaseAssetError("character extraction requires the release extra") from error
    value = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    characters = value["PostProcess"]["character_dict"]
    if not isinstance(characters, list) or len(characters) != expected_count:
        raise ReleaseAssetError("unexpected Paddle character dictionary")
    if any(not isinstance(character, str) or "\n" in character for character in characters):
        raise ReleaseAssetError("invalid Paddle character dictionary entry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(characters) + "\n", encoding="utf-8")
    temporary.replace(destination)


def verify_group(lock: dict[str, Any], key: str, directory: Path) -> None:
    for item in lock[key]:
        verify_file(directory / item["filename"], item["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hash-locked BiDan Lens assets")
    parser.add_argument("--lock", type=Path, default=Path("assets/release-assets.lock.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("destination", type=Path)
    extract = commands.add_parser("extract-models")
    extract.add_argument("sources", type=Path)
    extract.add_argument("destination", type=Path)
    characters = commands.add_parser("characters")
    characters.add_argument("configuration", type=Path)
    characters.add_argument("destination", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("group", choices=("sources", "outputs"))
    verify.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    lock = load_lock(arguments.lock)
    if arguments.command == "fetch":
        fetch_sources(lock, arguments.destination)
    elif arguments.command == "extract-models":
        extract_models(lock, arguments.sources, arguments.destination)
    elif arguments.command == "characters":
        output = next(item for item in lock["outputs"] if item["id"] == "recognition-characters")
        write_characters(arguments.configuration, arguments.destination, output["line_count"])
        verify_file(arguments.destination, output["sha256"])
    else:
        verify_group(lock, arguments.group, arguments.directory)
    print(f"Release asset {arguments.command} completed and verified")


if __name__ == "__main__":
    main()
