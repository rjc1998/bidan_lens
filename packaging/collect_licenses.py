from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

RUNTIME_ROOTS = (
    "PyQt6",
    "mss",
    "numpy",
    "onnxruntime",
    "Pillow",
    "kiwipiepy",
    "kiwipiepy_model",
    "platformdirs",
    "pynput",
    "psutil",
    "charset-normalizer",
    "PyYAML",
)
FALLBACKS = {
    "flatbuffers": "flatbuffers-25.12.19-LICENSE.txt",
    "kiwipiepy-model": "kiwipiepy-model-0.23.0-LICENSE.txt",
    "tqdm": "tqdm-4.67.1-LICENCE.txt",
}


def runtime_distributions() -> list:
    pending = list(RUNTIME_ROOTS)
    found = {}
    while pending:
        requested = pending.pop()
        key = canonicalize_name(requested)
        if key in found:
            continue
        try:
            package = distribution(requested)
        except PackageNotFoundError as error:
            raise RuntimeError(f"missing runtime distribution: {requested}") from error
        name = package.metadata["Name"]
        found[canonicalize_name(name)] = package
        for value in package.requires or ():
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return [found[key] for key in sorted(found)]


def _license_files(package) -> list[tuple[str, Path]]:
    files = []
    for relative in package.files or ():
        name = relative.name.lower()
        if not name.startswith(("license", "licence", "copying", "notice")):
            continue
        if name.endswith((".py", ".pyc")):
            continue
        source = Path(package.locate_file(relative))
        if source.is_file():
            files.append((str(relative), source))
    return files


def collect_dependency_licenses(project_root: Path) -> Path:
    packaging_root = (project_root / "packaging").resolve()
    destination = packaging_root / "generated_licenses"
    temporary = packaging_root / "generated_licenses.tmp"
    if destination.parent != packaging_root or temporary.parent != packaging_root:
        raise RuntimeError("unsafe dependency-license output path")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    manifest = []
    try:
        for package in runtime_distributions():
            name = package.metadata["Name"]
            canonical_name = canonicalize_name(name)
            output = temporary / f"{canonical_name}-{package.version}"
            output.mkdir()
            discovered = _license_files(package)
            if not discovered:
                fallback = FALLBACKS.get(canonical_name)
                if fallback is None:
                    raise RuntimeError(f"no complete license file found for {name}")
                source = project_root / "licenses" / "third_party" / fallback
                if not source.is_file():
                    raise RuntimeError(f"missing vendored license fallback for {name}")
                discovered = [(f"vendored:{fallback}", source)]
            copied = []
            for index, (relative, source) in enumerate(discovered, start=1):
                target = output / f"{index:02d}-{source.name}"
                shutil.copyfile(source, target)
                copied.append({"source": relative, "file": target.name})
            metadata = {
                "name": name,
                "version": package.version,
                "license": package.metadata.get("License-Expression")
                or package.metadata.get("License"),
                "homepage": package.metadata.get("Home-page")
                or package.metadata.get("Project-URL"),
                "files": copied,
            }
            (output / "METADATA.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest.append(metadata)
        python_license = Path(sys.base_prefix) / "LICENSE.txt"
        if not python_license.is_file():
            raise RuntimeError("missing CPython license file")
        python_output = temporary / f"cpython-{platform.python_version()}"
        python_output.mkdir()
        shutil.copyfile(python_license, python_output / "LICENSE.txt")
        python_metadata = {
            "name": "CPython",
            "version": platform.python_version(),
            "license": "PSF-2.0",
            "homepage": "https://www.python.org/",
            "files": [{"source": str(python_license), "file": "LICENSE.txt"}],
        }
        (python_output / "METADATA.json").write_text(
            json.dumps(python_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(python_metadata)
        (temporary / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.rmtree(destination, ignore_errors=True)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


if __name__ == "__main__":
    collect_dependency_licenses(Path(__file__).resolve().parents[1])
