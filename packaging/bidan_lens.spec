# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(project_root, "packaging"))
from collect_licenses import collect_dependency_licenses

dependency_licenses = collect_dependency_licenses(Path(project_root))
datas = collect_data_files("kiwipiepy") + collect_data_files("kiwipiepy_model")
datas += [
    (os.path.join(project_root, "LICENSE"), "."),
    (os.path.join(project_root, "NOTICE.md"), "."),
    (os.path.join(project_root, "THIRD_PARTY_LICENSES.md"), "."),
    (str(dependency_licenses), "licenses/dependencies"),
]
binaries = []
hiddenimports = ["_kiwipiepy", "kiwipiepy_model"]

analysis = Analysis(
    [os.path.join(project_root, "src", "bidan_lens", "main.py")],
    pathex=[os.path.join(project_root, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["matplotlib", "pandas", "pytest", "scipy", "tensorflow", "torch"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="bidan-lens",
    console=False,
    contents_directory=".",
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    name="bidan-lens",
)
