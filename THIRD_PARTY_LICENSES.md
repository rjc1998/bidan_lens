# Third-party software and data

This file is a distribution index, not a substitute for upstream license texts. The
PyInstaller build recursively collects the exact installed runtime distributions' license
files under `licenses/dependencies/`; it fails when a dependency has neither an installed
license file nor a reviewed version-specific fallback. Asset terms are embedded separately
in the selected asset bundle.

| Component | Purpose | License/source |
| --- | --- | --- |
| MeikiPop | Desktop-shell design basis | GPL-3.0 |
| PaddleOCR PP-OCRv5 models | Korean text detection/recognition | Apache-2.0; complete text in asset bundle |
| ONNX Runtime | Local neural inference | MIT |
| Kiwi / kiwipiepy | Korean morphology | LGPL-2.1-or-later project notice |
| Korean Basic Dictionary | English definitions | CC-BY-SA-2.0-KR; NIKL attribution and complete terms in asset bundle |
| PyQt6 | Desktop UI | GPL-3.0/commercial dual license |
| MSS, Pillow, NumPy, pynput, platformdirs | Runtime support | See each installed distribution |
