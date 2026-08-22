# Notices and attribution

BiDan Lens is licensed under GPL-3.0-or-later.

The Windows desktop interaction model, including the capture loop, hover-popup pattern,
and tray/settings organization, is adapted from MeikiPop, Copyright its contributors,
licensed under GPL-3.0. BiDan Lens is a clean Korean-focused implementation and does not
include MeikiPop's Japanese OCR or dictionary assets.

Runtime assets are not stored in this repository. Their bundle manifest must include a
source URL, version, SHA-256 digest, and license identifier. In particular:

- Korean Basic Dictionary (KRDict) data is provided by the National Institute of Korean
  Language under CC BY-SA 2.0 Korea; the bundle includes exact attribution and terms.
- PaddleOCR model files retain the licensing notices provided by PaddlePaddle.
- Kiwi/kiwipiepy is an independent LGPL-licensed dependency.

See `THIRD_PARTY_LICENSES.md` and the installed bundle's `manifest.json` for exact
versions and terms.
