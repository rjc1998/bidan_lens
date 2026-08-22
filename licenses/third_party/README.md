# Reviewed license fallbacks

These exact files cover runtime wheels that omit a license payload. The packaging collector
uses them only when the corresponding installed distribution has no license file, and still
records the installed name and version in its generated manifest.

| Distribution | Source | SHA-256 |
| --- | --- | --- |
| flatbuffers 25.12.19 | `https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| tqdm 4.67.1 | `https://github.com/tqdm/tqdm/blob/v4.67.1/LICENCE` | `dc33252e829015e3b150086fb9b3a40f6ad6fb32c2f4610ce812fa677d35986a` |
| kiwipiepy_model 0.23.0 | Installed kiwipiepy 0.23.2 project license notice | `da087325adcc3aff66b12f3dca74ef0153ba2e26b6948277b91bc7e0484590ad` |
