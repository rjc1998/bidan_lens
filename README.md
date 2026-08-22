# BiDan Lens

BiDan Lens is a Windows-first, local Korean reading aid. Point at Korean text on your
screen to see the complete eojeol, English dictionary definitions, its dictionary form,
and a beginner-friendly grammar breakdown.

The version-one pipeline is deliberately local: screenshots remain in memory, OCR runs
on the user's CPU, and dictionary and morphology lookups use downloaded local assets.
There is no telemetry, cloud OCR, or screenshot upload.

## Current status

This repository contains the version-one architecture and a working development shell:

- immutable OCR document and geometry models with whole-eojeol hit testing;
- PaddleOCR-compatible ONNX detection/recognition adapters;
- sentence-aware Korean analysis through Kiwi;
- a versioned KRDict SQLite builder and lookup adapter;
- verified, atomic asset installation and offline bundle import;
- a Windows capture/input/popup shell with automatic and hold-hotkey modes;
- unit and integration tests that do not require downloading production assets.

Production OCR models and the KRDict dataset are intentionally not committed. On first
run the setup flow installs a release asset bundle, or the user can import the same
bundle offline. See [docs/ASSETS.md](docs/ASSETS.md).

## Development

Use Python 3.12 on Windows 10 22H2 or later.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
ruff check .
bidan-lens
```

For an explicit release-latency run, start the app with an aggregate report destination:

```powershell
bidan-lens --latency-report D:\bidan-lens-results\latency.json
```

The report is written on normal exit. It discards five warm-up popups and records only
machine/bundle metadata, sample count, median, and p95 capture-start-to-popup-event-flush
latency. It never contains screenshots, recognized text, definitions, or individual
timings. A release result needs at least 500 successful clean-text popups; partial reports
remain useful during development but are marked incomplete.

The version-one release gate can be constructed without transcribing individual samples.
The developer-only workflow acquires pinned UD, font, and KRDict sources; renders 2,000
plain-text browser/desktop fixtures with exact geometry; derives held-out morphology and
dictionary expectations independently of production code; and creates a complete SHA-256
lock. All downloaded and generated data remains outside Git. See
[benchmarks/README.md](benchmarks/README.md) for setup, build, evaluation, and foreground
Windows benchmark commands.

The adjacent `meikipop/` directory is a local reference checkout and is explicitly
excluded from this repository.

## License

BiDan Lens is licensed under GPL-3.0-or-later. The Windows interaction and popup design
is adapted from MeikiPop; attribution is recorded in [NOTICE.md](NOTICE.md). Downloaded
data and models keep their own licenses, which are displayed during setup and recorded
in each asset manifest.
