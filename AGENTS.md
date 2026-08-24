# BiDan Lens agent guide

## Project orientation

BiDan Lens is a Windows-first, local Korean OCR popup dictionary built with Python 3.12,
PyQt6, and a `src/` package layout. The interactive path is screen capture -> OCR ->
whole-eojeol hit testing -> sentence-aware Kiwi analysis -> local KRDict lookup -> popup.
Keep normal use private, offline, responsive, and useful to beginner and intermediate
Korean learners.

## Read only the relevant source of truth

- `README.md`: supported environment, setup, and routine commands.
- `DEVELOPMENT_STATUS.md`: what exists now and what still blocks a public release.
- `docs/ARCHITECTURE.md`: subsystem ownership, data flow, concurrency, and privacy.
- `docs/ASSETS.md`: runtime bundle format, validation, installation, and publishing.
- `docs/QUALITY_TARGETS.md`: benchmark contract and release thresholds.
- `docs/ROADMAP.md`: locked version-one scope, non-goals, and future seams.
- `LICENSE`, `NOTICE.md`, and `THIRD_PARTY_LICENSES.md`: licensing, attribution, and
  distribution obligations.

Do not copy those details into new documents or comments. Update the document that owns
the fact. Treat privacy, licensing, roadmap, quality targets, and version-one boundaries
as product decisions: do not change them without explicit user direction. Update
technical documentation and development status when an implementation change makes
them inaccurate.

## Non-negotiable behavior

- Keep screenshots as in-memory objects. Never log, persist, upload, or include screenshot
  pixels or recognized text in normal diagnostics. Timing, versions, and exception types
  are safe diagnostics.
- Keep startup and lookup offline. Network access is allowed only in an explicit
  user-initiated asset setup or update action; do not introduce a remote fallback.
- Select one complete whitespace-delimited eojeol while retaining its containing sentence,
  exact character span, and geometry. Spaces and punctuation are not hover targets.
- Keep raw Kiwi tags internal. Present only conservative learner-facing labels, preserve
  dictionary sense order, and visibly mark any interpretation that differs from OCR text.
- Preserve the capacity-one latest-value pipeline: discard superseded work, do not queue
  stale frames, keep blocking OCR and analysis off the Qt thread, and marshal UI work back
  to that thread.
- Keep runtime models and dictionary databases out of Git. Validate bundle paths, schema
  and app compatibility, byte sizes, and SHA-256 hashes before atomic activation; a failed
  install must not replace the working bundle.
- `meikipop/` is an excluded reference checkout. Do not commit or modify it, copy Japanese
  providers or settings, or weaken its required attribution when adapting shell behavior.

## Implementation conventions

- Preserve subsystem boundaries: `gui` displays and routes input; `pipeline` coordinates;
  `ocr`, `analysis`, and `dictionary` own their respective domain logic.
- Prefer explicit type hints, small typed interfaces or protocols, injected collaborators,
  and frozen slotted dataclasses for immutable cross-stage values.
- Keep tests deterministic, offline, and independent of production models or KRDict data.
  Use fakes and temporary fixtures at external boundaries.
- Do not add or change a runtime dependency, model, dataset, or asset source unless the task
  requires it. When required, verify Windows and PyInstaller compatibility and update the
  relevant packaging, provenance, license, and notice records.
- Do not edit generated caches, `build/`, `dist/`, runtime assets, ONNX models, or SQLite
  outputs as source changes.

## Verification and handoff

- Run focused tests while iterating. Before handing off code changes, run
  `python -m ruff check .` and `python -m pytest`.
- Run `pyinstaller --noconfirm packaging/bidan_lens.spec` after changes to dependencies,
  entrypoints, packaging, hidden imports, or bundled resources.
- For documentation-only changes, validate content, internal links, and consistency; the
  Python suite is not required unless behavior or executable configuration also changed.
- Report the checks run and their results. Explicitly identify anything not verified with
  real Windows capture, production assets, benchmark corpora, or a clean packaged system;
  do not imply those release gates passed without evidence.
- After completing a task that changes tracked files and passing its required verification,
  inspect the final status and diff, stage only files changed for that task, create a concise
  commit, and push the current branch to its configured upstream before the final handoff.
- Do not commit or push while pausing for clarification, while the task is incomplete or
  blocked, when required verification fails, or when the repository is in a conflicted or
  detached-HEAD state. Do not create an empty commit when the task changes no tracked files.
- Never stage or commit pre-existing or unrelated user changes. Never force-push, rewrite
  history, switch branches, or change remotes merely to make the push succeed. If the push is
  rejected or authentication, network access, permissions, or a missing upstream prevents it,
  preserve the local commit and report the blocker.
- In the final handoff, report the commit hash, branch, verification performed, and push result.
