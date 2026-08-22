# Scope and roadmap

## Version one

BiDan Lens targets beginner and intermediate Korean learners on Windows. Reliable
websites, forums, news, blogs, social media, standard fonts, and subtitles come first.
Games, comics, and difficult text-on-image backgrounds are measured but secondary.

Every applicable popup provides the recognized complete eojeol, dictionary form,
English definitions, and a beginner-readable breakdown of particles, tense, honorifics,
speech level, and endings. Multiple plausible analyses remain navigable. The app is
CPU-first and fully local after an explicit first-run asset download or offline import.
Dictionary senses preserve source order and plausible analyses remain navigable. Contextual
best-sense ranking is not a version-one accuracy claim until expert-reviewed evidence exists.

## Explicit non-goals for version one

- sentence translation or generative-AI explanations;
- cloud OCR, telemetry, screenshot history, or automatic screenshot storage;
- multi-eojeol construction grouping (`먹고` and `싶어요` remain separate targets);
- romanization, text-to-speech, flashcards/SRS, or reading-history databases;
- raw morphology/POS tag views;
- vertical or handwritten Korean OCR;
- Linux, macOS, or Android releases.

## Later candidates

The retained sentence/span contract supports multi-eojeol grammar constructions without
changing OCR geometry. The dictionary adapter permits separately licensed sources for
slang, new vocabulary, dialects, and names. Linux can reuse the language core with a new
capture/input shell. Android is feasible but requires an Android-native overlay,
MediaProjection permission flow, mobile ONNX packaging, lifecycle work, and a separate
privacy/Play policy review; it is not a direct port of the Windows shell.
