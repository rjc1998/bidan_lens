import numpy as np

from bidan_lens.ocr.paddle import _recover_ctc_edge_punctuation


def test_ctc_edge_recovery_requires_paired_quotes_around_hangul() -> None:
    characters = [
        '<blank>',
        '\uac00',
        '\u2026',
        '\u2018',
        '\u2019',
        '\u201c',
        '\u201d',
        ' ',
    ]
    probabilities = np.zeros((4, len(characters)), dtype=np.float32)
    probabilities[:, 0] = 1.0
    probabilities[0, 5] = 0.01
    probabilities[1] = 0.0
    probabilities[1, 1] = 1.0
    probabilities[3, 6] = 0.02
    indices = probabilities.argmax(axis=1)

    recovered = _recover_ctc_edge_punctuation(
        '\uac00', probabilities, indices, characters
    )

    assert recovered == '\u201c\uac00\u201d'


def test_ctc_edge_recovery_ignores_weak_trailing_ellipsis_signal() -> None:
    characters = ['<blank>', '\uac00', '\u2026', ' ']
    probabilities = np.zeros((3, len(characters)), dtype=np.float32)
    probabilities[:, 0] = 1.0
    probabilities[0] = 0.0
    probabilities[0, 1] = 1.0
    probabilities[2, 2] = 0.0004
    indices = probabilities.argmax(axis=1)

    recovered = _recover_ctc_edge_punctuation(
        '\uac00', probabilities, indices, characters
    )

    assert recovered == '\uac00'


def test_ctc_edge_recovery_adds_strong_trailing_ellipsis_signal() -> None:
    characters = ['<blank>', '\uac00', '\u2026', ' ']
    probabilities = np.zeros((3, len(characters)), dtype=np.float32)
    probabilities[:, 0] = 1.0
    probabilities[0] = 0.0
    probabilities[0, 1] = 1.0
    probabilities[2, 2] = 0.001
    indices = probabilities.argmax(axis=1)

    recovered = _recover_ctc_edge_punctuation(
        '\uac00', probabilities, indices, characters
    )

    assert recovered == '\uac00\u2026'

