import pytest

from benchmarks.plain_live import REQUIRED_SAMPLES, WARMUP_SAMPLES, _fixed_attempts


def test_foreground_uses_five_warmups_and_500_fixed_attempts_without_replacement() -> None:
    samples = tuple(range(WARMUP_SAMPLES + REQUIRED_SAMPLES + 10))

    attempts = _fixed_attempts(samples)

    assert attempts == samples[:505]
    assert len(attempts[:WARMUP_SAMPLES]) == 5
    assert len(attempts[WARMUP_SAMPLES:]) == 500
    assert len(set(attempts)) == 505


def test_foreground_rejects_too_few_fixtures_instead_of_replacing_failures() -> None:
    with pytest.raises(RuntimeError, match='too few fixtures'):
        _fixed_attempts(tuple(range(504)))
