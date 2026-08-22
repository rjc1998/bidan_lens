import json
from pathlib import Path

from bidan_lens.diagnostics import LatencyRecorder


def test_latency_report_discards_warmup_and_contains_only_aggregates(tmp_path: Path) -> None:
    recorder = LatencyRecorder("2026.08.1", warmup_samples=1, required_samples=2)
    recorder.record(1.0, 1.9)
    recorder.record(2.0, 2.4)
    recorder.record(3.0, 3.8)

    report = recorder.snapshot()

    assert report["warmup_samples_discarded"] == 1
    assert report["samples"] == 2
    assert report["latency_median_ms"] == 600.0
    assert report["latency_p95_ms"] == 400.0
    assert report["passes_latency_targets"] is False
    assert "recognized" not in json.dumps(report)


def test_latency_report_is_written_atomically(tmp_path: Path) -> None:
    recorder = LatencyRecorder("test", warmup_samples=0, required_samples=1)
    recorder.record(1.0, 1.2)
    destination = tmp_path / "report.json"

    recorder.write(destination)

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["samples"] == 1
    assert report["passes_latency_targets"] is True
    assert not (tmp_path / ".report.json.tmp").exists()
