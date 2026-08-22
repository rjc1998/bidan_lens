from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


@dataclass(slots=True)
class LatencyRecorder:
    """Collect aggregate-only pointer-to-render timings for an explicit release run."""

    bundle_version: str
    warmup_samples: int = 5
    required_samples: int = 500
    _observed: int = 0
    _durations_ms: list[float] = field(default_factory=list)

    def record(self, requested_at: float, completed_at: float | None = None) -> None:
        completed_at = time.monotonic() if completed_at is None else completed_at
        if requested_at <= 0 or completed_at < requested_at:
            return
        self._observed += 1
        if self._observed <= self.warmup_samples:
            return
        self._durations_ms.append((completed_at - requested_at) * 1000)

    def snapshot(self) -> dict[str, object]:
        durations = self._durations_ms
        median = round(statistics.median(durations), 2) if durations else None
        p95 = round(_p95(durations), 2) if durations else None
        complete = len(durations) >= self.required_samples
        passes = bool(
            complete and median is not None and p95 is not None and median <= 500 and p95 <= 1000
        )
        return {
            "schema_version": 1,
            "measurement": "capture_start_to_popup_event_flush",
            "bundle_version": self.bundle_version,
            "machine": {
                "system": platform.system(),
                "release": platform.release(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            "warmup_samples_discarded": min(self._observed, self.warmup_samples),
            "samples": len(durations),
            "required_samples": self.required_samples,
            "latency_median_ms": median,
            "latency_p95_ms": p95,
            "complete_release_sample_count": complete,
            "passes_latency_targets": passes,
        }

    def write(self, path: Path) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        temporary.replace(path)
