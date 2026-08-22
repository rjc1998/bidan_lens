from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    automatic_scanning: bool = True
    activation_key: str = "shift"
    previous_result_key: str = "ctrl+shift+up"
    next_result_key: str = "ctrl+shift+down"
    scan_width: int = 720
    scan_height: int = 240
    scan_interval_ms: int = 180
    popup_delay_ms: int = 80
    max_results: int = 5

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        if not path.exists():
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        known = {field: values[field] for field in cls.__dataclass_fields__ if field in values}
        return cls(**known)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(path)
