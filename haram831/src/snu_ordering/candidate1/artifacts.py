"""Compact, explicit checkpoint layout for Candidate Model 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root))

    @property
    def adapter_dir(self) -> Path:
        return self.root / "adapter"

    @property
    def config_path(self) -> Path:
        return self.root / "run_config.json"

    @property
    def classifier_head_path(self) -> Path:
        return self.root / "classifier_head.pt"

    @property
    def metrics_path(self) -> Path:
        return self.root / "training_metrics.json"

    @property
    def memory_path(self) -> Path:
        return self.root / "memory_report.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    @property
    def trainer_state_path(self) -> Path:
        return self.root / "trainer_state.pt"

    def create(self) -> None:
        self.adapter_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, value: Mapping[str, Any] | list[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
