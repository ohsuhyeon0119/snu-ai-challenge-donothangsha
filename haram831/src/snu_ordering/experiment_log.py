"""Append-only JSON Lines experiment records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REQUIRED_FIELDS = ("experiment_id", "split_id", "config", "metrics")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def load_experiments(path: str | Path) -> list[dict[str, Any]]:
    """Read all records, returning an empty list for a missing log."""
    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Experiment log line {line_number} is not a JSON object")
            records.append(value)
    return records


def append_experiment(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one serializable record without replacing prior experiments."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Experiment record is missing required field(s): {', '.join(missing)}")
    experiment_id = record["experiment_id"]
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")
    if not isinstance(record["config"], Mapping):
        raise ValueError("config must be a mapping")
    if not isinstance(record["metrics"], Mapping):
        raise ValueError("metrics must be a mapping")

    log_path = Path(path)
    existing = load_experiments(log_path)
    if any(item.get("experiment_id") == experiment_id for item in existing):
        raise ValueError(f"duplicate experiment_id: {experiment_id}")

    serializable = _to_serializable(dict(record))
    try:
        encoded = json.dumps(serializable, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Experiment record is not JSON serializable: {exc}") from exc

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
