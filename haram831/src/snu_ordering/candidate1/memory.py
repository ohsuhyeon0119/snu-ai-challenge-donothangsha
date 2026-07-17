"""Reproducible CUDA memory measurements with explicit peak resets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GpuMemoryReporter:
    def __init__(self) -> None:
        self.measurements: dict[str, dict[str, float | bool | str]] = {}

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    def reset_peak(self) -> None:
        torch = self._torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def record_current(self, name: str) -> None:
        torch = self._torch()
        if not torch.cuda.is_available():
            self.measurements[name] = {"cuda_available": False}
            return
        torch.cuda.synchronize()
        self.measurements[name] = {
            "cuda_available": True,
            "device": torch.cuda.get_device_name(),
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        }

    def record_peak(self, name: str) -> None:
        torch = self._torch()
        if not torch.cuda.is_available():
            self.measurements[name] = {"cuda_available": False}
            return
        torch.cuda.synchronize()
        self.measurements[name] = {
            "cuda_available": True,
            "device": torch.cuda.get_device_name(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.measurements, indent=2), encoding="utf-8")

    def print_report(self) -> None:
        print("GPU memory report:")
        print(json.dumps(self.measurements, indent=2))
