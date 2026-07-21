"""Reproducible CUDA memory measurements with explicit peak resets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GpuMemoryReporter:
    def __init__(self) -> None:
        self.measurements: dict[str, dict[str, Any]] = {}

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

    def record_batch(self, name: str, batch: dict[str, Any]) -> None:
        """Record tensor shapes and storage without copying tensors off device."""

        torch = self._torch()
        tensors: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for key, value in batch.items():
            if not torch.is_tensor(value):
                continue
            size_bytes = int(value.numel() * value.element_size())
            total_bytes += size_bytes
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "storage_mib": size_bytes / 2**20,
            }
        details: dict[str, Any] = {
            "tensor_storage_mib": total_bytes / 2**20,
            "tensors": tensors,
        }
        labels = batch.get("labels")
        if torch.is_tensor(labels):
            details["supervised_tokens"] = int(labels.ne(-100).sum().item())
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            details["batch_size"] = int(input_ids.shape[0])
            details["sequence_length"] = int(input_ids.shape[1])
        image_grid = batch.get("image_grid_thw")
        if torch.is_tensor(image_grid):
            details["image_grid_volume"] = int(image_grid.prod(dim=1).sum().item())
        self.measurements[name] = details

    def record_values(self, name: str, values: dict[str, Any]) -> None:
        self.measurements[name] = dict(values)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.measurements, indent=2), encoding="utf-8")

    def print_report(self) -> None:
        print("GPU memory report:")
        print(json.dumps(self.measurements, indent=2))
