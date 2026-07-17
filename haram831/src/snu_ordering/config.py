"""Centralized, side-effect-free path configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LOCAL_DATA_ROOT = Path("C:/Project/data")
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass(frozen=True)
class DataPaths:
    """Paths derived from the observed competition dataset layout."""

    data_root: Path
    output_root: Path

    @classmethod
    def from_environment(
        cls,
        data_root: str | Path | None = None,
        *,
        output_root: str | Path | None = None,
    ) -> "DataPaths":
        selected_data = data_root if data_root is not None else os.environ.get("SNU_DATA_DIR", LOCAL_DATA_ROOT)
        selected_output = (
            output_root if output_root is not None else os.environ.get("SNU_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT)
        )
        return cls(Path(selected_data), Path(selected_output))

    @property
    def train_csv(self) -> Path:
        return self.data_root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_root / "test.csv"

    @property
    def sample_submission_csv(self) -> Path:
        return self.data_root / "sample_submission.csv"

    @property
    def train_images(self) -> Path:
        return self.data_root / "train"

    @property
    def test_images(self) -> Path:
        return self.data_root / "test"

    def validate(self) -> None:
        """Raise one clear error listing every required missing item."""
        required_files = (self.train_csv, self.test_csv, self.sample_submission_csv)
        required_dirs = (self.train_images, self.test_images)
        missing = [str(path) for path in required_files if not path.is_file()]
        missing.extend(str(path) for path in required_dirs if not path.is_dir())
        if missing:
            formatted = "\n - ".join(missing)
            raise FileNotFoundError(f"Required competition data items are missing:\n - {formatted}")
