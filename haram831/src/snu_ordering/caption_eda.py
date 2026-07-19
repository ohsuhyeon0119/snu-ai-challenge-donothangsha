"""CLI and reusable summaries for caption-boundary bucket analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .caption_structure import (
    BoundaryKind,
    event_count_bucket,
    extract_caption_structure,
)
from .dataset import load_metadata

BUCKETS = ("1", "2", "3", "4", "5+")
BOUNDARY_KINDS: tuple[BoundaryKind, ...] = (
    "NEXT",
    "BEFORE_AFTER",
    "OVERLAP",
    "STRONG",
    "WEAK",
)


def _ordered_histogram(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(value): counts[value] for value in sorted(counts)}


def build_caption_bucket_rows(
    frame: pd.DataFrame,
    *,
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """Return one auditable caption-structure record per metadata row."""

    if "Id" not in frame or "Sentence" not in frame:
        raise ValueError("frame must contain Id and Sentence columns")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        structure = extract_caption_structure(str(row["Sentence"]))
        confident_count = structure.confident_event_count(confidence_threshold)
        kind_counts = Counter(boundary.kind for boundary in structure.boundaries)
        record: dict[str, Any] = {
            "Id": str(row["Id"]),
            "surface_event_count": structure.event_count,
            "surface_bucket": event_count_bucket(structure.event_count),
            "confident_event_count": confident_count,
            "confident_bucket": event_count_bucket(confident_count),
            "segments": json.dumps(structure.segments, ensure_ascii=False),
            "boundaries": json.dumps(
                [
                    {
                        "kind": boundary.kind,
                        "marker": boundary.marker,
                        "confidence": boundary.confidence,
                    }
                    for boundary in structure.boundaries
                ],
                ensure_ascii=False,
            ),
        }
        for kind in BOUNDARY_KINDS:
            record[f"{kind.lower()}_count"] = kind_counts[kind]
        if "No_ordering" in frame:
            record["No_ordering"] = bool(row["No_ordering"])
        records.append(record)
    return pd.DataFrame.from_records(records)


def _bucket_summary(rows: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    total = max(1, len(rows))
    for bucket in BUCKETS:
        selected = rows.loc[rows[column] == bucket]
        entry: dict[str, float | int] = {
            "rows": int(len(selected)),
            "rate": float(len(selected) / total),
        }
        if "No_ordering" in rows:
            entry["no_ordering_rate"] = (
                float(selected["No_ordering"].mean()) if len(selected) else 0.0
            )
        summary[bucket] = entry
    return summary


def build_caption_eda_report(
    frame: pd.DataFrame,
    *,
    confidence_threshold: float = 0.7,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build a JSON-serializable aggregate report and per-row audit table."""

    rows = build_caption_bucket_rows(
        frame, confidence_threshold=confidence_threshold
    )
    surface = rows["surface_event_count"].astype(int)
    confident = rows["confident_event_count"].astype(int)
    relation_summary: dict[str, dict[str, float | int]] = {}
    for kind in BOUNDARY_KINDS:
        counts = rows[f"{kind.lower()}_count"].astype(int)
        relation_summary[kind] = {
            "boundary_count": int(counts.sum()),
            "sentences": int((counts > 0).sum()),
            "sentence_rate": float((counts > 0).mean()) if len(rows) else 0.0,
        }

    report: dict[str, Any] = {
        "rows": int(len(rows)),
        "confidence_threshold": confidence_threshold,
        "surface_event_count": {
            "mean": float(surface.mean()) if len(rows) else 0.0,
            "median": float(surface.median()) if len(rows) else 0.0,
            "histogram": _ordered_histogram(surface.tolist()),
            "buckets": _bucket_summary(rows, "surface_bucket"),
        },
        "confident_event_count": {
            "mean": float(confident.mean()) if len(rows) else 0.0,
            "median": float(confident.median()) if len(rows) else 0.0,
            "histogram": _ordered_histogram(confident.tolist()),
            "buckets": _bucket_summary(rows, "confident_bucket"),
        },
        "boundary_kinds": relation_summary,
    }
    return report, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze variable-length temporal action hints in training captions."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-rows-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_metadata(args.train_csv, split="train", limit=args.limit)
    report, rows = build_caption_eda_report(
        frame, confidence_threshold=args.confidence_threshold
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.output_rows_csv is not None:
        args.output_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.output_rows_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
