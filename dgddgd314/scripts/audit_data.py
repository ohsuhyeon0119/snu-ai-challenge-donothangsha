import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snu_frame_ordering.data_audit import audit_split, write_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing train/test CSVs and images")
    parser.add_argument("--out", default="outputs/data_audit.csv", help="Audit CSV output path")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument(
        "--skip-image-stats",
        action="store_true",
        help="Only check referenced image files exist; do not import Pillow or compute image statistics",
    )
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    rows = []
    for split in splits:
        rows.extend(
            audit_split(
                Path(args.data_dir),
                split,
                image_stats_enabled=not args.skip_image_stats,
            )
        )
    write_audit(rows, args.out)
    print(f"Wrote {len(rows)} image audit rows to {args.out}")


if __name__ == "__main__":
    main()
