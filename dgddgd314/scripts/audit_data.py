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
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    rows = []
    for split in splits:
        rows.extend(audit_split(Path(args.data_dir), split))
    write_audit(rows, args.out)
    print(f"Wrote {len(rows)} image audit rows to {args.out}")


if __name__ == "__main__":
    main()
