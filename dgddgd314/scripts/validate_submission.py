import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snu_frame_ordering.submission import validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing sample_submission.csv")
    parser.add_argument("--submission", required=True, help="Submission CSV to validate")
    args = parser.parse_args()

    sample_path = Path(args.data_dir) / "sample_submission.csv"
    errors = validate_submission(sample_path, args.submission)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()
