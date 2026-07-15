import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("images", nargs=4)
    args = parser.parse_args()

    from snu_frame_ordering.contact_sheet import save_contact_sheet

    save_contact_sheet(args.images, args.out, size=args.size)
    print(args.out)


if __name__ == "__main__":
    main()
