import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-load-model", action="store_true")
    args = parser.parse_args()

    from snu_frame_ordering.paligemma_prompts import build_prompt

    print(build_prompt("A person opens a door, walks into a room, and sits down.")[:200])

    if args.no_load_model:
        return

    from snu_frame_ordering.paligemma_common import load_base_model, load_processor

    processor = load_processor()
    model = load_base_model(load_4bit=True)
    print(type(processor).__name__)
    print(type(model).__name__)


if __name__ == "__main__":
    main()

