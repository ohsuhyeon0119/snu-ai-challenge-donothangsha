import argparse
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--no-load-model", action="store_true")
    parser.add_argument("--no-data", action="store_true")
    args = parser.parse_args()

    from snu_frame_ordering.paligemma_prompts import build_prompt

    print(build_prompt("A person opens a door, walks into a room, and sits down."))

    if args.no_load_model and args.no_data:
        return

    from snu_frame_ordering.orders import image_order_from_answer, target_text
    from snu_frame_ordering.paligemma_common import encode_supervised, load_base_model, load_processor, row_images

    processor = load_processor()
    print(type(processor).__name__)

    if not args.no_data:
        import pandas as pd

        csv_path = Path(args.data_dir) / f"{args.split}.csv"
        row = pd.read_csv(csv_path).iloc[args.row_index].to_dict()
        images = row_images(row, args.split)
        suffix = "[1, 2, 3, 4]"
        if args.split == "train" and "Answer" in row:
            suffix = target_text(image_order_from_answer(ast.literal_eval(row["Answer"])))
        inputs = encode_supervised(processor, images, row["Sentence"], suffix)
        print("processor keys:", sorted(inputs.keys()))
        for key, value in inputs.items():
            print(f"{key}: {shape_of(value)}")
        if "labels" in inputs:
            valid = int((inputs["labels"] != -100).sum().item())
            print("valid label tokens:", valid)

    if args.no_load_model:
        return

    model = load_base_model(load_4bit=True)
    print(type(model).__name__)


if __name__ == "__main__":
    main()
