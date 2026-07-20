import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--out", default="outputs/paligemma_submission.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from peft import PeftModel
    import pandas as pd
    from tqdm import tqdm

    from snu_frame_ordering.paligemma_common import (
        load_base_model,
        load_processor,
        row_contact_sheet,
        score_orders,
        submission_answer_from_image_order,
    )

    data_dir = Path(args.data_dir)
    test_csv = data_dir / "test.csv"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = load_processor()
    base = load_base_model(load_4bit=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    df = pd.read_csv(test_csv)
    if args.limit:
        df = df.head(args.limit)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Id", "Answer"])
        writer.writeheader()
        for row in tqdm(df.to_dict("records")):
            image = row_contact_sheet(row, "test")
            image_order = score_orders(model, processor, image, row["Sentence"])
            answer = submission_answer_from_image_order(image_order)
            writer.writerow({"Id": row["Id"], "Answer": str(answer)})

    print(out_path)


if __name__ == "__main__":
    main()
