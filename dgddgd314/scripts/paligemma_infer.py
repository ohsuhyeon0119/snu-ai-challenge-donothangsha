import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--out", default=os.environ.get("SNU_OUT", "outputs/paligemma_submission.csv"))
    parser.add_argument("--split", choices=["test", "val"], default=os.environ.get("SNU_SPLIT", "test"))
    parser.add_argument("--val-csv", default=os.environ.get("SNU_VAL_CSV", "data_work/clean_val.csv"))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("SNU_LIMIT", "0")))
    parser.add_argument("--tta-k", type=int, default=int(os.environ.get("SNU_TTA_K", "3")))
    parser.add_argument("--score-chunk", type=int, default=int(os.environ.get("SNU_SCORE_CHUNK", "4")))
    parser.add_argument("--metrics-out", default=os.environ.get("SNU_METRICS_OUT", ""))
    args = parser.parse_args()

    from peft import PeftModel
    import pandas as pd
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer
    from snu_frame_ordering.paligemma_common import (
        load_base_model,
        load_processor,
        parse_answer,
        predict_order,
        submission_answer_from_image_order,
    )

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = load_processor()
    base = load_base_model(load_4bit=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    if args.split == "test":
        df = pd.read_csv(data_dir / "test.csv")
        split = "test"
    else:
        df = pd.read_csv(args.val_csv)
        split = "train"
    if args.limit:
        df = df.head(args.limit)

    rows_out = []
    metrics = []
    correct = 0
    total = 0
    for row in tqdm(df.to_dict("records")):
        image_order, score_info = predict_order(
            model,
            processor,
            row,
            split,
            tta_k=args.tta_k,
            chunk=args.score_chunk,
            return_scores=True,
        )
        answer = submission_answer_from_image_order(image_order)
        rows_out.append({"Id": row["Id"], "Answer": str(answer)})
        metric = {"Id": row["Id"], "image_order": str(image_order), **score_info}
        if args.split != "test":
            truth = image_order_from_answer(parse_answer(row["Answer"]))
            metric["truth_order"] = str(truth)
            metric["correct"] = image_order == truth
            correct += int(metric["correct"])
            total += 1
        metrics.append(metric)

    sub = pd.DataFrame(rows_out)
    if args.split == "test":
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        sub = sample[["Id"]].merge(sub, on="Id", how="left")
        assert sub.Answer.notna().all(), "missing predictions"
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(sub)} tta_k={args.tta_k}")

    metrics_out = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".metrics.jsonl")
    with metrics_out.open("w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
    print(f"wrote {metrics_out}")
    if args.split != "test":
        print(f"val exact-match: {correct / max(1, total):.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
