import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_dump(path):
    matrices = {}
    dump_path = Path(path)
    if not dump_path.exists():
        return matrices
    with dump_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            matrices[record["Id"]] = record["scores"]
    return matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--out", default=os.environ.get("SNU_OUT", "outputs/paligemma_submission.csv"))
    parser.add_argument("--split", choices=["test", "val"], default=os.environ.get("SNU_SPLIT", "test"))
    parser.add_argument("--val-csv", default=os.environ.get("SNU_VAL_CSV", "data_work/clean_val.csv"))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("SNU_LIMIT", "0")))
    parser.add_argument("--tta-k", type=int, default=int(os.environ.get("SNU_TTA_K", "3")))
    parser.add_argument("--prior-alpha", type=float, default=float(os.environ.get("SNU_PRIOR_ALPHA", "0")))
    parser.add_argument("--score-chunk", type=int, default=int(os.environ.get("SNU_SCORE_CHUNK", "4")))
    parser.add_argument("--dump-scores", default=os.environ.get("SNU_DUMP_SCORES", ""))
    parser.add_argument("--metrics-out", default=os.environ.get("SNU_METRICS_OUT", ""))
    args = parser.parse_args()

    from peft import PeftModel
    import pandas as pd
    import torch
    from tqdm import tqdm

    from snu_frame_ordering.orders import (
        ALL_IMAGE_ORDERS,
        SIGMAS,
        aggregate_scores,
        answer_from_image_order,
        exact_match,
        image_order_from_answer,
    )
    from snu_frame_ordering.paligemma_common import (
        load_base_model,
        load_processor,
        parse_answer,
        score_row,
        submission_answer_from_image_order,
    )

    def predict_from_matrix(matrix):
        totals = aggregate_scores(matrix[:requested_k], prior_alpha=args.prior_alpha)
        best_idx = max(range(len(totals)), key=lambda idx: totals[idx])
        return ALL_IMAGE_ORDERS[best_idx], totals

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        df = pd.read_csv(data_dir / "test.csv")
        split = "test"
    else:
        df = pd.read_csv(args.val_csv)
        split = "train"
    if args.limit:
        df = df.head(args.limit)

    requested_k = max(1, args.tta_k)
    dumped = load_dump(args.dump_scores) if args.dump_scores else {}
    dumped = {
        row_id: matrix for row_id, matrix in dumped.items()
        if len(matrix) >= requested_k
    }

    done_csv = {}
    if not args.dump_scores and out_path.exists():
        previous = pd.read_csv(out_path)
        done_csv = dict(zip(previous.Id, previous.Answer))
    if dumped or done_csv:
        print(f"resuming: {len(dumped) or len(done_csv)} rows already done")

    rows = df.to_dict("records")
    todo = [row for row in rows if row["Id"] not in dumped and row["Id"] not in done_csv]

    model = processor = None
    if todo:
        processor = load_processor()
        base = load_base_model(load_4bit=True)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model.eval()

    dump_handle = None
    if args.dump_scores:
        dump_path = Path(args.dump_scores)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_handle = dump_path.open("a", encoding="utf-8")

    rows_out = []
    metrics = []
    truths = []
    preds = []
    new_rows = 0
    started = time.monotonic()

    for index, row in enumerate(tqdm(rows)):
        row_id = row["Id"]
        matrix = dumped.get(row_id)
        score_info = None

        if matrix is not None:
            image_order, totals = predict_from_matrix(matrix)
            ranked = sorted(range(len(totals)), key=lambda idx: totals[idx], reverse=True)
            score_info = {
                "score": totals[ranked[0]],
                "margin": totals[ranked[0]] - totals[ranked[1]],
                "tta_k": requested_k,
                "prior_alpha": args.prior_alpha,
            }
        elif row_id in done_csv:
            answer = parse_answer(done_csv[row_id])
            image_order = image_order_from_answer(answer)
        else:
            matrix = score_row(
                model,
                processor,
                row,
                split,
                sigmas=SIGMAS[:requested_k],
                chunk=args.score_chunk,
            )
            if dump_handle:
                dump_handle.write(json.dumps({"Id": row_id, "scores": matrix}) + "\n")
                dump_handle.flush()
            image_order, totals = predict_from_matrix(matrix)
            ranked = sorted(range(len(totals)), key=lambda idx: totals[idx], reverse=True)
            score_info = {
                "score": totals[ranked[0]],
                "margin": totals[ranked[0]] - totals[ranked[1]],
                "tta_k": requested_k,
                "prior_alpha": args.prior_alpha,
            }
            new_rows += 1
            if new_rows % 10 == 0 or index + 1 == len(rows):
                seconds_per_row = (time.monotonic() - started) / max(1, new_rows)
                peak_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                print(
                    f"{index + 1}/{len(rows)} {seconds_per_row:.2f}s/row "
                    f"ETA {(len(rows) - index - 1) * seconds_per_row / 60:.1f}min "
                    f"vram_peak {peak_gb:.1f}GB"
                )

        answer = submission_answer_from_image_order(image_order)
        rows_out.append({"Id": row_id, "Answer": str(answer)})
        metric = {"Id": row_id, "image_order": str(image_order)}
        if score_info:
            metric.update(score_info)
        if args.split != "test":
            truth_order = image_order_from_answer(parse_answer(row["Answer"]))
            truth_answer = answer_from_image_order(truth_order)
            metric["truth_order"] = str(truth_order)
            metric["correct"] = image_order == truth_order
            truths.append(truth_answer)
            preds.append(answer)
        metrics.append(metric)

        if new_rows and new_rows % 25 == 0:
            pd.DataFrame(rows_out).to_csv(out_path, index=False)

    if dump_handle:
        dump_handle.close()

    sub = pd.DataFrame(rows_out)
    if args.split == "test":
        sample = pd.read_csv(data_dir / "sample_submission.csv")
        sub = sample[["Id"]].merge(sub, on="Id", how="left")
        assert sub.Answer.notna().all(), "missing predictions"
    sub.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} rows={len(sub)} tta_k={requested_k} "
        f"alpha={args.prior_alpha}"
    )

    metrics_out = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".metrics.jsonl")
    with metrics_out.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric) + "\n")
    print(f"wrote {metrics_out}")
    if args.split != "test":
        correct = sum(pred == truth for pred, truth in zip(preds, truths))
        print(f"val exact-match: {exact_match(preds, truths):.4f} ({correct}/{len(truths)})")


if __name__ == "__main__":
    main()
