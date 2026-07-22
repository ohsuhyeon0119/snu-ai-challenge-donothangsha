import ast
import csv
from pathlib import Path


TRAIN_COLUMNS = ["Id", "Sentence", "Input_1", "Input_2", "Input_3", "Input_4", "Answer"]
TEST_COLUMNS = ["Id", "Sentence", "Input_1", "Input_2", "Input_3", "Input_4"]


def is_permutation_answer(value):
    return sorted(ast.literal_eval(value)) == [1, 2, 3, 4]


def image_stats(path):
    from PIL import Image, ImageStat

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        stat = ImageStat.Stat(rgb)
        mean = sum(stat.mean) / 3.0
        extrema = rgb.getextrema()
        flat_extrema = [x for channel in extrema for x in channel]
        contrast = max(flat_extrema) - min(flat_extrema)
        return {
            "exists": True,
            "width": rgb.width,
            "height": rgb.height,
            "mean": round(mean, 3),
            "contrast": contrast,
            "near_black": mean < 5.0,
            "error": "",
        }


def audit_split(data_dir, split, image_stats_enabled=True):
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split}.csv"
    image_root = data_dir / split
    required = TRAIN_COLUMNS if split == "train" else TEST_COLUMNS
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["Id"]
            sample_dir = image_root / sample_id
            answer_valid = is_permutation_answer(row["Answer"]) if split == "train" else ""
            for idx in range(1, 5):
                filename = row[f"Input_{idx}"]
                path = sample_dir / filename
                stats = image_stats(path) if image_stats_enabled else {
                    "exists": path.exists(),
                    "width": "",
                    "height": "",
                    "mean": "",
                    "contrast": "",
                    "near_black": "",
                    "error": "",
                }
                rows.append(
                    {
                        "split": split,
                        "Id": sample_id,
                        "input_index": idx,
                        "filename": filename,
                        "answer_valid": answer_valid,
                        **stats,
                    }
                )
    return rows


def write_audit(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "Id",
        "input_index",
        "filename",
        "answer_valid",
        "exists",
        "width",
        "height",
        "mean",
        "contrast",
        "near_black",
        "error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
