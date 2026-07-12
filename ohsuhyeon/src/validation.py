import ast
import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def make_split(df: pd.DataFrame, val_frac: float = 0.12, seed: int = 42):
    """Stratify on Answer so all 24 classes appear in both splits."""
    strata = df["Answer"]
    train_df, val_df = train_test_split(
        df, test_size=val_frac, random_state=seed, stratify=strata,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def exact_match_accuracy(preds, truths) -> float:
    """Mirrors the leaderboard metric: a row counts only if ALL 4 positions match."""
    assert len(preds) == len(truths)
    correct = sum(1 for p, t in zip(preds, truths) if tuple(p) == tuple(t))
    return correct / len(truths)

def write_submission(ids, preds, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Answer"])
        for id_, pred in zip(ids, preds):
            writer.writerow([id_, str(list(pred))])

def validate_submission_format(path: Path, expected_ids: set) -> list:
    """Returns a list of problems found; empty list means the file is safe to upload."""
    problems = []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seen_ids = {r["Id"] for r in rows}
    if seen_ids != expected_ids:
        problems.append(
            f"Id mismatch: {len(expected_ids - seen_ids)} missing, "
            f"{len(seen_ids - expected_ids)} unexpected"
        )
    for r in rows:
        try:
            parsed = ast.literal_eval(r["Answer"])
        except Exception:
            problems.append(f"Id {r['Id']}: Answer is not parseable: {r['Answer']!r}")
            continue
        if sorted(parsed) != [1, 2, 3, 4]:
            problems.append(f"Id {r['Id']}: Answer {parsed} is not a permutation of 1..4")
    return problems
