import ast
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["Id", "Answer"]


def parse_answer(value):
    return ast.literal_eval(value)


def read_ids(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row["Id"] for row in reader]


def validate_submission(sample_path, submission_path, allow_empty_answer=False):
    expected_ids = read_ids(sample_path)
    seen = []

    with Path(submission_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            sample_id = row["Id"]
            seen.append(sample_id)
            raw_answer = row["Answer"]
            if not allow_empty_answer or raw_answer:
                answer = parse_answer(raw_answer)
                assert sorted(answer) == [1, 2, 3, 4], f"Line {line_no}: invalid Answer={raw_answer!r}"

    assert len(seen) == len(expected_ids), f"Row count mismatch: expected {len(expected_ids)}, got {len(seen)}"
    assert len(set(seen)) == len(seen), "Duplicate Id values found"
    assert seen == expected_ids, "Id order does not match sample_submission.csv"
