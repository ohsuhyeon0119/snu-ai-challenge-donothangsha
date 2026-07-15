import ast
import csv
from pathlib import Path


REQUIRED_COLUMNS = ["Id", "Answer"]


def parse_answer(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    if len(parsed) != 4:
        return None
    if sorted(parsed) != [1, 2, 3, 4]:
        return None
    return parsed


def read_ids(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "Id" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no Id column")
        return [row["Id"] for row in reader]


def validate_submission(sample_path, submission_path):
    expected_ids = read_ids(sample_path)
    seen = []
    errors = []

    with Path(submission_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if fieldnames[:2] != REQUIRED_COLUMNS:
            errors.append(
                f"Columns must start with {REQUIRED_COLUMNS}, got {fieldnames}"
            )
        for line_no, row in enumerate(reader, start=2):
            sample_id = row.get("Id")
            seen.append(sample_id)
            answer = parse_answer(row.get("Answer", ""))
            if answer is None:
                errors.append(f"Line {line_no}: invalid Answer={row.get('Answer')!r}")

    if len(seen) != len(expected_ids):
        errors.append(f"Row count mismatch: expected {len(expected_ids)}, got {len(seen)}")
    if len(set(seen)) != len(seen):
        errors.append("Duplicate Id values found")
    if seen != expected_ids:
        errors.append("Id order does not match sample_submission.csv")

    return errors

