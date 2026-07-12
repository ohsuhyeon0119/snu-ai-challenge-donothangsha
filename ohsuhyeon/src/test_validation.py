from pathlib import Path
from validation import exact_match_accuracy, write_submission, validate_submission_format

acc = exact_match_accuracy([[1, 2, 3, 4], [2, 1, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
assert acc == 0.5, acc

tmp = Path("/tmp/sub_test.csv")
write_submission(["a", "b"], [[1, 2, 3, 4], [4, 3, 2, 1]], tmp)
problems = validate_submission_format(tmp, {"a", "b"})
assert problems == [], problems

problems = validate_submission_format(tmp, {"a", "c"})
assert len(problems) == 1 and "Id mismatch" in problems[0], problems

print("validation.py OK")
