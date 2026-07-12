import os
from pathlib import Path

# Override with `export SNU_DATA_DIR=/path/to/snuaichallenge_data` if the
# data lives somewhere other than the default sibling-of-repo location.
DATA_DIR = Path(os.environ.get(
    "SNU_DATA_DIR",
    Path(__file__).resolve().parents[3] / "snuaichallenge_data",
))

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"
TEST_IMG_DIR = DATA_DIR / "test"

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"       # extracted embeddings, gitignored
OUTPUT_DIR = REPO_ROOT / "outputs"    # checkpoints + submission.csv, gitignored
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
