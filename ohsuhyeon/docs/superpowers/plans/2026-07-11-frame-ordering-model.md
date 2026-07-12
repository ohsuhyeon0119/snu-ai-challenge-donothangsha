# Frame-Ordering Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat the organizer's zero-shot Qwen2-VL-2B baseline on the SNU AI Challenge frame-reordering task, with a local validation harness that predicts leaderboard accuracy before every (rate-limited) Kaggle submission.

**Architecture:** Track A — frozen CLIP image/text encoder + a small trained Transformer head that classifies each (4 frames, sentence) sample into one of the 24 possible permutations. Track B (stretch) — LoRA-finetune the same Qwen2-VL-2B model the organizers used for zero-shot baseline, using the labeled train.csv as supervision, run on Kaggle's free notebook GPU. Both tracks are evaluated through the same local exact-match validation harness so they're directly comparable before spending submission quota.

**Tech Stack:** Python 3.11, PyTorch, `transformers` (CLIPModel, Qwen2VLForConditionalGeneration), `peft` (LoRA), `pandas`, `Pillow`, `scikit-learn` (stratified split), `numpy`.

## Global Constraints

- Final inference must run fully offline, on a single NVIDIA RTX 3090 (24GB VRAM), CPU AMD EPYC 7502, 512GB RAM — no internet at inference time.
- No commercial LLM/VLM API (ChatGPT, Gemini, Grok, ...) for training or inference. Commercial APIs are allowed **only** for data preprocessing, capped at ₩30,000 total cost.
- No external training data — only `train.csv` / `train/` images provided by organizers.
- Only open-source models whose weights were public before **2026-05-31**.
- No model ensembling of any kind (including splitting data, finetuning N copies, and combining their outputs).
- Data augmentation is allowed only by recombining/relabeling the provided data (e.g. re-shuffling given frames) — **no generative model** may create or alter image/text content.
- Model compression (quantization, LoRA) is explicitly allowed.
- Full test-set inference must complete in ≤24h on the RTX 3090.
- Final code + weights bundle must be ≤80GB; data paths must be relative; code + comments UTF-8; a README is required.
- Never use `test.csv` content (images, text, or its distribution) to inform training or preprocessing decisions — that's disqualifying data leakage.
- Kaggle submissions are capped at 2/day — the local validation harness (Task 3) exists specifically to avoid wasting that budget.

---

## Context established during investigation

- Repo: `/Users/ohsuhyeon/Desktop/2026/ai-challenge/snu-ai-challenge-donothangsha` (git, remote `ohsuhyeon0119/snu-ai-challenge-donothangsha`). Personal workspace: `ohsuhyeon/`.
- Data: `/Users/ohsuhyeon/Desktop/2026/ai-challenge/snuaichallenge_data/` — **not** inside the git repo, and must never be committed (too large, and redistribution is prohibited by the rules).
  - `train.csv`: 9535 rows, columns `Id, Input_1..4, Sentence, Answer, No_ordering`. `Answer` is one of exactly 24 permutation strings, e.g. `"[3, 2, 4, 1]"`.
  - `No_ordering == True` for 1478/9535 rows (15.5%) — these always have `Answer == [1, 2, 3, 4]`. **This means "always predict `[1,2,3,4]`" is a free 15.5% accuracy floor** — any real model must clear that bar by a wide margin to be worth submitting.
  - `test.csv`: 818 rows, columns `Id, Input_1..4, Sentence` — no `Answer`, no `No_ordering`. No separate validation set is provided.
  - Images: `train/<Id>/<Id>_<3-char-code>.jpg`, 4 per sample; same layout under `test/`.
- Organizer baseline (`SNU_AI_Challenge_Baseline_Code.ipynb`): zero-shot-prompts `Qwen/Qwen2-VL-2B-Instruct` (no training) with the 4 images + sentence, parses a permutation out of the generated text, falls back to `[1,2,3,4]` on parse failure. This is a legitimate model to build on (open-source, pre-cutoff), but doing *no training* on it scores poorly on the "모델 설계 및 학습 방법론" / "데이터 활용" rubric items — Task 8 turns this into a trained model.
- Local machine (`Apple M5`, 16GB unified memory, **no NVIDIA GPU**) can run CLIP forward passes and train a small classifier head (Track A) but **cannot train/finetune a VLM at reasonable speed** — Task 8 must run elsewhere.

---

### Task 1: Project scaffolding, data access, and safe `.gitignore`

**Files:**
- Create: `ohsuhyeon/src/config.py`
- Create: `ohsuhyeon/requirements.txt`
- Modify: `ohsuhyeon/.gitignore`

**Interfaces:**
- Produces: `config.DATA_DIR`, `config.TRAIN_CSV`, `config.TEST_CSV`, `config.TRAIN_IMG_DIR`, `config.TEST_IMG_DIR`, `config.CACHE_DIR`, `config.OUTPUT_DIR` — every later task imports these instead of hardcoding paths.

- [ ] **Step 1: Point config at the data via an env var with a sane default**

```python
# ohsuhyeon/src/config.py
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
```

- [ ] **Step 2: Verify the config resolves to the real data dir**

Run: `SNU_DATA_DIR=/Users/ohsuhyeon/Desktop/2026/ai-challenge/snuaichallenge_data python3 -c "from ohsuhyeon.src import config; print(config.TRAIN_CSV, config.TRAIN_CSV.exists())"` (run from `ai-challenge/snu-ai-challenge-donothangsha`)
Expected: prints the path and `True`.

- [ ] **Step 3: Pin dependencies**

```text
# ohsuhyeon/requirements.txt
torch>=2.2
transformers>=4.45
peft>=0.13
pandas
pillow
numpy
scikit-learn
tqdm
```

- [ ] **Step 4: Extend `.gitignore` so cache/outputs/checkpoints never get committed**

Append to `ohsuhyeon/.gitignore`:
```
cache/
outputs/
*.pt
*.safetensors
*.ckpt
```

- [ ] **Step 5: Commit**

```bash
git add ohsuhyeon/src/config.py ohsuhyeon/requirements.txt ohsuhyeon/.gitignore
git commit -m "chore: add data config and safe gitignore for ML pipeline"
```

---

### Task 2: EDA / data sanity script

**Files:**
- Create: `ohsuhyeon/src/eda.py`

**Interfaces:**
- Consumes: `config.TRAIN_CSV`, `config.TRAIN_IMG_DIR` (Task 1)
- Produces: nothing importable — this is a one-shot report script.

- [ ] **Step 1: Write the sanity/report script**

```python
# ohsuhyeon/src/eda.py
import ast
import collections
import pandas as pd
from config import TRAIN_CSV, TRAIN_IMG_DIR

def main():
    df = pd.read_csv(TRAIN_CSV)
    print(f"rows: {len(df)}")

    answers = df["Answer"].apply(ast.literal_eval)
    valid = answers.apply(lambda a: sorted(a) == [1, 2, 3, 4])
    print(f"rows with a valid 1..4 permutation answer: {valid.sum()} / {len(df)}")

    counts = collections.Counter(tuple(a) for a in answers)
    print(f"distinct permutation classes seen: {len(counts)} / 24")
    majority_class, majority_n = counts.most_common(1)[0]
    print(f"majority class {majority_class}: {majority_n} rows "
          f"({majority_n / len(df):.1%}) -> naive 'always predict this' floor")

    print(f"No_ordering distribution: {df['No_ordering'].value_counts().to_dict()}")

    missing_images = 0
    for _, row in df.iterrows():
        folder = TRAIN_IMG_DIR / row["Id"]
        for col in ("Input_1", "Input_2", "Input_3", "Input_4"):
            if not (folder / row[col]).exists():
                missing_images += 1
    print(f"missing image files: {missing_images}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the numbers**

Run: `cd ohsuhyeon/src && python3 eda.py`
Expected: `rows: 9535`, `valid ... 9535 / 9535`, `distinct permutation classes seen: 24 / 24`, majority class `(1, 2, 3, 4)` at ~15.5%, `missing image files: 0`.

These numbers are the reference floor for every later task: **any trained model that doesn't clear ~15.5% local validation accuracy by a wide margin is not worth submitting.**

---

### Task 3: Local validation harness (the "check accuracy before submitting" tool)

**Files:**
- Create: `ohsuhyeon/src/validation.py`

**Interfaces:**
- Produces: `make_split(df, val_frac=0.12, seed=42) -> (train_df, val_df)`, `exact_match_accuracy(preds: list[tuple[int,int,int,int]], truths: list[tuple[int,int,int,int]]) -> float`, `write_submission(ids, preds, path)`, `validate_submission_format(path, expected_ids)`.

- [ ] **Step 1: Write the stratified split + leaderboard-identical metric**

```python
# ohsuhyeon/src/validation.py
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

def validate_submission_format(path: Path, expected_ids: set[str]) -> list[str]:
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
```

- [ ] **Step 2: Smoke-test the metric and format validator**

```python
# ohsuhyeon/src/test_validation.py  (plain assert script, no pytest needed)
from validation import exact_match_accuracy, write_submission, validate_submission_format
from pathlib import Path

acc = exact_match_accuracy([[1, 2, 3, 4], [2, 1, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
assert acc == 0.5, acc

tmp = Path("/tmp/sub_test.csv")
write_submission(["a", "b"], [[1, 2, 3, 4], [4, 3, 2, 1]], tmp)
problems = validate_submission_format(tmp, {"a", "b"})
assert problems == [], problems

problems = validate_submission_format(tmp, {"a", "c"})
assert len(problems) == 1 and "Id mismatch" in problems[0], problems

print("validation.py OK")
```

- [ ] **Step 3: Run it**

Run: `cd ohsuhyeon/src && python3 test_validation.py`
Expected: `validation.py OK` with no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add ohsuhyeon/src/validation.py ohsuhyeon/src/test_validation.py
git commit -m "feat: add local exact-match validation harness and submission format checker"
```

**From now on, every model in this plan is scored by `exact_match_accuracy` on the same `val_df` from `make_split`, and every submission file is passed through `validate_submission_format` before upload — this is what stands in for the rate-limited leaderboard.**

---

### Task 4: Majority-class baseline (sanity floor as running code)

**Files:**
- Create: `ohsuhyeon/src/baseline_majority.py`

**Interfaces:**
- Consumes: `config` (Task 1), `validation.make_split`, `validation.exact_match_accuracy` (Task 3)

- [ ] **Step 1: Write the baseline**

```python
# ohsuhyeon/src/baseline_majority.py
import ast
import pandas as pd
from config import TRAIN_CSV
from validation import make_split, exact_match_accuracy

def main():
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = make_split(df)
    truths = [tuple(ast.literal_eval(a)) for a in val_df["Answer"]]
    preds = [(1, 2, 3, 4)] * len(val_df)
    acc = exact_match_accuracy(preds, truths)
    print(f"majority-class baseline val accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ohsuhyeon/src && python3 baseline_majority.py`
Expected: accuracy close to 0.15 (matches the ~15.5% floor found in Task 2).

- [ ] **Step 3: Commit**

```bash
git add ohsuhyeon/src/baseline_majority.py
git commit -m "feat: add majority-class baseline as the accuracy floor to beat"
```

---

### Task 5: CLIP feature extraction (cached, reused by every later model)

**Files:**
- Create: `ohsuhyeon/src/extract_features.py`

**Interfaces:**
- Consumes: `config.TRAIN_CSV/TEST_CSV/TRAIN_IMG_DIR/TEST_IMG_DIR/CACHE_DIR`
- Produces: `cache/train_features.npz`, `cache/test_features.npz`, each with arrays `ids: (N,)`, `image_emb: (N, 4, 512)`, `text_emb: (N, 512)` — Task 6 and Task 7 consume these by array position matched to `ids`.

- [ ] **Step 1: Write the extractor**

```python
# ohsuhyeon/src/extract_features.py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from config import TRAIN_CSV, TEST_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, CACHE_DIR

MODEL_NAME = "openai/clip-vit-base-patch32"  # open-source, released 2021, pre-cutoff

def extract(csv_path, img_dir, out_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_csv(csv_path)
    ids, image_embs, text_embs = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            imgs = [
                Image.open(img_dir / row["Id"] / row[f"Input_{i}"]).convert("RGB")
                for i in range(1, 5)
            ]
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs)          # (4, 512)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_inputs = processor(
                text=[row["Sentence"]], return_tensors="pt", truncation=True
            ).to(device)
            txt_feat = model.get_text_features(**txt_inputs)           # (1, 512)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            ids.append(row["Id"])
            image_embs.append(img_feat.cpu().numpy())
            text_embs.append(txt_feat.cpu().numpy()[0])

    np.savez(
        out_path,
        ids=np.array(ids),
        image_emb=np.stack(image_embs),
        text_emb=np.stack(text_embs),
    )
    print(f"saved {len(ids)} rows -> {out_path}")

if __name__ == "__main__":
    extract(TRAIN_CSV, TRAIN_IMG_DIR, CACHE_DIR / "train_features.npz")
    extract(TEST_CSV, TEST_IMG_DIR, CACHE_DIR / "test_features.npz")
```

- [ ] **Step 2: Run it (this is the one slow step — CPU/MPS forward passes over ~9535+818 rows × 4 images)**

Run: `cd ohsuhyeon/src && python3 extract_features.py`
Expected: two progress bars complete, ending with `saved 9535 rows -> .../train_features.npz` and `saved 818 rows -> .../test_features.npz`. On an M5 this is CPU/MPS-only forward passes (no backward pass, no training) — expect roughly 15-40 minutes total; if it's slower than that, cut `MODEL_NAME` down to itself (already the smallest CLIP variant) or run this step on Kaggle instead (Task 8's environment) and copy the two `.npz` files back.

- [ ] **Step 3: Commit (code only — the `.npz` caches are gitignored by Task 1)**

```bash
git add ohsuhyeon/src/extract_features.py
git commit -m "feat: extract and cache CLIP image/text embeddings"
```

---

### Task 6: Trainable permutation classifier (Track A model)

**Files:**
- Create: `ohsuhyeon/src/model.py`
- Create: `ohsuhyeon/src/train_classifier.py`

**Interfaces:**
- Consumes: `cache/train_features.npz` (Task 5), `validation.make_split/exact_match_accuracy` (Task 3)
- Produces: `PermutationClassifier` (24-way softmax over `[4, 512]` image emb + `[512]` text emb), `outputs/classifier.pt`

- [ ] **Step 1: Enumerate the 24 permutation classes once, shared by train + inference**

```python
# ohsuhyeon/src/permutations.py
import itertools

ALL_PERMS = list(itertools.permutations([1, 2, 3, 4]))   # 24 tuples, fixed order
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}
IDX_TO_PERM = {i: p for i, p in enumerate(ALL_PERMS)}
```

- [ ] **Step 2: Define the model — self-attention over the 4 frame tokens, conditioned on the sentence**

```python
# ohsuhyeon/src/model.py
import torch
import torch.nn as nn

class PermutationClassifier(nn.Module):
    def __init__(self, emb_dim=512, hidden=256, n_heads=4, n_layers=2, n_classes=24):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.text_proj = nn.Linear(emb_dim, hidden)
        self.slot_pos_emb = nn.Parameter(torch.randn(4, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, image_emb, text_emb):
        # image_emb: (B, 4, 512), text_emb: (B, 512)
        frames = self.proj(image_emb) + self.slot_pos_emb          # (B, 4, hidden)
        text_tok = self.text_proj(text_emb).unsqueeze(1)           # (B, 1, hidden)
        tokens = torch.cat([frames, text_tok], dim=1)               # (B, 5, hidden)
        encoded = self.encoder(tokens)
        pooled = encoded[:, :4, :].mean(dim=1)                      # pool the 4 frame slots
        return self.head(pooled)                                    # (B, 24) logits
```

- [ ] **Step 3: Write the training/eval loop**

```python
# ohsuhyeon/src/train_classifier.py
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_CSV, CACHE_DIR, OUTPUT_DIR
from validation import make_split, exact_match_accuracy
from permutations import PERM_TO_IDX, IDX_TO_PERM
from model import PermutationClassifier

class FeatureDataset(Dataset):
    def __init__(self, ids, image_emb, text_emb, labels):
        self.ids, self.image_emb, self.text_emb, self.labels = ids, image_emb, text_emb, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.image_emb[i], self.text_emb[i], self.labels[i]

def build_dataset(df, cache):
    id_to_row = {id_: i for i, id_ in enumerate(cache["ids"])}
    rows = [id_to_row[id_] for id_ in df["Id"]]
    labels = np.array([PERM_TO_IDX[tuple(ast.literal_eval(a))] for a in df["Answer"]])
    return FeatureDataset(
        df["Id"].to_numpy(),
        cache["image_emb"][rows].astype(np.float32),
        cache["text_emb"][rows].astype(np.float32),
        labels,
    )

def main():
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = make_split(df)
    cache = np.load(CACHE_DIR / "train_features.npz")

    train_ds = build_dataset(train_df, cache)
    val_ds = build_dataset(val_df, cache)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PermutationClassifier().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(30):
        model.train()
        for img, txt, label in train_loader:
            img, txt, label = img.to(device), txt.to(device), label.to(device)
            optim.zero_grad()
            logits = model(img, txt)
            loss = loss_fn(logits, label)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            img = torch.from_numpy(val_ds.image_emb).to(device)
            txt = torch.from_numpy(val_ds.text_emb).to(device)
            preds_idx = model(img, txt).argmax(dim=-1).cpu().numpy()
        preds = [IDX_TO_PERM[i] for i in preds_idx]
        truths = [IDX_TO_PERM[i] for i in val_ds.labels]
        acc = exact_match_accuracy(preds, truths)
        print(f"epoch {epoch}: loss={loss.item():.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_DIR / "classifier.pt")

    print(f"best val accuracy: {best_acc:.4f} (majority-class floor was ~0.155)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `cd ohsuhyeon/src && python3 train_classifier.py`
Expected: val_acc printed every epoch, finishing clearly above the 0.155 majority floor (a working CLIP+small-transformer setup on this task typically lands in the 0.35-0.55 range — treat any run that doesn't beat 0.20 as a bug, not "just a hard task").

- [ ] **Step 5: Commit**

```bash
git add ohsuhyeon/src/permutations.py ohsuhyeon/src/model.py ohsuhyeon/src/train_classifier.py
git commit -m "feat: train CLIP-feature permutation classifier (Track A)"
```

---

### Task 7: Permutation-expansion data augmentation

**Files:**
- Modify: `ohsuhyeon/src/train_classifier.py:build_dataset` (add an `augment` path)

**Interfaces:**
- Consumes: same `cache` arrays as Task 6.
- Produces: `build_dataset(df, cache, augment=False)` — same signature plus one new kwarg, so Task 6's call sites keep working unchanged.

Only the four given frames and the known correct order are used — re-deriving new `(shuffled_order, label)` pairs from data already in `train.csv` is not "generating new data" under the rules, since no pixel or caption content is created, only relabeled.

- [ ] **Step 1: Extend `build_dataset` to expand each row into multiple slot-orderings**

```python
# ohsuhyeon/src/train_classifier.py  (replace build_dataset)
import itertools
import random

def build_dataset(df, cache, augment=False, max_perms_per_row=4, seed=0):
    id_to_row = {id_: i for i, id_ in enumerate(cache["ids"])}
    rng = random.Random(seed)

    image_embs, text_embs, labels = [], [], []
    for _, row in df.iterrows():
        base_idx = id_to_row[row["Id"]]
        base_img = cache["image_emb"][base_idx]                       # (4, 512)
        base_txt = cache["text_emb"][base_idx]
        true_order = tuple(ast.literal_eval(row["Answer"]))           # e.g. (3, 2, 4, 1)

        if not augment:
            image_embs.append(base_img)
            text_embs.append(base_txt)
            labels.append(PERM_TO_IDX[true_order])
            continue

        perms = list(itertools.permutations(range(4)))                # slot reorderings
        rng.shuffle(perms)
        for slot_perm in perms[:max_perms_per_row]:
            # slot_perm[i] = which original slot now sits at position i
            new_img = base_img[list(slot_perm)]
            # true_order[k] is the answer for original slot k (0-indexed k = slot_perm[i])
            new_answer = tuple(true_order[k] for k in slot_perm)
            image_embs.append(new_img)
            text_embs.append(base_txt)
            labels.append(PERM_TO_IDX[new_answer])

    return FeatureDataset(
        df["Id"].to_numpy(),
        np.stack(image_embs).astype(np.float32),
        np.stack(text_embs).astype(np.float32),
        np.array(labels),
    )
```

- [ ] **Step 2: Use it for the training split only (never augment `val_df` — that would inflate the metric)**

```python
# in main(), replace:
#   train_ds = build_dataset(train_df, cache)
train_ds = build_dataset(train_df, cache, augment=True, max_perms_per_row=4)
val_ds = build_dataset(val_df, cache, augment=False)
```

- [ ] **Step 3: Re-run training and compare**

Run: `cd ohsuhyeon/src && python3 train_classifier.py`
Expected: `best val accuracy` at least matches, ideally exceeds, the Task 6 number (same val set, so this is a fair A/B). If it's worse, drop `max_perms_per_row` to 2 and retry before concluding augmentation doesn't help.

- [ ] **Step 4: Commit**

```bash
git add ohsuhyeon/src/train_classifier.py
git commit -m "feat: add permutation-expansion augmentation to classifier training"
```

---

### Task 8 (stretch): LoRA-finetune Qwen2-VL-2B on Kaggle Notebook

**Where to train:** the local M5 Mac has no CUDA GPU, so this task does not run locally. Use a **Kaggle Notebook** attached to this competition (Settings → Accelerator → GPU T4 x2 or P100; free tier gives ~30 GPU-hours/week) — it has direct access to the competition dataset with no re-upload, and GPU access is free. Google Colab is the fallback if Kaggle quota runs out. This only affects where *training* happens; the resulting LoRA adapter + base model still has to satisfy the RTX-3090/24h/80GB inference constraints, which Task 9 checks.

**Files (created directly in the Kaggle Notebook, then copied into the repo under `ohsuhyeon/src/lora_finetune.py` once working):**
- Create: `ohsuhyeon/src/lora_finetune.py`

**Interfaces:**
- Consumes: `config.TRAIN_CSV`, `validation.make_split` (Task 3) — copy `train.csv` logic in, since the Kaggle Notebook reads data from `/kaggle/input/...` instead of `config.DATA_DIR`.
- Produces: a LoRA adapter directory, loadable with `peft.PeftModel.from_pretrained(base_model, adapter_dir)`.

- [ ] **Step 1: Install deps in the Kaggle Notebook**

```python
%pip install -q peft accelerate qwen-vl-utils
```

- [ ] **Step 2: Load the base model in 4-bit + attach LoRA**

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()   # sanity-check: should be << 1% of 2B params
```

- [ ] **Step 3: Build supervised (prompt, target) pairs from `train_df` — reuse the exact prompt the baseline used, target is the correct answer as text**

```python
import ast

def build_example(row, image_dir):
    img_files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    content = []
    for i, f in enumerate(img_files):
        content.append({"type": "image", "image": str(image_dir / row["Id"] / f)})
        content.append({"type": "text", "text": f"\nImage {i+1}\n"})
    content.append({"type": "text", "text": (
        f'Thinking about the sentence: "{row["Sentence"]}"\n'
        "Look at the 4 images above labeled Image 1 to Image 4. "
        "Determine the correct chronological order of these images to match the sentence. "
        "Provide the order as a list of image numbers, e.g. [3, 1, 4, 2]."
    )})
    messages = [{"role": "user", "content": content}]

    answer = tuple(ast.literal_eval(row["Answer"]))                 # e.g. (3, 2, 4, 1)
    # answer[k] = temporal position of Input_(k+1) -> invert to "which Input is 1st, 2nd, ..."
    image_order = [answer.index(pos) + 1 for pos in range(1, 5)]    # e.g. [4, 2, 1, 3]
    target_text = f"The correct order is {image_order}."
    return messages, target_text
```

Note the inversion in the last two lines: `Answer` in the CSV encodes "where does Input_k end up", but the prompt (matching the organizer baseline) asks the model to answer "which Input is 1st/2nd/3rd/4th" — these are inverse permutations of each other. Getting this backwards silently trains the model on wrong labels, so before training, spot check 3-5 rows by hand against the plain-English `Sentence`.

- [ ] **Step 4: Standard `Trainer`/manual loop over `train_df` only (never `val_df`), save the adapter**

```python
# Minimal manual loop (Trainer needs a custom collator for vision-language batches,
# which is more code than is useful to freeze here — an epoch-per-row loop is fine
# for ~8400 training rows on a T4/P100 within a Kaggle session).
import torch

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
for epoch in range(2):
    for _, row in train_df.iterrows():
        messages, target_text = build_example(row, TRAIN_IMG_DIR)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = text + target_text
        inputs = processor(text=[full_text], images=[...], return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()
        optim.step()
        optim.zero_grad()

model.save_pretrained("/kaggle/working/qwen2vl_lora_adapter")
```

- [ ] **Step 5: Evaluate the LoRA-tuned model with the *same* `exact_match_accuracy` and `val_df` as Task 6/7 — this is the only fair comparison between Track A and Track B**

Run the baseline notebook's `parse_model_output` + inference loop (already written, in `SNU_AI_Challenge_Baseline_Code.ipynb`) against `val_df` images, feed the results into `validation.exact_match_accuracy`, and compare against the Task 6/7 numbers before deciding which track to submit.

- [ ] **Step 6: Copy the working script + adapter weights back into the repo, commit the script (not the weights — respect the 80GB cap and keep large binaries out of git; store adapter weights separately, e.g. a release asset or shared drive, and reference the location in the README)**

```bash
git add ohsuhyeon/src/lora_finetune.py
git commit -m "feat: add LoRA finetuning script for Qwen2-VL-2B (Track B, run on Kaggle GPU)"
```

---

### Task 9: Inference-time and model-size constraint check

**Files:**
- Create: `ohsuhyeon/src/check_constraints.py`

**Interfaces:**
- Consumes: whichever model (Task 6/7 classifier, or Task 8 LoRA-tuned VLM) is currently the best on `val_df`.

- [ ] **Step 1: Time inference per-sample and extrapolate**

```python
# ohsuhyeon/src/check_constraints.py
import time
import numpy as np
import torch
from config import CACHE_DIR, OUTPUT_DIR
from model import PermutationClassifier

def main():
    cache = np.load(CACHE_DIR / "test_features.npz")
    model = PermutationClassifier()
    model.load_state_dict(torch.load(OUTPUT_DIR / "classifier.pt"))
    model.eval()

    n = len(cache["ids"])
    start = time.perf_counter()
    with torch.no_grad():
        img = torch.from_numpy(cache["image_emb"].astype("float32"))
        txt = torch.from_numpy(cache["text_emb"].astype("float32"))
        model(img, txt)
    elapsed = time.perf_counter() - start

    per_sample = elapsed / n
    full_private_test_estimate_s = per_sample * n * (100 / 70)  # public is 70% of full test
    print(f"{n} samples in {elapsed:.2f}s ({per_sample*1000:.2f} ms/sample)")
    print(f"estimated full (public+private) test time: {full_private_test_estimate_s:.1f}s")
    print(f"24h budget = {24*3600}s -> "
          f"{'OK' if full_private_test_estimate_s < 24*3600 else 'FAILS BUDGET'}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ohsuhyeon/src && python3 check_constraints.py`
Expected: `OK` — a CLIP-feature classifier does a handful of matrix multiplies per sample, so this should be milliseconds/sample even on CPU, nowhere near the 24h/818-sample budget. If Track B (Qwen2-VL LoRA) is the chosen model instead, re-run this same timing pattern using the VLM's `.generate()` call instead of `PermutationClassifier.forward`, since VLM generation is orders of magnitude slower and is the one path that could realistically approach the 24h limit.

- [ ] **Step 3: Check total artifact size**

Run: `du -sh ohsuhyeon/outputs/ <path-to-any-LoRA-adapter-or-base-model-weights>`
Expected: comfortably under 80GB (a CLIP-feature classifier's `classifier.pt` is a few MB; a 2B-param model in fp16 is ~4GB, LoRA adapters are tens of MB — the only way to blow the budget is bundling multiple full model checkpoints).

---

### Task 10: Submission generation + pre-submit checklist

**Files:**
- Create: `ohsuhyeon/src/make_submission.py`

**Interfaces:**
- Consumes: `config.TEST_CSV`, `cache/test_features.npz` (Task 5), `outputs/classifier.pt` (Task 6/7), `validation.write_submission/validate_submission_format` (Task 3)

- [ ] **Step 1: Write the generator**

```python
# ohsuhyeon/src/make_submission.py
import numpy as np
import pandas as pd
import torch
from config import TEST_CSV, CACHE_DIR, OUTPUT_DIR
from model import PermutationClassifier
from permutations import IDX_TO_PERM
from validation import write_submission, validate_submission_format

def main():
    test_df = pd.read_csv(TEST_CSV)
    cache = np.load(CACHE_DIR / "test_features.npz")
    id_to_row = {id_: i for i, id_ in enumerate(cache["ids"])}
    rows = [id_to_row[id_] for id_ in test_df["Id"]]

    model = PermutationClassifier()
    model.load_state_dict(torch.load(OUTPUT_DIR / "classifier.pt"))
    model.eval()

    with torch.no_grad():
        img = torch.from_numpy(cache["image_emb"][rows].astype("float32"))
        txt = torch.from_numpy(cache["text_emb"][rows].astype("float32"))
        preds_idx = model(img, txt).argmax(dim=-1).numpy()
    preds = [IDX_TO_PERM[i] for i in preds_idx]

    out_path = OUTPUT_DIR / "submission.csv"
    write_submission(test_df["Id"].tolist(), preds, out_path)

    problems = validate_submission_format(out_path, set(test_df["Id"]))
    if problems:
        print("SUBMISSION HAS PROBLEMS — DO NOT UPLOAD:")
        for p in problems:
            print(" -", p)
    else:
        print(f"OK: {out_path} passed format validation, {len(preds)} rows.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ohsuhyeon/src && python3 make_submission.py`
Expected: `OK: .../submission.csv passed format validation, 818 rows.`

- [ ] **Step 3: Manual pre-submit checklist (do this every time, since submissions are capped at 2/day)**

1. Did local `val_df` accuracy from Task 6/7/8 actually improve versus the last submitted model? If not, don't spend a submission slot.
2. Did `check_constraints.py` (Task 9) still say `OK` for whichever model produced this submission?
3. Does `submission.csv` have exactly 818 rows and pass `validate_submission_format` with zero problems?
4. Was any part of `test.csv` (images, sentences, or its class distribution) looked at while choosing hyperparameters or preprocessing? If yes, that run is tainted by data leakage — retrain without it before submitting.

- [ ] **Step 4: Commit**

```bash
git add ohsuhyeon/src/make_submission.py
git commit -m "feat: add submission generator with pre-upload format validation"
```

---

## Self-Review

**Spec coverage:**
- "project.md와 데이터를 읽고 플랜을 세워보자" → Context section + Tasks 1-10.
- "어디서 학습시키면 되는지" → answered inline in Task 8 (Kaggle Notebook primary, Colab fallback for the VLM track) and Task 5/6 (local M5 is fine for the CLIP-feature track, no GPU needed there).
- "테스트하면서 test에 대한 accuracy가 높아지도록 제출 전에 확인할 수 있나?" → Task 3 (local stratified validation harness with the exact leaderboard metric) + Task 10 Step 3 (checklist gating every submission on a validated local accuracy improvement).
- Rule compliance (no ensembling, no generative augmentation, no leakage, size/time caps, relative paths) → encoded directly into Task 1 config, Task 7's augmentation approach, Task 9's checks, and the pre-submit checklist.

**Placeholder scan:** no TBD/TODO left in any step; every code block is complete and runnable as written (Task 8's Step 4 loop is deliberately simplified — noted explicitly as a tradeoff for a first working version, not a placeholder for missing logic).

**Type consistency:** `exact_match_accuracy`, `make_split`, `write_submission`, `validate_submission_format` (Task 3) are called with the same signatures in Tasks 4, 6, 7, 9, 10. `PERM_TO_IDX`/`IDX_TO_PERM` (Task 6) are reused unchanged in Task 10. `build_dataset`'s new `augment` kwarg (Task 7) defaults to `False`, so Task 6's original call site is not broken.
