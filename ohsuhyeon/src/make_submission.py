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
