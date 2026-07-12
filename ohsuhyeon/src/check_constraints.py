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
