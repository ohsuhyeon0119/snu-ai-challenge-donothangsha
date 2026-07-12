import ast
import itertools
import random

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

def build_dataset(df, cache, augment=False, max_perms_per_row=4, seed=0):
    id_to_row = {id_: i for i, id_ in enumerate(cache["ids"])}
    rng = random.Random(seed)
    # Materialize once: cache["image_emb"]/["text_emb"] decompress the *entire*
    # array from the .npz on every index, so doing it inside the row loop
    # re-reads ~78MB thousands of times and can OOM the process.
    all_image_emb = cache["image_emb"]
    all_text_emb = cache["text_emb"]

    ids, image_embs, text_embs, labels = [], [], [], []
    for _, row in df.iterrows():
        base_idx = id_to_row[row["Id"]]
        base_img = all_image_emb[base_idx]                            # (4, 512)
        base_txt = all_text_emb[base_idx]
        true_order = tuple(ast.literal_eval(row["Answer"]))           # e.g. (3, 2, 4, 1)

        if not augment:
            ids.append(row["Id"])
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
            ids.append(row["Id"])
            image_embs.append(new_img)
            text_embs.append(base_txt)
            labels.append(PERM_TO_IDX[new_answer])

    return FeatureDataset(
        np.array(ids),
        np.stack(image_embs).astype(np.float32),
        np.stack(text_embs).astype(np.float32),
        np.array(labels),
    )

def evaluate(model, ds, device):
    model.eval()
    with torch.no_grad():
        img = torch.from_numpy(ds.image_emb).to(device)
        txt = torch.from_numpy(ds.text_emb).to(device)
        preds_idx = model(img, txt).argmax(dim=-1).cpu().numpy()
    preds = [IDX_TO_PERM[i] for i in preds_idx]
    truths = [IDX_TO_PERM[i] for i in ds.labels]
    return exact_match_accuracy(preds, truths)

def main(augment=False, max_perms_per_row=24, n_epochs=40, checkpoint_name="classifier.pt", seed=0):
    torch.manual_seed(seed)
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = make_split(df)
    cache = np.load(CACHE_DIR / "train_features.npz")

    train_ds = build_dataset(train_df, cache, augment=augment, max_perms_per_row=max_perms_per_row)
    val_ds = build_dataset(val_df, cache, augment=False)
    batch_size = 512 if augment else 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # A plain 256-hidden/2-layer head with lr=1e-3 memorizes the ~8.4k unique
    # training rows within a couple of epochs (train acc climbs, val acc
    # collapses toward/below the majority-class floor) because CLIP's pooled
    # embedding gives this tiny model very little generalizable ordering
    # signal to work with. This smaller/more-regularized config + permutation
    # augmentation (Task 7) is what actually generalizes past the floor.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PermutationClassifier(hidden=128, n_layers=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4 if augment else 1e-4, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        for img, txt, label in train_loader:
            img, txt, label = img.to(device), txt.to(device), label.to(device)
            optim.zero_grad()
            logits = model(img, txt)
            loss = loss_fn(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        acc = evaluate(model, val_ds, device)
        print(f"epoch {epoch}: loss={loss.item():.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_DIR / checkpoint_name)

    print(f"best val accuracy: {best_acc:.4f} (majority-class floor was ~0.155)")
    return best_acc

if __name__ == "__main__":
    main()
