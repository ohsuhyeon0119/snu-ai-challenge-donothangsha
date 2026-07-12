import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from config import TRAIN_CSV, TEST_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, CACHE_DIR

MODEL_NAME = "openai/clip-vit-base-patch32"  # open-source, released 2021, pre-cutoff

def extract(csv_path, img_dir, out_path, limit=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    ids, image_embs, text_embs = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            imgs = [
                Image.open(img_dir / row["Id"] / row[f"Input_{i}"]).convert("RGB")
                for i in range(1, 5)
            ]
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)
            img_feat = model.get_image_features(**img_inputs).pooler_output  # (4, 512)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_inputs = processor(
                text=[row["Sentence"]], return_tensors="pt", truncation=True
            ).to(device)
            txt_feat = model.get_text_features(**txt_inputs).pooler_output   # (1, 512)
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
