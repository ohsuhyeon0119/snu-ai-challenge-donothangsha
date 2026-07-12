"""
Track B (stretch): LoRA-finetune the organizer's baseline model
(Qwen2-VL-2B-Instruct) on train.csv.

This is meant to be run in a cloud GPU notebook (Colab or Kaggle), NOT
locally (the dev machine has no CUDA GPU). Copy this file's contents into
a notebook cell, or `!python lora_finetune.py` after adjusting DATA_DIR
below to wherever the notebook mounted the competition data.

Colab setup (run once, in a cell, before this script):
    !pip install -q peft accelerate qwen-vl-utils bitsandbytes
    !pip install -q kaggle
    # upload kaggle.json (Kaggle Settings -> API -> Create New Token) first
    import os
    os.environ['KAGGLE_CONFIG_DIR'] = '/content'
    !kaggle competitions download -c <competition-slug> -p /content/data
    !unzip -q /content/data/*.zip -d /content/data
    # then set DATA_DIR below to "/content/data/snuaichallenge_data"

Kaggle notebook setup: the dataset is usually already mounted under
/kaggle/input/<competition-slug>/ — set DATA_DIR to that path instead.
"""
import ast
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info

# --- adjust this for the notebook environment ---
DATA_DIR = Path("/content/data/snuaichallenge_data")   # Colab default; use
                                                          # /kaggle/input/<slug>/ on Kaggle
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"
# Must be on Google Drive, not /content — /content is wiped whenever the
# Colab session disconnects (idle timeout, 12h cap, GPU reclaim), which
# would silently erase every checkpoint. Mount Drive first:
#   from google.colab import drive; drive.mount('/content/drive')
ADAPTER_OUT_DIR = Path("/content/drive/MyDrive/qwen2vl_lora_adapter")
VAL_SPLIT_CSV = Path("/content/drive/MyDrive/lora_val_split.csv")

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VAL_FRAC = 0.12
SEED = 42
N_EPOCHS = 2
LR = 1e-4
CHECKPOINT_EVERY_N_ROWS = 500   # save adapter periodically so a 12h session
                                  # limit never loses more than this much progress


def build_example(row, image_dir):
    """Same prompt format as the organizer baseline notebook, so Track A/B
    predictions stay comparable and the parsing logic can be reused."""
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

    # Answer[k] = temporal position of Input_(k+1). The prompt asks the
    # inverse question ("which Input is 1st/2nd/3rd/4th"), so invert here.
    # Verified by hand against a few rows' plain-English Sentence before
    # trusting this at scale — get this backwards and every label is wrong.
    answer = tuple(ast.literal_eval(row["Answer"]))
    image_order = [answer.index(pos) + 1 for pos in range(1, 5)]
    target_text = f"The correct order is {image_order}."
    return messages, target_text


def load_model_and_processor():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if (ADAPTER_OUT_DIR / "adapter_config.json").exists():
        print(f"found existing adapter at {ADAPTER_OUT_DIR}, resuming from it")
        model = PeftModel.from_pretrained(base_model, ADAPTER_OUT_DIR, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def train(model, processor, train_df):
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    ADAPTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = ADAPTER_OUT_DIR / "progress.json"
    start_step = 0
    if progress_path.exists():
        start_step = json.loads(progress_path.read_text())["step"]
        print(f"resuming: skipping the first {start_step} rows already trained on")

    step = 0
    for epoch in range(N_EPOCHS):
        for _, row in train_df.iterrows():
            step += 1
            if step <= start_step:
                continue   # already trained on this row in a previous session

            messages, target_text = build_example(row, TRAIN_IMG_DIR)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            full_text = text + target_text
            inputs = processor(
                text=[full_text], images=image_inputs, return_tensors="pt"
            ).to(model.device)

            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            optim.step()
            optim.zero_grad()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}: loss={outputs.loss.item():.4f}")
            if step % CHECKPOINT_EVERY_N_ROWS == 0:
                model.save_pretrained(ADAPTER_OUT_DIR)
                progress_path.write_text(json.dumps({"step": step}))
                print(f"checkpointed adapter at step {step} -> {ADAPTER_OUT_DIR}")

    model.save_pretrained(ADAPTER_OUT_DIR)
    progress_path.write_text(json.dumps({"step": step}))
    print(f"final adapter saved -> {ADAPTER_OUT_DIR}")


if __name__ == "__main__":
    assert TRAIN_CSV.exists(), f"{TRAIN_CSV} not found — fix DATA_DIR at the top of this file"

    df = pd.read_csv(TRAIN_CSV)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRAC, random_state=SEED, stratify=df["Answer"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    VAL_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)  # reused by lora_eval.py

    model, processor = load_model_and_processor()

    start = time.perf_counter()
    train(model, processor, train_df)
    print(f"training took {(time.perf_counter() - start) / 3600:.2f}h "
          f"for {len(train_df)} rows x {N_EPOCHS} epochs")
