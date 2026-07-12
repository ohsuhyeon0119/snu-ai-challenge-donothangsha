"""
Track B (stretch): LoRA-finetune the organizer's baseline model
(Qwen2-VL-2B-Instruct) on train.csv.

This is meant to be run on a cloud GPU (Vast.ai, Colab, or Kaggle), NOT
locally (the dev machine has no CUDA GPU).

All paths are overridable via env vars so the same script runs unchanged
on any of those platforms:
    SNU_DATA_DIR      (default /workspace/data/snuaichallenge_data — Vast.ai)
    SNU_ADAPTER_DIR   (default /workspace/qwen2vl_lora_adapter)
    SNU_VAL_SPLIT_CSV (default /workspace/lora_val_split.csv)
    SNU_BATCH_SIZE    (default 1 — Qwen2's ~152k vocab makes the cross-entropy
                       logits tensor scale directly with batch_size * seq_len;
                       batch_size=2 already OOMs a 24GB card here even with
                       gradient checkpointing on, so batching isn't viable for
                       this model on this hardware — kept as a knob for a
                       bigger GPU, not because it currently helps)

Colab: /content is wiped on every session disconnect, so point
SNU_ADAPTER_DIR/SNU_VAL_SPLIT_CSV at a mounted Google Drive path instead,
e.g. `export SNU_ADAPTER_DIR=/content/drive/MyDrive/qwen2vl_lora_adapter`
(after `from google.colab import drive; drive.mount('/content/drive')`).
Kaggle: the dataset is usually already mounted under
/kaggle/input/<competition-slug>/ — point SNU_DATA_DIR there.
"""
import ast
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info

DATA_DIR = Path(os.environ.get("SNU_DATA_DIR", "/workspace/data/snuaichallenge_data"))
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"
ADAPTER_OUT_DIR = Path(os.environ.get("SNU_ADAPTER_DIR", "/workspace/qwen2vl_lora_adapter"))
VAL_SPLIT_CSV = Path(os.environ.get("SNU_VAL_SPLIT_CSV", "/workspace/lora_val_split.csv"))
BATCH_SIZE = int(os.environ.get("SNU_BATCH_SIZE", "1"))

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VAL_FRAC = 0.12
SEED = 42
N_EPOCHS = 2
LR = 1e-4
# ~500 rows worth, matching the old per-row cadence, so a crash/interruption
# never loses more than this much progress.
CHECKPOINT_EVERY_N_STEPS = max(1, 500 // BATCH_SIZE)


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
    # Trades compute for memory (recomputes activations in the backward pass
    # instead of storing them) — needed headroom on a 24GB card at this
    # task's sequence length (4 images/example, batched).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model, processor


def iter_batches(df, batch_size):
    """Fixed, non-shuffled chunks — same order every run, so resuming by
    'skip the first N batches' always skips exactly the same rows."""
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]


def build_batch_inputs(processor, batch_df, image_dir):
    texts, all_images = [], []
    for _, row in batch_df.iterrows():
        messages, target_text = build_example(row, image_dir)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        texts.append(text + target_text)
        all_images.extend(image_inputs)   # flat list: every example contributes exactly 4

    inputs = processor(text=texts, images=all_images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100   # ignore padding in the loss
    return inputs, labels


def train(model, processor, train_df):
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    ADAPTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = ADAPTER_OUT_DIR / "progress.json"
    start_step = 0
    if progress_path.exists():
        start_step = json.loads(progress_path.read_text())["step"]
        print(f"resuming: skipping the first {start_step} batches already trained on")

    batches_per_epoch = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE
    step = 0
    for epoch in range(N_EPOCHS):
        for batch_df in iter_batches(train_df, BATCH_SIZE):
            step += 1
            if step <= start_step:
                continue   # already trained on this batch in a previous session

            inputs, labels = build_batch_inputs(processor, batch_df, TRAIN_IMG_DIR)
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            optim.step()
            optim.zero_grad()

            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{batches_per_epoch}: loss={outputs.loss.item():.4f}")
            if step % CHECKPOINT_EVERY_N_STEPS == 0:
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
