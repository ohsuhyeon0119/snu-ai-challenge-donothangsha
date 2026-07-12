"""
Generate the Kaggle submission CSV from the LoRA fine-tuned model in
lora_finetune.py, by running inference over test.csv (which has no
ground-truth Answer column, unlike train.csv/lora_eval.py's val split).

Run on the same GPU box that has the trained adapter at ADAPTER_OUT_DIR.
"""
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info

from lora_finetune import load_model_and_processor, DATA_DIR
from lora_eval import parse_image_order

TEST_CSV = DATA_DIR / "test.csv"
TEST_IMG_DIR = DATA_DIR / "test"
SUBMISSION_OUT = Path("/workspace/submission.csv")


def build_test_prompt(row, image_dir):
    """Same template as build_example() in lora_finetune.py, minus the
    Answer-derived target text — test.csv has no Answer column."""
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
    return [{"role": "user", "content": content}]


def generate_prediction(model, processor, row, image_dir):
    messages = build_test_prompt(row, image_dir)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_image_order(output_text)


if __name__ == "__main__":
    assert TEST_CSV.exists(), f"{TEST_CSV} not found — check DATA_DIR/SNU_DATA_DIR"
    test_df = pd.read_csv(TEST_CSV)

    model, processor = load_model_and_processor()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    rows = []
    start = time.perf_counter()
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = generate_prediction(model, processor, row, TEST_IMG_DIR)
        rows.append((row["Id"], list(pred)))
        if (i + 1) % 50 == 0:
            print(f"inference {i + 1}/{len(test_df)}  elapsed={time.perf_counter() - start:.1f}s")

    with open(SUBMISSION_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Answer"])
        for id_, pred in rows:
            writer.writerow([id_, str(pred)])

    print(f"wrote {len(rows)} rows -> {SUBMISSION_OUT}")
