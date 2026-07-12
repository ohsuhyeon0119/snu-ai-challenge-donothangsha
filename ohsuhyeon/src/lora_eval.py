"""
Evaluate the LoRA-tuned model from lora_finetune.py on the held-out
lora_val_split.csv it produced, using the exact same exact-match metric
as validation.py — so Track A and Track B are directly comparable.

Run in the same Colab/Kaggle notebook session as lora_finetune.py (needs
the same model/adapter/processor already loaded), or reload the adapter
fresh with peft.PeftModel.from_pretrained(base_model, ADAPTER_OUT_DIR).
"""
import ast
import re

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info

from lora_finetune import build_example, TRAIN_IMG_DIR, ADAPTER_OUT_DIR


def parse_image_order(generated_text):
    """Inverse of build_example's target format: 'The correct order is [3, 1, 4, 2].'
    -> submission-style answer tuple (1, 4, 2, 3) meaning Input_1 is 1st temporally, etc.
    Falls back to identity order on any parse failure, matching the organizer
    baseline's fallback behavior."""
    match = re.search(r"\[([^\]]+)\]", generated_text)
    if match:
        try:
            image_order = ast.literal_eval(f"[{match.group(1)}]")
            if sorted(image_order) == [1, 2, 3, 4]:
                answer = [0, 0, 0, 0]
                for position, image_num in enumerate(image_order, start=1):
                    answer[image_num - 1] = position
                return tuple(answer)
        except Exception:
            pass
    return (1, 2, 3, 4)


def generate_prediction(model, processor, row, image_dir):
    messages, _ = build_example(row, image_dir)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_image_order(output_text)


def evaluate(model, processor):
    val_df = pd.read_csv("/content/lora_val_split.csv")
    preds, truths = [], []
    for _, row in val_df.iterrows():
        pred = generate_prediction(model, processor, row, TRAIN_IMG_DIR)
        truth = tuple(ast.literal_eval(row["Answer"]))
        preds.append(pred)
        truths.append(truth)

    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    acc = correct / len(truths)
    print(f"LoRA fine-tuned val accuracy: {acc:.4f} "
          f"(compare against Track A's val accuracy from train_classifier.py, "
          f"and the ~0.155 majority-class floor)")
    return acc
