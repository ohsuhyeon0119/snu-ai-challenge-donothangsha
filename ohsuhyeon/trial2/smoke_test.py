"""Quick pipeline sanity check before committing to a multi-hour run.

Loads the model, runs one masked-loss forward/backward, and one constrained
scoring pass on a few train rows, printing shapes/loss/acc. No checkpoints.
"""
import ast
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model

import common as C
from train import build_masked_inputs, LORA_TARGETS


def main():
    df = pd.read_csv(C.DATA_DIR / "train.csv").iloc[:5]
    processor = C.load_processor()
    model = C.load_base_model()
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    row = df.iloc[0]
    inputs, labels = build_masked_inputs(processor, row)
    n_supervised = int((labels != -100).sum())
    print(f"seq_len={inputs['input_ids'].shape[1]} supervised_tokens={n_supervised} "
          f"(should be small, ~10-15)", flush=True)
    loss = model(**inputs.to(model.device), labels=labels.to(model.device)).loss
    loss.backward()
    print(f"masked loss={loss.item():.4f} (finite, backward ok)", flush=True)

    model.eval(); model.config.use_cache = True
    t = time.perf_counter()
    correct = 0
    for _, r in df.iterrows():
        io = C.score_orders(model, C.load_processor() if False else processor,
                            C.build_messages(r, C.TRAIN_IMG_DIR))
        pred = C.answer_from_image_order(io)
        truth = ast.literal_eval(r["Answer"])
        correct += int(pred == truth)
    dt = (time.perf_counter() - t) / len(df)
    print(f"scored 5 rows, untuned acc={correct}/5, {dt:.2f}s/row "
          f"(x24 candidates). test-set ETA ~ {dt * 819 / 60:.1f}min", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
