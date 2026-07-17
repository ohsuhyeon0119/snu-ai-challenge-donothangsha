"""
GPU smoke test — run FIRST on any freshly rented box, before training.
Validates the full stack in ~3 minutes:

  1. permutation math (CPU, exhaustive)
  2. model + processor load (bf16), VRAM readout
  3. fast KV-cache scorer == slow reference scorer (numerics + argmax)
  4. one masked-SFT collate + forward/backward through LoRA
     (checks supervised-token count == completion length)

Usage:  SNU_DATA_DIR=... python smoke_test.py
"""
import json

import pandas as pd
import torch

import test_perms
from common import (ALL_ORDERS, SIGMAS, TRAIN_IMG_DIR, build_messages,
                    image_order_from_answer, load_model, load_processor,
                    lora_target_modules, score_completions,
                    score_completions_reference, slot_order, target_text)


def run_cpu_tests():
    for name in dir(test_perms):
        if name.startswith("test_"):
            getattr(test_perms, name)()
    print("[1/4] permutation tests: OK")


def main():
    run_cpu_tests()

    df = pd.read_csv("data_work/clean_val.csv").head(3)
    rows = df.to_dict("records")

    processor = load_processor()
    model = load_model()
    model.eval()
    print(f"[2/4] model loaded: {model.config.model_type}, "
          f"vram {torch.cuda.memory_allocated() / 2**30:.1f}GB")

    # --- scorer equivalence on 2 rows x 2 sigmas ---
    for row in rows[:2]:
        for sigma in SIGMAS[:2]:
            messages = build_messages(row, TRAIN_IMG_DIR, sigma=sigma)
            comps = [target_text(slot_order(io, sigma)) for io in ALL_ORDERS]
            fast = score_completions(model, processor, messages, comps)
            ref = score_completions_reference(model, processor, messages, comps)
            fa, ra = max(range(24), key=fast.__getitem__), \
                max(range(24), key=ref.__getitem__)
            diff = max(abs(f - r) for f, r in zip(fast, ref))
            print(f"   row {row['Id']} sigma {sigma}: argmax fast={fa} "
                  f"ref={ra}, max|dlogp|={diff:.3f}")
            assert fa == ra, "fast scorer argmax != reference argmax"
            assert diff < 1.0, f"fast/ref logprob drift too large: {diff}"
    print(f"[3/4] KV-cache scorer == reference: OK  "
          f"(vram peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)")

    # --- one masked training step ---
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, task_type="CAUSAL_LM",
        target_modules=lora_target_modules(model)))

    from train import Rows, make_collate
    ds = Rows(pd.DataFrame(rows))
    ds.epoch = 1  # exercise the sigma-shuffle path
    collate = make_collate(processor)
    batch = collate([ds[i] for i in range(2)])

    tok = processor.tokenizer
    exp = len(tok(target_text([1, 2, 3, 4]), add_special_tokens=False).input_ids)
    sup = (batch["labels"] != -100).sum(dim=1)
    print(f"   supervised tokens per sample: {sup.tolist()} (expected {exp})")
    assert all(s == exp for s in sup.tolist()), "label mask wrong"

    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    assert torch.isfinite(out.loss), "loss not finite"
    out.loss.backward()
    g = [p.grad for p in model.parameters() if p.requires_grad and
         p.grad is not None]
    assert g, "no LoRA grads"
    print(f"[4/4] masked SFT forward/backward: OK  loss={out.loss.item():.3f}  "
          f"vram peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
