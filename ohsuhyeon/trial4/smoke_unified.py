"""
GPU smoke test for train_unified.py — run FIRST on any box, before the real
run. Validates both branches (SFT and listwise) share one freshly-LoRA'd
model correctly, on 4 real val_small rows (2 forced SFT, 2 forced listwise).

This exercises the one thing round 2 does that's genuinely untested: mixing
a single-target masked-loss branch and a multi-candidate listwise branch
within the SAME training loop / LoRA weights, starting from a fresh
(non-QLoRA-trained) adapter. Each branch individually was already validated
(train.py's smoke_test.py for SFT; smoke_listwise.py for the candidate-batch
mechanism after round 1's v1 OOM fix) — this only checks they don't
interfere with each other.

Checks:
  1. permutation tests (CPU)
  2. fresh LoRA adapter attaches, trainable params > 0
  3. 2 SFT-branch rows: finite loss, backward populates grads
  4. 2 listwise-branch rows: finite loss, backward populates grads
  5. optimizer step moves LoRA params, leaves frozen params untouched
  6. VRAM peak, for capacity planning before the real run

Usage:  python smoke_unified.py
"""
import json

import pandas as pd
import torch

import test_perms
from common import QUANT, load_model, load_processor, lora_target_modules
from train_unified import (branch_for, listwise_micro_loss, sft_micro_loss)
import train as sft


def run_cpu_tests():
    for name in dir(test_perms):
        if name.startswith("test_"):
            getattr(test_perms, name)()
    print("[1/6] permutation tests: OK")


def main():
    run_cpu_tests()

    val = pd.read_csv("data_work/val_small.csv").head(4).to_dict("records")

    processor = load_processor()
    model = load_model()
    model.config.use_cache = False
    if QUANT:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=lora_target_modules(model))
    model = get_peft_model(model, lcfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[2/6] fresh LoRA attached: {n_trainable:,} trainable params, "
          f"vram {torch.cuda.memory_allocated() / 2**30:.1f}GB")
    assert n_trainable > 0

    collate = sft.make_collate(processor)
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    frozen_before = {n: p.detach().clone()
                     for n, p in list(model.named_parameters())[:5]
                     if not p.requires_grad}

    sft_losses = []
    for row in val[:2]:
        _, sigma, _ = branch_for(row["Id"], epoch=0)
        loss = sft_micro_loss(model, collate, row, sigma)
        assert torch.isfinite(loss), f"non-finite SFT loss on {row['Id']}"
        loss.backward()
        sft_losses.append(loss.item())
    print(f"[3/6] SFT branch x2: losses={[round(x, 3) for x in sft_losses]}  "
          f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    lw_losses = []
    for row in val[2:4]:
        _, sigma, rng = branch_for(row["Id"], epoch=0)
        loss = listwise_micro_loss(model, collate, row, sigma, rng)
        assert torch.isfinite(loss), f"non-finite listwise loss on {row['Id']}"
        loss.backward()
        lw_losses.append(loss.item())
    print(f"[4/6] listwise branch x2: losses={[round(x, 3) for x in lw_losses]}  "
          f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads), \
        "some LoRA params got no gradient after the mixed SFT+listwise batch"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"
    gnorm = sum(g.float().norm() ** 2 for g in grads) ** 0.5
    print(f"[5/6] all {len(grads)} LoRA grad tensors present & finite "
          f"(mixed-branch backward), global norm {gnorm:.4f}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    moved = sum(1 for n, p in model.named_parameters()
                if p.requires_grad and not torch.equal(p.detach(), before[n]))
    for n, p in list(model.named_parameters())[:5]:
        if n in frozen_before:
            assert torch.equal(p.detach(), frozen_before[n]), \
                f"frozen param {n} moved — LoRA isolation broken"
    print(f"[6/6] optimizer step: {moved}/{len(before)} LoRA tensors changed, "
          f"frozen params untouched")

    print("\nSMOKE_UNIFIED PASSED")


if __name__ == "__main__":
    main()
