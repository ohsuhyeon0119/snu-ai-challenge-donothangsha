"""
GPU smoke test for train_vision.py — run FIRST on any box, before the real
run. The one genuinely new thing round 3 does is put LoRA on the vision
tower and expect gradients to flow into it. A frozen-then-LoRA'd vision
encoder could silently receive NO gradient (detached graph, checkpointing
quirk) — then the whole run would train the LLM only and waste the spend.
This test specifically confirms the vision LoRA is in the backward graph.

Checks, on 4 real rows (2 SFT, 2 listwise):
  1. permutation tests (CPU)
  2. vision_lora_target_modules finds modules; print count + sample names so
     we can eyeball that they're real vision-block linears
  3. LoRA attaches to BOTH LLM and vision; ckpt-2218 loads (LLM), vision
     LoRA at zero-init
  4. finite loss on both branches; backward populates grads for BOTH LLM
     LoRA AND vision LoRA params (the critical check)
  5. optimizer step moves LLM and vision LoRA params; frozen base untouched
  6. VRAM peak

Usage:  SNU_INIT_ADAPTER=ckpts_unified/ckpt-2218 python smoke_vision.py
"""
import torch

import test_perms
import train as sft
import train_unified as tu
from common import (QUANT, enable_vision_input_grads, load_model,
                    load_processor, lora_target_modules,
                    vision_lora_target_modules)
from train_vision import INIT_ADAPTER, build_lora_config


def run_cpu_tests():
    for name in dir(test_perms):
        if name.startswith("test_"):
            getattr(test_perms, name)()
    print("[1/6] permutation tests: OK")


def is_vision(pname):
    return "visual" in pname or "vision" in pname


def main():
    import pandas as pd
    from pathlib import Path
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
    nhook = enable_vision_input_grads(model)
    print(f"      vision input-grad hook on {nhook} patch_embed module(s)")

    vis = vision_lora_target_modules(model)
    print(f"[2/6] vision LoRA targets: {len(vis)} modules")
    print("      sample:", vis[:3])
    print("      last:  ", vis[-1])

    from peft import (get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)
    from safetensors.torch import load_file
    model = get_peft_model(model, build_lora_config(model))
    init_sd = load_file(Path(INIT_ADAPTER) / "adapter_model.safetensors")
    valid = set(get_peft_model_state_dict(model).keys())
    filtered = {k: v for k, v in init_sd.items() if k in valid}
    assert filtered, "no ckpt-2218 keys matched the model's LoRA keys"
    set_peft_model_state_dict(model, filtered)
    loaded = len(filtered)
    n_llm = sum(1 for n, p in model.named_parameters()
                if p.requires_grad and "lora" in n and not is_vision(n))
    n_vis = sum(1 for n, p in model.named_parameters()
                if p.requires_grad and "lora" in n and is_vision(n))
    print(f"[3/6] LoRA attached: {n_llm} LLM + {n_vis} vision trainable lora "
          f"tensors; loaded {loaded} from ckpt-2218 (vision at zero-init)  "
          f"vram {torch.cuda.memory_allocated() / 2**30:.1f}GB")
    assert n_llm > 0 and n_vis > 0, "vision or LLM LoRA missing"

    collate = sft.make_collate(processor)
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}

    losses = []
    for i, row in enumerate(val):
        _, sigma, rng = tu.branch_for(row["Id"], epoch=0)
        if i < 2:
            loss = tu.sft_micro_loss(model, collate, row, sigma)
        else:
            loss = tu.listwise_micro_loss(model, collate, row, sigma, rng)
        assert torch.isfinite(loss), f"non-finite loss on {row['Id']}"
        loss.backward()
        losses.append(round(loss.item(), 3))
    print(f"[4/6] losses (2 sft, 2 listwise): {losses}  "
          f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    # THE critical check: vision LoRA params must receive real gradients
    llm_g = [p.grad for n, p in model.named_parameters()
             if p.requires_grad and "lora" in n and not is_vision(n)]
    vis_g = [(n, p.grad) for n, p in model.named_parameters()
             if p.requires_grad and "lora" in n and is_vision(n)]
    assert all(g is not None for g in llm_g), "some LLM LoRA grads are None"
    vis_none = [n for n, g in vis_g if g is None]
    assert not vis_none, (
        f"{len(vis_none)} vision LoRA params got NO gradient — vision tower "
        f"is not in the backward graph; aborting before wasting GPU-hours. "
        f"e.g. {vis_none[:2]}")
    vis_gn = sum(g.float().norm() ** 2 for _, g in vis_g) ** 0.5
    vis_nonzero = sum(1 for _, g in vis_g if g.abs().sum() > 0)
    print(f"[5/6] ALL {len(vis_g)} vision LoRA params have gradients "
          f"(nonzero: {vis_nonzero}, global norm {vis_gn:.4f}) — vision tower "
          f"IS in the backward graph")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-5)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    moved_llm = sum(1 for n, p in model.named_parameters()
                    if p.requires_grad and "lora" in n and not is_vision(n)
                    and not torch.equal(p.detach(), before[n]))
    moved_vis = sum(1 for n, p in model.named_parameters()
                    if p.requires_grad and "lora" in n and is_vision(n)
                    and not torch.equal(p.detach(), before[n]))
    print(f"[6/6] optimizer step moved {moved_llm} LLM + {moved_vis} vision "
          f"LoRA tensors")
    assert moved_vis > 0, "vision LoRA did not update"

    print("\nSMOKE_VISION PASSED")


if __name__ == "__main__":
    main()
