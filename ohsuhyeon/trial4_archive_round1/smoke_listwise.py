"""
GPU smoke test for train_listwise.py's core assumption BEFORE spending
mine_hard.py / train_listwise.py GPU-hours on it: that the batched-forward
listwise loss (checkpointed, reusing train.py's make_collate) backprops
correctly into the LoRA adapter loaded from ckpt-400, and fits in VRAM.

v1 of this design shared one prompt forward across candidates via a KV
cache kept differentiable (no gradient checkpointing, since checkpointing
forces use_cache=False). It OOM'd a 32B model at ~93GB on the FIRST row —
under autograd, cache-sharing doesn't free memory the way it does under
no_grad (inference), because every candidate's cache-extension tensors
must stay alive for backward. v2 (this test) uses an ordinary checkpointed
batched forward instead — this smoke test exists specifically to catch a
repeat of that failure mode (or any new one) in minutes, not hours.

Checks, on 2 real val rows:
  1. ckpt-400 loads as a TRAINABLE PeftModel (grad checkpointing enabled)
  2. listwise_loss() returns a finite loss and its backward() populates
     LoRA grads (not None, all finite)
  3. an optimizer step moves LoRA params but not frozen ones
  4. VRAM peak, printed for capacity planning before the real run

Usage:  python smoke_listwise.py
"""
import torch

from common import QUANT, load_model, load_processor
import train as sft
from train_listwise import BASE_ADAPTER, listwise_loss


def main():
    import pandas as pd
    val = pd.read_csv("data_work/clean_val.csv").head(2).to_dict("records")

    processor = load_processor()
    model = load_model()
    model.config.use_cache = False  # required for gradient checkpointing
    if QUANT:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1/4] adapter loaded trainable: {BASE_ADAPTER}, "
          f"{n_trainable:,} trainable params, "
          f"vram {torch.cuda.memory_allocated() / 2**30:.1f}GB")
    assert n_trainable > 0

    collate = sft.make_collate(processor)
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    frozen_before = {n: p.detach().clone()
                     for n, p in list(model.named_parameters())[:5]
                     if not p.requires_grad}

    losses, accs = [], []
    for row in val:
        loss, correct = listwise_loss(model, processor, collate, row, epoch=0)
        assert torch.isfinite(loss), f"non-finite loss on row {row['Id']}"
        loss.backward()
        losses.append(loss.item())
        accs.append(correct)
    print(f"[2/4] listwise_loss on {len(val)} rows: "
          f"losses={[round(x, 3) for x in losses]} top1={accs}  "
          f"vram_peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads), \
        "some LoRA params got no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"
    gnorm = sum(g.float().norm() ** 2 for g in grads) ** 0.5
    print(f"[3/4] all {len(grads)} LoRA grad tensors present & finite, "
          f"global norm {gnorm:.4f}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    moved = sum(1 for n, p in model.named_parameters()
                if p.requires_grad and not torch.equal(p.detach(), before[n]))
    print(f"[4/4] optimizer step: {moved}/{len(before)} LoRA tensors changed")
    assert moved > 0, "optimizer step had no effect"
    for n, p in list(model.named_parameters())[:5]:
        if n in frozen_before:
            assert torch.equal(p.detach(), frozen_before[n]), \
                f"frozen param {n} moved — LoRA isolation broken"

    print("\nSMOKE_LISTWISE PASSED")


if __name__ == "__main__":
    main()
