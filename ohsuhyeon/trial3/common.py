"""
Trial 3 shared code — Qwen3-VL-8B (bf16, no quantization), permutation math,
prompt construction, and the KV-cache-sharing 24-way constrained scorer.

Design: docs/superpowers/specs/2026-07-17-frame-ordering-trial3-design.md
Key ideas vs trial2:
  - bf16 end-to-end (train == inference precision). No 4-bit anywhere.
  - Scoring shares the prompt forward across the 24 candidates via KV cache
    (trial2 re-forwarded the full sequence 24x -> 5GB logit spikes, 16s/row).
  - Presentation-shuffle support: train-time augmentation and inference-time
    TTA both re-present the 4 frames in a permuted order sigma; all label
    mapping goes through the helpers here (unit-tested in test_perms.py).

Env vars:
    SNU_MODEL        default Qwen/Qwen3-VL-8B-Instruct
    SNU_DATA_DIR     default ./data   (contains train.csv/test.csv/train//test/)
    SNU_MAX_PIXELS   default 512*28*28 (>= native 640x360, so no downscaling)
    SNU_MIN_PIXELS   default 64*28*28
"""
import os
from itertools import permutations
from pathlib import Path

MODEL_NAME = os.environ.get("SNU_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
# "" = bf16 (8B fits a 3090 unquantized); "nf4" = 4-bit QLoRA path (32B).
# Train and inference MUST use the same value -> precision-consistent.
QUANT = os.environ.get("SNU_QUANT", "nf4")
DATA_DIR = Path(os.environ.get("SNU_DATA_DIR", "data"))
TRAIN_IMG_DIR = DATA_DIR / "train"
TEST_IMG_DIR = DATA_DIR / "test"
MAX_PIXELS = int(os.environ.get("SNU_MAX_PIXELS", str(512 * 28 * 28)))
MIN_PIXELS = int(os.environ.get("SNU_MIN_PIXELS", str(64 * 28 * 28)))

# All 24 candidate image_orders, canonical (Input-index) space.
# image_order[t] = 1-based Input index shown at temporal rank t+1.
ALL_ORDERS = [list(p) for p in permutations([1, 2, 3, 4])]

# Deterministic, diverse presentation orders for augmentation/TTA.
# sigma[j] = 1-based Input index displayed at slot j+1. First is identity.
SIGMAS = [
    [1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3], [3, 1, 4, 2],
    [2, 1, 4, 3], [4, 2, 3, 1], [3, 4, 1, 2], [1, 3, 2, 4],
]

INSTRUCTION = (
    'Caption: "{sentence}"\n'
    "The 4 images above (Image 1 to Image 4) are frames from a single video, "
    "given in shuffled order. The caption describes what happens in chronological "
    "order. Decide the correct chronological order of the images and output it as a "
    "list of the four image numbers, e.g. [3, 1, 4, 2]."
)

IM_END = "<|im_end|>"


# ---------------- permutation math (pure python, unit-tested) ----------------

def image_order_from_answer(answer):
    """answer[k] = temporal position of Input_(k+1)  ->  image_order (inverse)."""
    answer = list(answer)
    return [answer.index(pos) + 1 for pos in range(1, 5)]


def answer_from_image_order(image_order):
    """image_order -> submission Answer format (inverse of the above)."""
    answer = [0, 0, 0, 0]
    for t, img in enumerate(image_order, start=1):
        answer[img - 1] = t
    return answer


def inv_perm(sigma):
    """inv[v-1] = 1-based position of value v in sigma."""
    inv = [0, 0, 0, 0]
    for j, v in enumerate(sigma, start=1):
        inv[v - 1] = j
    return inv


def slot_order(image_order, sigma):
    """Canonical image_order -> order of presentation slots under sigma.
    slot_order[t] = which displayed slot (1-4) is temporally (t+1)-th."""
    inv = inv_perm(sigma)
    return [inv[img - 1] for img in image_order]


def image_order_from_slot_order(so, sigma):
    """Inverse of slot_order(): map a slot-space order back to canonical."""
    return [sigma[s - 1] for s in so]


def target_text(order):
    """Completion string for a (slot-space) order. All 24 have equal length."""
    return f"The correct order is {order}." + IM_END


# ---------------- prompt construction ----------------

def build_messages(row, image_dir, sigma=None):
    """row: mapping with Id, Input_1..4, Sentence. sigma: presentation order."""
    sigma = sigma or SIGMAS[0]
    files = [row[f"Input_{i}"] for i in range(1, 5)]
    content = []
    for j, inp in enumerate(sigma, start=1):
        content.append({
            "type": "image",
            "image": str(Path(image_dir) / row["Id"] / files[inp - 1]),
            "max_pixels": MAX_PIXELS,
            "min_pixels": MIN_PIXELS,
        })
        content.append({"type": "text", "text": f"\nImage {j}\n"})
    content.append({"type": "text", "text": INSTRUCTION.format(sentence=row["Sentence"])})
    return [{"role": "user", "content": content}]


# ---------------- model loading (bf16, no quantization) ----------------

def load_processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(
            MODEL_NAME, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    except TypeError:  # image processors that don't take pixel kwargs
        return AutoProcessor.from_pretrained(MODEL_NAME)


def load_model(device_map="cuda:0"):
    import torch
    from transformers import AutoModelForImageTextToText
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa",
              device_map=device_map)
    if QUANT == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **kw)


def lora_target_modules(model):
    """Exact names of LLM-decoder linear layers (vision tower excluded).
    Matches by class name so bnb Linear4bit (QLoRA) is included too."""
    keep = {"q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"}
    names = []
    for name, m in model.named_modules():
        if name.rsplit(".", 1)[-1] not in keep:
            continue
        if "visual" in name or "vision" in name:
            continue
        if not m.__class__.__name__.endswith(
                ("Linear", "Linear4bit", "Linear8bitLt")):
            continue
        names.append(name)
    assert names, "no LoRA target modules found - model layout changed?"
    return names


# ---------------- constrained scoring ----------------

def _prep_prompt(processor, messages):
    from qwen_vl_utils import process_vision_info
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return prompt_text, image_inputs


def encode_completions(tokenizer, completions):
    enc = [tokenizer(c, add_special_tokens=False).input_ids for c in completions]
    L = len(enc[0])
    assert all(len(e) == L for e in enc), "candidate lengths differ"
    return enc, L


def score_completions(model, processor, messages, completions):
    """logprob sum of each completion given the (image-bearing) prompt.

    Prompt is forwarded ONCE with use_cache=True; each candidate then runs a
    short continuation forward and the cache is cropped back. Peak logits are
    [1, L-1, V] instead of trial2's [chunk, full_len, V]."""
    import torch
    with torch.no_grad():
        prompt_text, image_inputs = _prep_prompt(processor, messages)
        inputs = processor(text=[prompt_text], images=image_inputs,
                           return_tensors="pt").to(model.device)
        plen = inputs["input_ids"].shape[1]
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values
        first_lp = out.logits[:, -1, :].float().log_softmax(-1)[0]  # [V]

        enc, L = encode_completions(processor.tokenizer, completions)
        scores = []
        for ids in enc:
            s = first_lp[ids[0]].item()
            if L > 1:
                t = torch.tensor([ids], device=model.device)
                attn = torch.ones((1, plen + L - 1), dtype=torch.long,
                                  device=model.device)
                o = model(input_ids=t[:, :-1], attention_mask=attn,
                          past_key_values=cache, use_cache=True,
                          cache_position=torch.arange(
                              plen, plen + L - 1, device=model.device))
                lp = o.logits.float().log_softmax(-1)  # [1, L-1, V]
                idx = torch.arange(L - 1, device=model.device)
                s += lp[0, idx, t[0, 1:]].sum().item()
                cache.crop(plen)
            scores.append(s)
    return scores


def score_completions_reference(model, processor, messages, completions):
    """Slow exact reference (full re-forward per candidate). Used by
    smoke_test.py to validate the fast path numerically."""
    import torch
    with torch.no_grad():
        prompt_text, image_inputs = _prep_prompt(processor, messages)
        pids = processor(text=[prompt_text], images=image_inputs,
                         return_tensors="pt")
        plen = pids["input_ids"].shape[1]
        scores = []
        for c in completions:
            inputs = processor(text=[prompt_text + c], images=image_inputs,
                               return_tensors="pt").to(model.device)
            logits = model(**inputs).logits.float()
            lp = logits[:, plen - 1:-1, :].log_softmax(-1)
            toks = inputs["input_ids"][:, plen:]
            idx = torch.arange(toks.shape[1], device=model.device)
            scores.append(lp[0, idx, toks[0]].sum().item())
    return scores


def predict_order(model, processor, row, image_dir, tta_k=1):
    """TTA over the first tta_k SIGMAS; returns canonical image_order."""
    totals = [0.0] * len(ALL_ORDERS)
    for sigma in SIGMAS[:max(1, tta_k)]:
        messages = build_messages(row, image_dir, sigma=sigma)
        comps = [target_text(slot_order(io, sigma)) for io in ALL_ORDERS]
        for i, s in enumerate(score_completions(model, processor, messages, comps)):
            totals[i] += s
    best = max(range(len(totals)), key=lambda i: totals[i])
    return ALL_ORDERS[best]


def exact_match(preds_answer, truths_answer):
    correct = sum(1 for p, t in zip(preds_answer, truths_answer)
                  if list(p) == list(t))
    return correct / max(1, len(truths_answer))
