"""
Trial 4 shared code — Qwen3-VL-32B nf4 QLoRA, permutation math, prompt
construction, and the KV-cache-sharing 24-way constrained scorer.

Inherited from trial3 (LB 0.89); trial4 additions:
  - score_row(): per-sigma 24-candidate score matrix in CANONICAL space,
    so inference can dump raw scores and TTA-k / prior sweeps run offline.
  - Train-derived identity prior: train.csv answers are identity iff
    No_ordering (15.5%); presentation-shuffle augmentation flattens that
    prior out of the model, so predict_order can add alpha*log P_train(perm)
    back at decision time (alpha tuned on clean-val only).
  - predict_order aggregates by MEAN over sigmas (argmax-identical to the
    trial3 sum when alpha=0, keeps alpha's meaning independent of tta_k).

Env vars:
    SNU_MODEL        default Qwen/Qwen3-VL-32B-Instruct
    SNU_QUANT        "nf4" (default) | "" for bf16
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


# ---------------- train-derived identity prior ----------------

# Fraction of identity answers in the raw train.csv (1478/9535 rows, all of
# them No_ordering=True; the 23 non-identity perms are near-uniform there).
# Train-side statistic only — no test data involved.
TRAIN_P_IDENTITY = 1478 / 9535

IDENTITY_IDX = ALL_ORDERS.index([1, 2, 3, 4])  # == 0


def log_prior_vector(p_identity=TRAIN_P_IDENTITY):
    """log P_train(perm) aligned to ALL_ORDERS (canonical space)."""
    import math
    other = (1.0 - p_identity) / (len(ALL_ORDERS) - 1)
    return [math.log(p_identity) if i == IDENTITY_IDX else math.log(other)
            for i in range(len(ALL_ORDERS))]


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
        # keep the vision tower in bf16: vision encoders are the most
        # quantization-sensitive part and cost only ~1GB extra
        skip = ["lm_head"]
        if os.environ.get("SNU_SKIP_VISION_QUANT", "1") == "1":
            skip.append("visual")
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=skip)
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


def vision_lora_target_modules(model):
    """Linear layers INSIDE the vision tower's transformer blocks.

    Round 2 trained LoRA on the LLM decoder only, leaving the vision encoder
    frozen. Ordering 4 near-identical frames of the same scene is a fine-
    grained visual-difference task, and ~20.8% of captions carry no temporal
    cue words at all (pure-visual rows) — the frozen general-purpose encoder
    never adapted to that. This selects the vision blocks' linear layers so a
    LoRA can teach the encoder to attend to inter-frame differences.

    Matches any nn.Linear whose qualified name is under the visual module AND
    inside its transformer blocks (so patch-embed / merger / pos-embed stay
    frozen). Names vary across HF versions (attn.qkv vs q/k/v_proj, mlp.fc1/
    fc2 vs gate/up/down) so we match by position, not a fixed name set;
    smoke_vision.py prints the count + sample names to verify on the real
    model before spending GPU-hours."""
    names = []
    for name, m in model.named_modules():
        if "visual" not in name and "vision" not in name:
            continue
        if "block" not in name:  # exclude patch_embed / merger / pos_embed
            continue
        if not m.__class__.__name__.endswith(
                ("Linear", "Linear4bit", "Linear8bitLt")):
            continue
        names.append(name)
    assert names, ("no vision LoRA target modules found - inspect "
                   "model.named_modules() (vision layout differs by HF version)")
    return names


def enable_vision_input_grads(model):
    """Make the vision patch-embed output require grad, so gradient
    checkpointing tracks the backward graph THROUGH the vision blocks.

    HF's enable_input_require_grads / prepare_model_for_kbit_training only
    hooks the TEXT embeddings. The vision tower's input (pixel_values) is a
    leaf with requires_grad=False, so its checkpointed blocks fall out of the
    backward graph and any LoRA on them receives NO gradient (smoke_vision.py
    catches exactly this). Mirroring HF's fix for the text side, we register a
    forward hook on the vision patch-embed that flags its output as requiring
    grad; the requirement then propagates through every downstream block."""
    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.requires_grad_(True)
    n = 0
    for name, mod in model.named_modules():
        if name.endswith("visual.patch_embed"):
            mod.register_forward_hook(hook)
            n += 1
    assert n > 0, "vision patch_embed not found for input-grad hook"
    return n


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
                # NOTE: no attention_mask here on purpose. With a cache
                # present, Qwen3VL derives mrope positions from the FULL
                # mask length (past+new) but applies them to the new tokens
                # only -> shape crash. With mask=None it uses
                # arange(past, past+new) + rope_deltas, which is correct
                # (batch=1, no padding, causal attention over full KV).
                o = model(input_ids=t[:, :-1],
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


def score_row(model, processor, row, image_dir, sigmas):
    """Per-sigma 24-candidate logprob matrix, CANONICAL space.

    Returns [len(sigmas)][24]; entry [s][i] is the logprob of the completion
    for canonical ALL_ORDERS[i] presented under sigmas[s]. Index i always
    means the same canonical order regardless of sigma, so rows can be
    summed/averaged across sigmas directly."""
    mat = []
    for sigma in sigmas:
        messages = build_messages(row, image_dir, sigma=sigma)
        comps = [target_text(slot_order(io, sigma)) for io in ALL_ORDERS]
        mat.append(score_completions(model, processor, messages, comps))
    return mat


def aggregate_scores(mat, prior_alpha=0.0, p_identity=TRAIN_P_IDENTITY):
    """Mean over sigmas + alpha * log-prior -> 24 totals (canonical space).

    Pure math (no model), shared by predict_order and the offline analyzer
    so tuned alphas transfer exactly."""
    k = len(mat)
    totals = [sum(col) / k for col in zip(*mat)]
    if prior_alpha:
        lp = log_prior_vector(p_identity)
        totals = [t + prior_alpha * p for t, p in zip(totals, lp)]
    return totals


def predict_order(model, processor, row, image_dir, tta_k=1,
                  prior_alpha=0.0, return_scores=False):
    """TTA over the first tta_k SIGMAS; returns canonical image_order.

    prior_alpha > 0 adds the train-derived identity prior at decision time.
    return_scores=True also returns the raw per-sigma score matrix (for
    offline TTA-k / alpha sweeps)."""
    mat = score_row(model, processor, row, image_dir, SIGMAS[:max(1, tta_k)])
    totals = aggregate_scores(mat, prior_alpha=prior_alpha)
    best = max(range(len(totals)), key=lambda i: totals[i])
    return (ALL_ORDERS[best], mat) if return_scores else ALL_ORDERS[best]


def exact_match(preds_answer, truths_answer):
    correct = sum(1 for p, t in zip(preds_answer, truths_answer)
                  if list(p) == list(t))
    return correct / max(1, len(truths_answer))


# ---------------- listwise stage-2: hard-negative candidate sets ----------------

def kendall_distance(a, b):
    """Pairwise-inversion distance between two canonical image_orders (0..6)."""
    pa = {v: i for i, v in enumerate(a)}
    pb = {v: i for i, v in enumerate(b)}
    return sum(1 for i in range(1, 5) for j in range(i + 1, 5)
               if (pa[i] - pa[j]) * (pb[i] - pb[j]) < 0)


def adjacent_transpositions(order):
    """The 3 canonical orders one adjacent swap away from `order`
    (kendall_distance == 1 to `order`, always exactly 3 for n=4)."""
    out = []
    for i in range(3):
        o = list(order)
        o[i], o[i + 1] = o[i + 1], o[i]
        out.append(o)
    return out


def hard_candidate_set(truth, n_random=2, rng=None):
    """truth + its 3 adjacent-swap perms + n_random other perms, deduped.
    Returns canonical image_orders with truth ALWAYS first (index 0)."""
    import random
    rng = rng or random
    truth = list(truth)
    seen = {tuple(truth)}
    cands = [truth]
    for o in adjacent_transpositions(truth):
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            cands.append(o)
    pool = [o for o in ALL_ORDERS if tuple(o) not in seen]
    rng.shuffle(pool)
    cands.extend(pool[:n_random])
    return cands


def broadened_candidate_set(truth, n_adjacent=3, n_near=2, n_random=1, rng=None):
    """truth + kendall=1 (adjacent) + kendall={2,3} (near) + uniform-random.

    hard_candidate_set only targets kendall=1 confusions (36% of ckpt-300's
    val errors); kendall 2-3 combined were ~30% more and went untouched by
    round-1 listwise. This widens the competitor set to cover both, at the
    same total candidate-count order of magnitude (default 7 vs 6).
    Mahonian counts for n=4 (distance:count) are 0:1 1:3 2:5 3:6 4:5 5:3 6:1,
    so the kendall={2,3} pool always has 11 members to draw n_near from.
    Returns canonical image_orders with truth ALWAYS first (index 0)."""
    import random
    rng = rng or random
    truth = list(truth)
    seen = {tuple(truth)}
    cands = [truth]

    adj = adjacent_transpositions(truth)
    rng.shuffle(adj)
    for o in adj[:n_adjacent]:
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            cands.append(o)

    near_pool = [o for o in ALL_ORDERS if tuple(o) not in seen
                 and kendall_distance(truth, o) in (2, 3)]
    rng.shuffle(near_pool)
    for o in near_pool[:n_near]:
        t = tuple(o)
        if t not in seen:
            seen.add(t)
            cands.append(o)

    rand_pool = [o for o in ALL_ORDERS if tuple(o) not in seen]
    rng.shuffle(rand_pool)
    cands.extend(rand_pool[:n_random])
    return cands


def candidate_logprob_sum(logits, labels):
    """Per-example summed log-prob of the completion span, from a standard
    batched causal-LM forward's logits + masked labels (labels[b] == -100
    outside the completion, matching train.py's make_collate() convention).

    Used by listwise training: unlike the KV-cache-sharing scorer (which
    only saves memory under no_grad — under autograd each candidate's cache
    extension must be RETAINED for backward, so 6 candidates OOM'd a 32B
    model at ~93GB even with checkpointing off), this reuses the plain
    batched-forward-with-gradient-checkpointing path already proven safe by
    train.py's masked SFT (measured ~48-65GB peak for mb4-8 there).

    Standard causal shift: logits[:, i] predicts the token at labels[:, i+1]
    (same convention already validated in score_completions_reference)."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    logp = shift_logits.log_softmax(-1)
    mask = shift_labels != -100
    safe = shift_labels.clone()
    safe[~mask] = 0
    tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp * mask.float()
    return tok_lp.sum(dim=1)
