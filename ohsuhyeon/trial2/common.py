"""
Trial 2 shared code — model/processor loading, prompt construction, the exact
label<->answer mapping, and constrained permutation scoring.

Design rationale (see docs/superpowers/specs/2026-07-12-frame-ordering-trial2-design.md):
- The caption narrates events in temporal order, so the task is "align the 4
  frames to the caption's narrated sequence". We feed the WHOLE caption + all 4
  frames and let the VLM learn the alignment end-to-end (no caption splitting).
- Inference scores all 24 candidate orders by model likelihood and takes argmax,
  so the output is always a valid permutation (no parse failures).

Everything is env-var overridable so the same code runs on 2B (Stage A) and
7B (Stage B) unchanged:
    SNU_MODEL       default Qwen/Qwen2-VL-2B-Instruct
    SNU_DATA_DIR    default /workspace/data/snuaichallenge_data
    SNU_ADAPTER_DIR default /workspace/trial2_adapter
    SNU_VAL_SPLIT_CSV default /workspace/trial2_val_split.csv
    SNU_MAX_PIXELS  default 256*28*28  (caps vision tokens/img; ordering is coarse)
    SNU_MIN_PIXELS  default 128*28*28
"""
import ast
import os
from itertools import permutations
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_NAME = os.environ.get("SNU_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DATA_DIR = Path(os.environ.get("SNU_DATA_DIR", "/workspace/data/snuaichallenge_data"))
ADAPTER_DIR = Path(os.environ.get("SNU_ADAPTER_DIR", "/workspace/trial2_adapter"))
VAL_SPLIT_CSV = Path(os.environ.get("SNU_VAL_SPLIT_CSV", "/workspace/trial2_val_split.csv"))
TRAIN_IMG_DIR = DATA_DIR / "train"
TEST_IMG_DIR = DATA_DIR / "test"
MAX_PIXELS = int(os.environ.get("SNU_MAX_PIXELS", str(256 * 28 * 28)))
MIN_PIXELS = int(os.environ.get("SNU_MIN_PIXELS", str(128 * 28 * 28)))

# The 24 candidate image_orders. image_order[t] = 1-based Input index that sits
# at temporal position t+1 (i.e. "which image is shown 1st, 2nd, 3rd, 4th").
ALL_ORDERS = [list(p) for p in permutations([1, 2, 3, 4])]

INSTRUCTION = (
    'Caption: "{sentence}"\n'
    "The 4 images above (Image 1 to Image 4) are frames from a single video, "
    "given in shuffled order. The caption describes what happens in chronological "
    "order. Decide the correct chronological order of the images and output it as a "
    "list of the four image numbers, e.g. [3, 1, 4, 2]."
)


def image_order_from_answer(answer):
    """Answer[k] = temporal position of Input_(k+1)  ->  image_order (inverse)."""
    answer = list(answer)
    return [answer.index(pos) + 1 for pos in range(1, 5)]


def answer_from_image_order(image_order):
    """image_order -> submission Answer format (inverse of the above)."""
    answer = [0, 0, 0, 0]
    for t, img in enumerate(image_order, start=1):
        answer[img - 1] = t
    return answer


def target_text(image_order):
    """Completion string. str([3,1,4,2]) == '[3, 1, 4, 2]'."""
    return f"The correct order is {image_order}."


def build_messages(row, image_dir):
    imgs = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    content = []
    for i, f in enumerate(imgs):
        content.append({
            "type": "image",
            "image": str(Path(image_dir) / row["Id"] / f),
            "max_pixels": MAX_PIXELS,
            "min_pixels": MIN_PIXELS,
        })
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": INSTRUCTION.format(sentence=row["Sentence"])})
    return [{"role": "user", "content": content}]


def _model_class():
    name = MODEL_NAME.lower()
    if "2.5" in name or "2_5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    return Cls


def load_processor():
    return AutoProcessor.from_pretrained(
        MODEL_NAME, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
    )


def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    Cls = _model_class()
    model = Cls.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model


SCORE_CHUNK = int(os.environ.get("SNU_SCORE_CHUNK", "12"))


@torch.no_grad()
def score_orders(model, processor, messages, chunk=SCORE_CHUNK):
    """Return the best image_order (list) for one example by constrained scoring.

    All 24 completions share the identical prompt and have identical token
    length (every digit is a single character 1-4), so the batched sequences
    need no padding and the prompt length is constant -> clean, exact scoring.
    """
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    # prompt length (with expanded vision tokens) — identical for every candidate
    prompt_ids = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
    plen = prompt_ids["input_ids"].shape[1]

    full_texts = [prompt_text + target_text(io) for io in ALL_ORDERS]
    device = model.device
    logps = []
    for start in range(0, len(full_texts), chunk):
        texts = full_texts[start:start + chunk]
        images = image_inputs * len(texts)  # 4 imgs per text, in order
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits  # [B, L, V]
        # predict token t from logits at t-1: completion tokens live at [plen:L]
        comp_logits = logits[:, plen - 1:-1, :].float()
        comp_tokens = inputs["input_ids"][:, plen:]
        logp = torch.log_softmax(comp_logits, dim=-1)
        tok_logp = logp.gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, Lc]
        logps.extend(tok_logp.sum(dim=-1).tolist())
    best = int(max(range(len(logps)), key=lambda i: logps[i]))
    return ALL_ORDERS[best]


_ORDER_RE = __import__("re").compile(r"\[([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\]")


@torch.no_grad()
def generate_order(model, processor, messages, max_new_tokens=24):
    """Fast greedy-generation prediction (for the in-training monitor on big
    models, where 24-way constrained scoring is too slow). Falls back to
    identity on any malformed / non-permutation output."""
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = _ORDER_RE.search(text)
    if m:
        io = [int(m.group(i)) for i in range(1, 5)]
        if sorted(io) == [1, 2, 3, 4]:
            return io
    return [1, 2, 3, 4]


def exact_match(preds_answer, truths_answer):
    correct = sum(1 for p, t in zip(preds_answer, truths_answer) if list(p) == list(t))
    return correct / max(1, len(truths_answer))
