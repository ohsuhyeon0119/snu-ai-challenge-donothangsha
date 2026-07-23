import ast
import os
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

from snu_frame_ordering.contact_sheet import make_contact_sheet
from snu_frame_ordering.orders import (
    ALL_IMAGE_ORDERS,
    SIGMAS,
    aggregate_scores,
    answer_from_image_order,
    slot_order,
    target_text,
)
from snu_frame_ordering.paligemma_prompts import build_prompt


MODEL_NAME = os.environ.get("SNU_MODEL", "google/paligemma2-10b-pt-224")
DATA_DIR = Path(os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
ADAPTER_DIR = Path(os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
CONTACT_SHEET_SIZE = int(os.environ.get("SNU_CONTACT_SHEET_SIZE", "448"))
SCORE_CHUNK = int(os.environ.get("SNU_SCORE_CHUNK", "4"))


def model_device(model):
    return next(model.parameters()).device


def row_image_paths(row, split, sigma=None):
    sigma = sigma or SIGMAS[0]
    root = DATA_DIR / split / row["Id"]
    return [root / row[f"Input_{input_idx}"] for input_idx in sigma]


def open_rgb(path):
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def row_images(row, split, sigma=None):
    return [open_rgb(path) for path in row_image_paths(row, split, sigma=sigma)]


def row_contact_sheet(row, split, sigma=None):
    return make_contact_sheet(row_image_paths(row, split, sigma=sigma), size=CONTACT_SHEET_SIZE)


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_NAME)


def load_base_model(load_4bit=True):
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)



def lora_target_modules(model):
    keep = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    names = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        lname = name.lower()
        if leaf not in keep:
            continue
        if "vision" in lname or "visual" in lname or "projector" in lname:
            continue
        if not module.__class__.__name__.endswith(("Linear", "Linear4bit", "Linear8bitLt")):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("No language-decoder LoRA target modules found")
    return names

def prompt_for_images(images, sentence):
    image_count = len(images) if isinstance(images, (list, tuple)) else 1
    return "<image>" * image_count + "\n" + build_prompt(sentence)

def encode_prompt(processor, images, sentence, return_tensors="pt"):
    prompt = prompt_for_images(images, sentence)
    return processor(text=[prompt], images=[images], return_tensors=return_tensors)


def encode_supervised(processor, images, sentence, completion):
    prompt = prompt_for_images(images, sentence)
    return processor(
        text=[prompt],
        images=[images],
        suffix=[completion],
        return_tensors="pt",
        padding=True,
    )


def score_candidate_orders(model, processor, images, sentence, candidate_orders, chunk=SCORE_CHUNK):
    prompt = prompt_for_images(images, sentence)
    device = model_device(model)
    scores = []

    for start in range(0, len(candidate_orders), chunk):
        orders = candidate_orders[start:start + chunk]
        suffixes = [target_text(order) for order in orders]
        batch_images = [images for _ in suffixes]
        inputs = processor(
            text=[prompt] * len(suffixes),
            images=batch_images,
            suffix=suffixes,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].float()
        labels = inputs["labels"][:, 1:]
        valid = labels != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        token_scores = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        scores.extend((token_scores * valid).sum(dim=-1).detach().cpu().tolist())

    return scores


def score_row(model, processor, row, split, sigmas, chunk=SCORE_CHUNK):
    """Return per-sigma 24-candidate scores in canonical image-order space."""
    matrix = []
    for sigma in sigmas:
        images = row_images(row, split, sigma=sigma)
        slot_candidates = [slot_order(order, sigma) for order in ALL_IMAGE_ORDERS]
        scores = score_candidate_orders(
            model,
            processor,
            images,
            row["Sentence"],
            slot_candidates,
            chunk=chunk,
        )
        matrix.append(scores)
    return matrix


@torch.no_grad()
def predict_order(
    model,
    processor,
    row,
    split,
    tta_k=1,
    chunk=SCORE_CHUNK,
    prior_alpha=0.0,
    return_scores=False,
):
    sigmas = SIGMAS[:max(1, tta_k)]
    matrix = score_row(model, processor, row, split, sigmas, chunk=chunk)
    totals = aggregate_scores(matrix, prior_alpha=prior_alpha)

    ranked = sorted(range(len(totals)), key=lambda idx: totals[idx], reverse=True)
    best_idx = ranked[0]
    result = ALL_IMAGE_ORDERS[best_idx]
    if not return_scores:
        return result
    margin = totals[ranked[0]] - totals[ranked[1]] if len(ranked) > 1 else 0.0
    return result, {
        "score": totals[ranked[0]],
        "margin": margin,
        "tta_k": len(sigmas),
        "prior_alpha": prior_alpha,
    }


@torch.no_grad()
def score_orders(model, processor, row, split, chunk=SCORE_CHUNK, tta_k=1):
    return predict_order(model, processor, row, split, tta_k=tta_k, chunk=chunk)


ORDER_RE = re.compile(r"\[([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\]")


def parse_image_order(text):
    match = ORDER_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse image order from {text!r}")
    order = [int(match.group(idx)) for idx in range(1, 5)]
    if sorted(order) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid image order: {order}")
    return order


def parse_answer(value):
    return ast.literal_eval(value)


def submission_answer_from_image_order(image_order):
    return answer_from_image_order(image_order)
