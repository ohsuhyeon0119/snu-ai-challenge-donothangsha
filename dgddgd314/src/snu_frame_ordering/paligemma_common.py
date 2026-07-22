import ast
import os
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

from snu_frame_ordering.contact_sheet import make_contact_sheet
from snu_frame_ordering.orders import ALL_IMAGE_ORDERS, answer_from_image_order, target_text
from snu_frame_ordering.paligemma_prompts import build_prompt


MODEL_NAME = os.environ.get("SNU_MODEL", "google/paligemma-3b-pt-448")
DATA_DIR = Path(os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
ADAPTER_DIR = Path(os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
CONTACT_SHEET_SIZE = int(os.environ.get("SNU_CONTACT_SHEET_SIZE", "448"))
SCORE_CHUNK = int(os.environ.get("SNU_SCORE_CHUNK", "8"))


def row_image_paths(row, split):
    root = DATA_DIR / split / row["Id"]
    return [root / row[f"Input_{idx}"] for idx in range(1, 5)]


def row_contact_sheet(row, split):
    return make_contact_sheet(row_image_paths(row, split), size=CONTACT_SHEET_SIZE)


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_NAME)


def load_base_model(load_4bit=True):
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    }
    return PaliGemmaForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)


def encode_prompt(processor, image, sentence, return_tensors="pt"):
    prompt = build_prompt(sentence)
    return processor(text=prompt, images=image, return_tensors=return_tensors)


def encode_supervised(processor, image, sentence, completion):
    prompt = build_prompt(sentence)
    full_text = prompt + "\n" + completion
    prompt_inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = processor(text=full_text, images=image, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    inputs["labels"] = labels
    return inputs


@torch.no_grad()
def score_orders(model, processor, image, sentence, chunk=SCORE_CHUNK):
    prompt = build_prompt(sentence)
    prompt_inputs = processor(text=prompt, images=image, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    device = model.device
    logps = []

    for start in range(0, len(ALL_IMAGE_ORDERS), chunk):
        orders = ALL_IMAGE_ORDERS[start:start + chunk]
        texts = [prompt + "\n" + target_text(order) for order in orders]
        images = [image] * len(texts)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits
        comp_logits = logits[:, prompt_len - 1:-1, :].float()
        comp_tokens = inputs["input_ids"][:, prompt_len:]
        logp = torch.log_softmax(comp_logits, dim=-1)
        tok_logp = logp.gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)
        mask = inputs["attention_mask"][:, prompt_len:]
        logps.extend((tok_logp * mask).sum(dim=-1).tolist())

    best = int(max(range(len(logps)), key=lambda idx: logps[idx]))
    return ALL_IMAGE_ORDERS[best]


ORDER_RE = re.compile(r"\[([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\]")


def parse_image_order(text):
    match = ORDER_RE.search(text)
    order = [int(match.group(idx)) for idx in range(1, 5)]
    return order


def parse_answer(value):
    return ast.literal_eval(value)


def submission_answer_from_image_order(image_order):
    return answer_from_image_order(image_order)
