import argparse
import ast
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FrameOrderingDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def parse_answer(value):
    return ast.literal_eval(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SNU_DATA_DIR", "data/snuaichallenge_data"))
    parser.add_argument("--adapter-dir", default=os.environ.get("SNU_ADAPTER_DIR", "outputs/paligemma_lora"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from snu_frame_ordering.orders import image_order_from_answer, target_text
    from snu_frame_ordering.paligemma_common import (
        encode_supervised,
        load_base_model,
        load_processor,
        row_contact_sheet,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "train.csv")
    rows = df.to_dict("records")
    random.shuffle(rows)

    processor = load_processor()
    model = load_base_model(load_4bit=True)
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(FrameOrderingDataset(rows), batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    for step, row in enumerate(tqdm(loader), start=1):
        image = row_contact_sheet(row, "train")
        answer = parse_answer(row["Answer"])
        image_order = image_order_from_answer(answer)
        completion = target_text(image_order)
        batch = encode_supervised(processor, image, row["Sentence"], completion).to(model.device)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step >= args.max_steps:
            break

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(adapter_dir)


if __name__ == "__main__":
    main()
