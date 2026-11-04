"""Generate a summary from a raw text or from the eval split."""
import argparse

import torch
from transformers import AutoTokenizer

from src.data import load_and_split
from src.generation import generate
from src.model import BertSummarizer
from src.utils import get_device, load_config


def _load_model(config: dict, device: torch.device):
    model_cfg = config["model"]
    tcfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(tcfg["tokenizer_dir"])
    model = BertSummarizer(
        bert_model_name=model_cfg["bert_model_name"],
        hidden_size=model_cfg["hidden_size"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    ckpt = torch.load(tcfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


def main(config_path: str, text: str | None, dataset_index: int | None, strategy: str) -> None:
    config = load_config(config_path)
    device = get_device(config.get("device", "auto"))
    model, tokenizer = _load_model(config, device)

    if text is None:
        if dataset_index is None:
            raise ValueError("Provide either --text or --dataset-index")
        _, eval_ds = load_and_split(config["data"], tokenizer)
        example = eval_ds[dataset_index]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        print("reference:", tokenizer.decode(example["labels"], skip_special_tokens=True))
    else:
        enc = tokenizer(
            text,
            max_length=config["data"]["max_input_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

    gen_cfg = dict(config["generation"])
    gen_cfg["strategy"] = strategy
    summary = generate(model, input_ids, attention_mask, tokenizer, **gen_cfg)
    print(f"summary [{strategy}]: {summary[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument(
        "--strategy", choices=["greedy", "top_k", "nucleus", "beam"], default="greedy"
    )
    args = parser.parse_args()
    main(args.config, args.text, args.dataset_index, args.strategy)