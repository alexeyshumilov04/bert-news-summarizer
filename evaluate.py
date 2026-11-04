"""Evaluate the custom summarizer and/or the ruT5-small baseline."""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.baseline import load_baseline
from src.data import build_t5_eval_dataloader, load_and_split
from src.metrics import evaluate_baseline, evaluate_summarizer
from src.model import BertSummarizer
from src.utils import get_device, load_config
from torch.utils.data import DataLoader


def _eval_custom(config: dict, device: torch.device, strategy: str | None) -> None:
    model_cfg = config["model"]
    tcfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(tcfg["tokenizer_dir"])
    _, eval_ds = load_and_split(config["data"], tokenizer)
    eval_loader = DataLoader(
        eval_ds, batch_size=config["data"]["batch_size"], shuffle=False
    )

    model = BertSummarizer(
        bert_model_name=model_cfg["bert_model_name"],
        hidden_size=model_cfg["hidden_size"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    ckpt = torch.load(tcfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    gen_cfg = dict(config["generation"])
    if strategy:
        gen_cfg["strategy"] = strategy

    metrics = evaluate_summarizer(model, eval_loader, tokenizer, device, gen_cfg)
    print(f"custom [{gen_cfg['strategy']}]:", metrics)


def _eval_baseline(config: dict, device: torch.device) -> None:
    model, tokenizer = load_baseline(config["baseline"]["model_name"], device)
    raw = load_dataset(config["data"]["dataset_name"], split=config["data"]["split"])
    loader = build_t5_eval_dataloader(raw, tokenizer, config["data"])
    metrics = evaluate_baseline(model, loader, tokenizer, device)
    print("baseline:", metrics)


def main(config_path: str, which: str, strategy: str | None) -> None:
    config = load_config(config_path)
    device = get_device(config.get("device", "auto"))

    if which in ("custom", "both"):
        _eval_custom(config, device, strategy)
    if which in ("baseline", "both"):
        _eval_baseline(config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--which", choices=["custom", "baseline", "both"], default="custom")
    parser.add_argument(
        "--strategy", choices=["greedy", "top_k", "nucleus", "beam"], default=None
    )
    args = parser.parse_args()
    main(args.config, args.which, args.strategy)