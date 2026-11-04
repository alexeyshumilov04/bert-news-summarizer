"""Train the BERT encoder + custom decoder summarizer."""
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.data import build_dataloaders, load_and_split
from src.model import BertSummarizer
from src.training import train_model
from src.utils import get_device, load_config, set_global_seed


def main(config_path: str) -> None:
    config = load_config(config_path)
    set_global_seed(config["seed"])
    device = get_device(config.get("device", "auto"))
    print(f"device: {device}")

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["bert_model_name"])

    train_ds, eval_ds = load_and_split(config["data"], tokenizer)
    train_loader, eval_loader = build_dataloaders(
        train_ds, eval_ds, config["data"]["batch_size"]
    )

    model = BertSummarizer(
        bert_model_name=model_cfg["bert_model_name"],
        hidden_size=model_cfg["hidden_size"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)

    tcfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=tcfg["epochs"],
        device=device,
        log_dir=tcfg["log_dir"],
        checkpoint_path=tcfg["checkpoint_path"],
        plot_path="artifacts/loss_curve.png",
    )
    tokenizer.save_pretrained(tcfg["tokenizer_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)