"""Dataset loading, tokenization and DataLoader construction."""
from __future__ import annotations

from typing import Tuple

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


def load_and_split(config: dict, tokenizer) -> Tuple[Dataset, Dataset]:
    """Load the gazeta corpus, tokenize with a BERT-like tokenizer, and split."""
    ds = load_dataset(config["dataset_name"], split=config["split"])
    max_in = config["max_input_length"]
    max_tgt = config["max_target_length"]

    def preprocess(examples):
        inputs = tokenizer(
            examples["text"],
            max_length=max_in,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=max_tgt,
            padding="max_length",
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = ds.map(preprocess, batched=False)
    tokenized.set_format("torch")
    split = tokenized.train_test_split(test_size=config["test_size"], seed=42)
    return split["train"], split["test"]


def build_dataloaders(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader


def build_t5_eval_dataloader(raw_dataset: Dataset, tokenizer, config: dict) -> DataLoader:
    """Tokenize with T5 tokenizer for the baseline model; returns eval loader only."""
    max_in = config["max_input_length"]
    max_tgt = config["max_target_length"]

    def preprocess(examples):
        inputs = tokenizer(
            examples["text"],
            max_length=max_in,
            padding="max_length",
            truncation=True,
        )
        targets = tokenizer(
            examples["summary"],
            max_length=max_tgt,
            padding="max_length",
            truncation=True,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = raw_dataset.map(preprocess, batched=False)
    tokenized.set_format("torch")
    split = tokenized.train_test_split(test_size=config["test_size"], seed=42)
    return DataLoader(split["test"], batch_size=config["batch_size"], shuffle=False)