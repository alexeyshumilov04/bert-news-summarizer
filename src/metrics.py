"""Automatic summarization metrics and evaluation loop."""
from __future__ import annotations

import evaluate
import torch
from torch.utils.data import DataLoader

from src.generation import generate

_rouge = evaluate.load("rouge")
_bleu = evaluate.load("bleu")
_bertscore = evaluate.load("bertscore")


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge = _rouge.compute(predictions=predictions, references=references)
    bleu = _bleu.compute(predictions=predictions, references=references)
    bert = _bertscore.compute(predictions=predictions, references=references, lang="ru")

    def _mean(xs):
        return sum(xs) / len(xs)

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu["bleu"],
        "bertscore_precision": _mean(bert["precision"]),
        "bertscore_recall": _mean(bert["recall"]),
        "bertscore_f1": _mean(bert["f1"]),
    }


@torch.no_grad()
def evaluate_summarizer(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    generation_config: dict,
) -> dict:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        references.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
        predictions.extend(
            generate(model, input_ids, attention_mask, tokenizer, **generation_config)
        )

    return compute_metrics(predictions, references)


@torch.no_grad()
def evaluate_baseline(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50,
) -> dict:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        references.extend(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return compute_metrics(predictions, references)