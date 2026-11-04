"""Reference baseline: pretrained ruT5-small, no fine-tuning."""
from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_baseline(model_name: str, device):
    """Load a pretrained seq2seq baseline (e.g. cointegrated/rut5-small)."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def summarize(text: str, model, tokenizer, device, max_len: int = 50) -> str:
    """Convenience helper for a single string (mirrors evaluate_baseline in batch)."""
    import torch

    with torch.no_grad():
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        out = model.generate(**enc, max_new_tokens=max_len)
    return tokenizer.decode(out[0], skip_special_tokens=True)