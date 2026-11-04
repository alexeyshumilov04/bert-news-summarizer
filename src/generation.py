"""Decoding strategies: greedy, top-k, nucleus (top-p), beam search."""
from __future__ import annotations

import torch

from src.model import BertSummarizer


def _select_next_token(
    logits: torch.Tensor,
    strategy: str,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """logits: [batch, vocab]. Returns [batch, 1] token ids."""
    if strategy == "greedy":
        return logits.argmax(dim=-1, keepdim=True)

    if strategy == "top_k":
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
        sample = torch.multinomial(top_probs, 1)
        return top_idx.gather(-1, sample)

    if strategy == "nucleus":
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sample = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, sample)

    raise ValueError(f"Unknown strategy: {strategy}")


def _strip_special(seq: list[int], tokenizer) -> list[int]:
    end = len(seq)
    for i, tok in enumerate(seq):
        if tok == tokenizer.sep_token_id:
            end = i
            break
    skip = {tokenizer.cls_token_id, tokenizer.pad_token_id}
    return [t for t in seq[:end] if t not in skip]


@torch.no_grad()
def generate(
    model: BertSummarizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    max_len: int = 50,
    strategy: str = "greedy",
    top_k: int = 50,
    top_p: float = 0.9,
    num_beams: int = 3,
) -> list[str]:
    """Generate one summary per item in the batch."""
    model.eval()
    device = input_ids.device

    if strategy == "beam":
        return _beam_generate(model, input_ids, attention_mask, tokenizer, max_len, num_beams)

    memory = model.encode(input_ids, attention_mask)
    batch = input_ids.size(0)
    decoder_ids = torch.full(
        (batch, 1), tokenizer.cls_token_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len):
        if finished.all():
            break
        log_probs = model.decode(memory, decoder_ids)[:, -1, :]
        next_token = _select_next_token(log_probs, strategy, top_k, top_p)
        finished |= next_token.squeeze(-1) == tokenizer.sep_token_id
        decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

    return [
        tokenizer.decode(_strip_special(seq, tokenizer), skip_special_tokens=True)
        for seq in decoder_ids.tolist()
    ]


@torch.no_grad()
def _beam_generate(
    model: BertSummarizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    max_len: int,
    num_beams: int,
    length_penalty: float = 0.6,
) -> list[str]:
    device = input_ids.device
    memory = model.encode(input_ids, attention_mask)
    results: list[str] = []

    def norm(item):
        seq, score = item
        return score / (seq.size(1) ** length_penalty)

    for b in range(input_ids.size(0)):
        mem = memory[b : b + 1]
        beams = [
            (torch.full((1, 1), tokenizer.cls_token_id, dtype=torch.long, device=device), 0.0)
        ]
        finished: list = []

        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == tokenizer.sep_token_id:
                    finished.append((seq, score))
                    continue
                log_probs = model.decode(mem, seq)[:, -1, :]
                top_lp, top_ids = torch.topk(log_probs, num_beams, dim=-1)
                for j in range(num_beams):
                    candidates.append(
                        (
                            torch.cat([seq, top_ids[:, j : j + 1]], dim=1),
                            score + top_lp[0, j].item(),
                        )
                    )
            if not candidates:
                break
            candidates.sort(key=norm, reverse=True)
            beams = candidates[:num_beams]

        pool = beams + finished
        if not pool:
            pool = [(torch.full((1, 1), tokenizer.cls_token_id, dtype=torch.long, device=device), 0.0)]
        pool.sort(key=norm, reverse=True)
        best_seq = pool[0][0][0].tolist()
        results.append(
            tokenizer.decode(_strip_special(best_seq, tokenizer), skip_special_tokens=True)
        )

    return results