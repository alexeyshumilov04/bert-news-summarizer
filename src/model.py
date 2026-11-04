"""BERT encoder + custom Transformer decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class DecoderLayer(nn.Module):
    """One decoder block: masked self-attention, cross-attention, feed-forward."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask=None) -> torch.Tensor:
        residual = x
        x_t = x.transpose(0, 1)
        x_t, _ = self.self_attn(x_t, x_t, x_t, attn_mask=tgt_mask)
        x = self.norm1(residual + self.dropout(x_t.transpose(0, 1)))

        residual = x
        x_t = x.transpose(0, 1)
        mem_t = memory.transpose(0, 1)
        x_t, _ = self.cross_attn(x_t, mem_t, mem_t)
        x = self.norm2(residual + self.dropout(x_t.transpose(0, 1)))

        residual = x
        x = self.norm3(residual + self.dropout(self.ffn(x)))
        return x


class BertSummarizer(nn.Module):
    """Encoder-decoder summarizer with a BERT encoder and a from-scratch decoder."""

    def __init__(
        self,
        bert_model_name: str = "deepvk/bert-base-uncased",
        hidden_size: int = 768,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.bert.config.vocab_size, hidden_size)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, dropout) for _ in range(num_decoder_layers)]
        )
        self.fc_out = nn.Linear(hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.embedding.apply(self._init_weights)
        for layer in self.decoder_layers:
            layer.apply(self._init_weights)
        self.fc_out.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def causal_mask(size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def decode(self, memory: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(decoder_input_ids)
        mask = self.causal_mask(x.size(1), x.device)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=mask)
        return self.log_softmax(self.fc_out(x))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(input_ids, attention_mask)
        return self.decode(memory, decoder_input_ids)