"""Encoder and decoder layers / stacks (Transformer §3.1)."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.dropout(F.relu(self.l1(x))))


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    """Causal self-attention, encoder–decoder cross-attention, feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
