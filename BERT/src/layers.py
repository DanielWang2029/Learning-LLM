"""Transformer encoder layer and stack (BERT §3.1; Transformer §3.1).

Each sub-layer uses a residual connection followed by layer normalization:
``LayerNorm(x + Sublayer(x))`` (post-norm, as in the original BERT).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import PositionwiseFeedForward


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderLayer(nn.Module):
    """Bidirectional self-attention followed by a position-wise feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """Stack of ``num_layers`` identical encoder layers."""

    def __init__(self, layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
