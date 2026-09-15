"""Encoder and decoder layers / stacks (paper Section 3.1).

Each sub-layer uses a residual connection followed by layer normalization:
``LayerNorm(x + Sublayer(x))``.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderLayer(nn.Module):
    """Self-attention sub-layer followed by a position-wise feed-forward."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder attention, feed-forward."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        cross = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class Encoder(nn.Module):
    """Stack of ``num_layers`` identical encoder layers."""

    def __init__(self, layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Stack of ``num_layers`` identical decoder layers."""

    def __init__(self, layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
