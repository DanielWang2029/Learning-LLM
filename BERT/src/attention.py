"""Bidirectional multi-head self-attention (BERT §3.1; Transformer §3.2).

BERT is an *encoder*: every token attends to every other token in both
directions. There is no causal mask — the whole point of BERT is that each
representation is conditioned on both the left and right context at once.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V.

    ``mask`` (broadcastable to the score matrix) marks *padding* positions with
    ``0``/``False`` so they receive no attention weight. Unlike a decoder there
    is no triangular (causal) mask here.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, value), attn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with ``num_heads`` parallel heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dim
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        attended, self.attn_weights = scaled_dot_product_attention(q, k, v, mask)
        return self.w_o(self._merge(self.dropout(attended)))
