"""Scaled dot-product and multi-head attention (paper Section 3.2)."""

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
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V.

    Shapes (``h`` = number of heads, ``d_k`` = key/query dim per head):
        query: (batch, h, seq_q, d_k)
        key:   (batch, h, seq_k, d_k)
        value: (batch, h, seq_k, d_v)
        mask:  broadcastable to (batch, h, seq_q, seq_k); positions that
               are ``0``/``False`` are masked out with -inf before softmax.

    Returns the attended values and the attention weight matrix.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, value)
    return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention (paper Section 3.2.2).

    Projects queries, keys and values ``h`` times with distinct learned
    linear projections, applies scaled dot-product attention in parallel,
    concatenates and projects the result.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined projections for Q, K, V and the final output projection.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, h, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, h, d_k)
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # mask: (batch, seq_q, seq_k) -> add head dim for broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        attended, self.attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        return self.w_o(self._merge_heads(attended))
