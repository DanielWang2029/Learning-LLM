"""Minimal multi-head attention, used for LM self-attention and cross-attention.

Kept deliberately small and readable — this is the same scaled dot-product
attention from "Attention Is All You Need" that Flamingo builds on top of.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Supports both self-attention (query == key/value source) and cross-attention
    (queries from the language model, keys/values from the visual tokens).
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key_value))
        v = self._split(self.w_v(key_value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()

        out = torch.matmul(attn, v)                       # (b, h, t_q, d_k)
        b, _, t_q, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t_q, self.num_heads * self.d_k)
        return self.w_o(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (d_model -> d_ff -> d_model)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
