"""Causal (masked) multi-head self-attention (GPT-2 §2; Transformer §3.2).

GPT-2 is a decoder-only Transformer: a lower-triangular mask forces position
*i* to attend only to positions *j <= i*, so the model factorizes the sequence
probability autoregressively as p(x) = Πᵢ p(xᵢ | x<ᵢ).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.w_o(out)
