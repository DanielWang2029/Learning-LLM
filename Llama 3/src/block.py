"""A Llama 3 transformer block: pre-norm GQA, then pre-norm SwiGLU (§3.1).

Pre-normalization (the residual carries the un-normalized signal):

    h = x + GroupedQueryAttention(RMSNorm(x))
    y = h + SwiGLU(RMSNorm(h))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, attn: GroupedQueryAttention, ffn: SwiGLU) -> None:
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
