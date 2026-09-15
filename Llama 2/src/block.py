"""A Llama 2 transformer block: pre-norm GQA, then pre-norm SwiGLU MLP.

Identical residual structure to LLaMA (pre-normalization), but the attention
sub-layer is Grouped-Query Attention (Llama 2 paper, Section 2.2):

    h = x + GQA(RMSNorm(x))
    y = h + SwiGLU(RMSNorm(h))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, attn: nn.Module, ffn: nn.Module) -> None:
        super().__init__()
        from .normalization import RMSNorm

        self.attn = attn
        self.ffn = ffn
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
