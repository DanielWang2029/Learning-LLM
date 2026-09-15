"""RMSNorm — root-mean-square layer normalization (Llama 3, §3.1).

Llama 3 keeps the Llama-family pre-normalization: each sub-layer normalizes its
*input* with RMSNorm (Zhang & Sennrich 2019), which drops LayerNorm's
mean-subtraction and bias and rescales only by the root-mean-square:

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
