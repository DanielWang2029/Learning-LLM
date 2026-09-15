"""RMSNorm — root-mean-square layer normalization.

LLaMA normalizes the *input* of each sub-layer (pre-normalization) using
RMSNorm instead of LayerNorm (LLaMA paper, Section 2.2, "Pre-normalization
[GPT3]"; Zhang & Sennrich 2019). RMSNorm drops the mean-subtraction and the
bias of LayerNorm, rescaling only by the root-mean-square of the activations:

    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

It is cheaper than LayerNorm and works just as well in practice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
