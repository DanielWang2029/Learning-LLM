"""RMSNorm — root-mean-square layer normalization (from the LLaMA recipe).

Llama 2 keeps LLaMA's RMSNorm pre-normalization (Llama 2 paper, Section 2.2):

    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

Cheaper than LayerNorm (no mean-subtraction, no bias) and just as effective.
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
