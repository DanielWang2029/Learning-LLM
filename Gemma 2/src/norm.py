"""RMSNorm (root-mean-square layer norm).

Gemma 2 normalizes with RMSNorm and, distinctively, applies it BOTH before and
after every attention / MLP sub-block ("pre-norm + post-norm", paper §2).  This
is the same RMSNorm as Zhang & Sennrich (2019): no mean-subtraction, a single
learned scale, and — as in the Gemma code — the scale is stored as ``(1 + w)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Stored as an offset from 1.0, matching the reference Gemma implementation.
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for stability, then rescale by (1 + weight).
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = norm * (1.0 + self.weight.float())
        return out.to(dtype)
