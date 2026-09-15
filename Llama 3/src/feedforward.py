"""SwiGLU feed-forward network — Llama 3, §3.1.

Llama 3 keeps the SwiGLU MLP (Shazeer 2020) of the Llama family:

    SwiGLU(x) = ( SiLU(x·W_gate) ⊙ (x·W_up) ) · W_down

The hidden size follows the (2/3)·4·dim rule so the three-matrix SwiGLU has a
parameter count comparable to a two-matrix ReLU MLP of hidden size 4·dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden_dim(dim: int, multiple_of: int = 32) -> int:
    hidden = int(2 * (4 * dim) / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
