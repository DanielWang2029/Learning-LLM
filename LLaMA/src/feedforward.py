"""Feed-forward networks: SwiGLU (LLaMA's choice) and a ReLU baseline.

LLaMA replaces the ReLU MLP with a SwiGLU variant (LLaMA paper, Section 2.2,
"SwiGLU activation function [PaLM]"; Shazeer 2020):

    SwiGLU(x) = ( SiLU(x·W_gate) ⊙ (x·W_up) ) · W_down

LLaMA also scales the hidden size to (2/3)·4·dim so the three-matrix SwiGLU has
a parameter count comparable to a two-matrix ReLU MLP with hidden size 4·dim.

The plain ``ReLUFFN`` is included only so the demo can ablate SwiGLU and show
the difference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden_dim(dim: int, multiple_of: int = 32) -> int:
    """LLaMA's rule of thumb: hidden = (2/3)·4·dim, rounded up to a multiple."""
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


class ReLUFFN(nn.Module):
    """Original Transformer MLP: max(0, x·W1)·W2 — used only for ablation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))
