"""SwiGLU feed-forward network.

PaLM replaces the ReLU MLP of the original Transformer with a SwiGLU variant
(PaLM paper, Section 2, "SwiGLU Activation"; Shazeer 2020):

    SwiGLU(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down

The gating branch (``SiLU(x W_gate)``) modulates the up-projection branch
element-wise before the down-projection. PaLM reported that SwiGLU
"significantly increased quality" over ReLU. Because SwiGLU uses three weight
matrices instead of two, the hidden size is scaled down so the parameter count
matches a standard MLP. No biases are used (see ``normalization.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
