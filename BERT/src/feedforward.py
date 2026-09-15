"""Position-wise feed-forward network (Transformer §3.3).

BERT uses a GELU non-linearity between the two linear layers rather than the
original Transformer's ReLU (BERT paper, §3.1 / "GELU").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = GELU(x W1 + b1) W2 + b2, applied at each position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
