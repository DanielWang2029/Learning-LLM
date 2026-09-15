"""A tiny MLP used to compare optimizers (kept deliberately minimal)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small 2-hidden-layer MLP. Its weight *matrices* are what Muon steps."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
