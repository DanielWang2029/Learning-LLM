"""A tiny MLP classifier used as the pretrained backbone (same as LoRA).

The base weights of this network are what QLoRA quantizes to 4 bits and freezes;
the LoRA adapters are added on top by :func:`src.qlora.inject_qlora`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_weight_numel(model: nn.Module) -> int:
    """Total elements in the Linear *weight* matrices (what gets quantized)."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n += m.weight.numel()
    return n


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).argmax(dim=-1)
    return (pred == y).float().mean().item()
