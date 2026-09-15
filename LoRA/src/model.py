"""A tiny MLP classifier used as the pretrained backbone to adapt.

Nothing here is LoRA-specific: it is an ordinary feed-forward network of
``nn.Linear`` layers.  LoRA is injected *after* pretraining by
:func:`src.lora.inject_lora`, which swaps each ``nn.Linear`` for a
``LoRALinear`` — exactly the "adapt a frozen pretrained model" setting the
paper targets (Section 1).
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


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).argmax(dim=-1)
    return (pred == y).float().mean().item()
