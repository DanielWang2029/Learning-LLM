"""Group-relative advantage (the GRPO baseline used to train the search policy).

For each question we sample a group of rollouts and use the group's own mean
reward as the baseline — no value network. Above-average rollouts are reinforced,
below-average ones suppressed.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Standardize rewards within a group -> advantages. ``rewards`` shape (G,)."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
