"""Group-Relative Policy Optimization (GRPO), the algorithm behind DeepSeek-R1.

GRPO removes the value network of PPO. For each prompt it samples a **group** of
G outputs, and uses the group's own mean reward as the baseline:

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

The policy is then nudged to make above-average outputs more likely and
below-average ones less likely (a REINFORCE-style policy gradient, no critic):

    loss = - mean_i [ advantage_i * logπ(output_i | prompt) ]

That is the entire update implemented here.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Standardize rewards within a group -> advantages (the GRPO baseline).

    ``rewards`` shape (num_prompts, group_size); returns the same shape.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Policy-gradient loss given per-sample log-probs and advantages (flat)."""
    return -(advantages.detach() * logprobs).mean()
