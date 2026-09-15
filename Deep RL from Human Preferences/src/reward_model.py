"""The learned reward model and the Bradley-Terry preference loss.

The reward model r̂_ψ maps a state observation to a scalar reward. A segment's
predicted return is the sum of r̂ over its states. Given a preference over two
segments, the probability the model assigns to the human's choice follows the
Bradley-Terry model (paper Eq. 1):

    P(σ¹ ≻ σ²) = exp Σ r̂(σ¹) / ( exp Σ r̂(σ¹) + exp Σ r̂(σ²) )
               = σ( Σ r̂(σ¹) − Σ r̂(σ²) )

and it is fit by minimizing the cross-entropy between these probabilities and
the human labels.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """A small MLP reward network over state features."""

    def __init__(self, num_states: int, hidden: int = 32) -> None:
        super().__init__()
        self.features = torch.eye(num_states)  # one-hot observation per state
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def state_reward(self, states: torch.Tensor) -> torch.Tensor:
        """r̂ for a batch of state ids."""
        return self.net(self.features[states]).squeeze(-1)

    def segment_return(self, seg: List[int]) -> torch.Tensor:
        """Predicted return Σ r̂(s) over one segment."""
        idx = torch.tensor(seg, dtype=torch.long)
        return self.state_reward(idx).sum()

    def reward_field(self) -> torch.Tensor:
        """r̂ for every state (for planning / visualization)."""
        with torch.no_grad():
            return self.net(self.features).squeeze(-1)


def preference_loss(
    ret_a: torch.Tensor, ret_b: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Bradley-Terry cross-entropy. labels[i]=1 ⇒ segment a preferred (paper Eq. 1)."""
    logit = ret_a - ret_b  # log-odds that a ≻ b
    return F.binary_cross_entropy_with_logits(logit, labels)
