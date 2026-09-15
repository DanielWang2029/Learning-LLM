"""A *thinking* reasoner and test-time compute scaling by self-consistency.

Gemini 2.5's documented headline is a thinking model that gets more accurate when
given more test-time compute. The standard, well-documented mechanism for that is
**parallel sampling + majority vote** (self-consistency): sample several
independent reasoning paths and answer with the most common result. More sampled
paths = more test-time compute = higher accuracy, saturating.

Here a small MLP (`StepReasoner`) learns one arithmetic operation at a time. At
test time we decode each reasoning path *stochastically* (temperature sampling
with a little logit noise, standing in for an imperfect thinker), so different
paths occasionally slip on different steps. Aggregating them with a majority vote
cancels those independent slips.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import MOD, N_OPS


class StepReasoner(nn.Module):
    """Learns the single-operation transition ``(value, op) -> value``."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MOD + N_OPS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, MOD),
        )

    def step_logits(self, value: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.one_hot(value, MOD).float(),
                       F.one_hot(op, N_OPS).float()], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample_path(self, start, op_ids, temperature, noise, generator):
        """Decode ONE stochastic reasoning path; returns the final value.

        Temperature sampling plus small Gaussian logit noise makes each path an
        independent, occasionally-fallible attempt — exactly what self-consistency
        needs to aggregate.
        """
        v = torch.tensor(start)
        for o in op_ids:
            logits = self.step_logits(v.unsqueeze(0), torch.tensor([o])).squeeze(0)
            logits = logits / temperature + noise * torch.randn(
                MOD, generator=generator)
            probs = F.softmax(logits, dim=-1)
            v = torch.multinomial(probs, 1, generator=generator).squeeze(0)
        return int(v)

    @torch.no_grad()
    def self_consistency(self, start, op_ids, k, temperature, noise, generator):
        """Sample ``k`` reasoning paths and return (majority answer, vote counts)."""
        votes = torch.zeros(MOD, dtype=torch.long)
        for _ in range(k):
            votes[self.sample_path(start, op_ids, temperature, noise, generator)] += 1
        return int(votes.argmax()), votes.tolist()
