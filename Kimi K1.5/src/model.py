"""A tiny two-headed policy for RL fine-tuning.

To study Kimi k1.5's *length penalty* in isolation, the policy makes two
decisions for each problem:

* an **answer** — the reasoning output, produced by a small network over the
  problem ``(a, b)`` (this is the part that must become *correct*), and
* a **chain length** — how many scratch/"think" tokens ``T`` to emit before the
  answer (the "thinking budget"), drawn from a length policy.

Separating the two heads lets the length penalty act on the *thinking budget*
directly, exactly as Kimi's reward does, without it being entangled with the
trivial "answer as soon as possible" shortcut that dominates when a single
autoregressive head both thinks and answers.

Crucially, thinking is *useful up to a point*: the answer is only **revealed**
after ``think_need`` steps. With fewer steps the emitted answer is a blend of the
head's prediction and a uniform guess (``reveal_weight`` below), so too-short
chains lose accuracy — but thinking beyond ``think_need`` gains nothing. This
makes the optimum a **small positive** length, so the length penalty *controls*
overthinking rather than eliminating reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyPolicy(nn.Module):
    def __init__(self, num_values: int = 10, max_think: int = 8,
                 think_need: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.num_values = num_values
        self.max_think = max_think
        self.think_need = think_need
        # Answer policy: a small MLP reasoner over the two input numbers.
        self.a_emb = nn.Embedding(num_values, hidden)
        self.b_emb = nn.Embedding(num_values, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.answer_head = nn.Linear(hidden, num_values)
        # Length policy: logits over chain lengths 0..max_think, shared across
        # problems, initialised to favour a *long* chain (the "overthinking"
        # starting point the length penalty is meant to fix).
        self.length_logits = nn.Parameter(torch.zeros(max_think + 1))
        with torch.no_grad():
            self.length_logits[max_think] = 4.0

    def answer_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = self.mlp(torch.cat([self.a_emb(a), self.b_emb(b)], dim=-1))
        return self.answer_head(h)

    def length_dist(self, batch: int) -> torch.Tensor:
        return self.length_logits.unsqueeze(0).expand(batch, -1)

    def reveal_weight(self, length: torch.Tensor) -> torch.Tensor:
        """Fraction of the answer that is 'revealed' after ``length`` think steps.

        The answer is fully revealed once the chain reaches ``think_need`` steps;
        below that only a small fraction leaks through (the rest is a uniform
        guess). Thinking *beyond* ``think_need`` adds nothing. This makes the
        optimum a small **positive** length (== ``think_need``): shorter chains
        sharply lose accuracy, longer chains just waste tokens."""
        return torch.where(length >= self.think_need,
                           torch.ones_like(length, dtype=torch.float),
                           torch.full_like(length, 0.25, dtype=torch.float))
