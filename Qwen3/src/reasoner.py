"""The hybrid reasoner: one model, two modes (Qwen3 Technical Report, 2025).

Qwen3 unifies *thinking* and *non-thinking* behaviour in a single model and lets
the caller control a **thinking budget** (how much step-by-step computation is
allowed before the answer is produced). We reproduce that structure at tiny
scale with two learnable heads that share the same task:

* ``StepReasoner``  — the *thinking* path. A small MLP that learns the
  single-operation transition ``(value, op) -> value``. At inference it applies
  operations one at a time (one "thinking token" per step), carrying the running
  value forward, up to a caller-chosen **thinking budget**.
* ``DirectAnswerer`` — the *non-thinking* path. A small MLP that must map the
  whole problem to the final answer in a single shot, with no intermediate work.

Both are trained on the same problems. Turning thinking on (and raising the
budget) lets the step reasoner actually execute the chain, which is what drives
accuracy up at the cost of more tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import MOD, N_OPS

PAD_OP = N_OPS  # extra token used to pad chains for the one-shot answerer


class StepReasoner(nn.Module):
    """Thinking path: learns one operation step, then is iterated at test time."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        # input = one-hot(value in 0..9) concatenated with one-hot(op)
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
    def run(self, start: int, op_ids: list[int], budget: int):
        """Apply up to ``budget`` operations one at a time (the thinking loop).

        Returns ``(answer, trace)`` where ``trace`` is the running value after
        each executed step. If ``budget`` runs out before the chain is finished,
        the current running value is returned as the (premature) answer.
        """
        v = torch.tensor(start)
        trace = [int(v)]
        for o in op_ids[:budget]:
            v = self.step_logits(v.unsqueeze(0),
                                  torch.tensor([o])).argmax(-1).squeeze(0)
            trace.append(int(v))
        return int(v), trace


class DirectAnswerer(nn.Module):
    """Non-thinking path: map the entire padded problem to an answer in one shot."""

    def __init__(self, max_len: int, hidden: int = 128) -> None:
        super().__init__()
        self.max_len = max_len
        in_dim = MOD + max_len * (N_OPS + 1)  # start one-hot + per-slot op one-hot
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, MOD),
        )

    def _encode(self, starts: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
        s = F.one_hot(starts, MOD).float()
        o = F.one_hot(ops, N_OPS + 1).float().reshape(ops.size(0), -1)
        return torch.cat([s, o], dim=-1)

    def logits(self, starts: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
        return self.net(self._encode(starts, ops))

    @torch.no_grad()
    def answer(self, start: int, op_ids: list[int]) -> int:
        ops = op_ids + [PAD_OP] * (self.max_len - len(op_ids))
        logits = self.logits(torch.tensor([start]), torch.tensor([ops]))
        return int(logits.argmax(-1))
