"""The reasoner with a learned stop decision + budget forcing (s1, 2025).

Two learned pieces:

* ``step_net`` — a small MLP that learns one arithmetic operation ``(value, op)
  -> value``. This executes each reasoning step.
* ``stop_logits`` — a learned per-step-index stop head, trained by SFT on a *tiny
  curated* set. Because that set is skewed toward short chains, the head learns to
  stop thinking early — it under-thinks on long problems.

**Budget forcing** (the paper's mechanism) intervenes at decode time:

* to think *more*, whenever the model wants to stop before a minimum number of
  steps, we suppress the end token and append **"Wait"**, forcing another step;
* to think *less*, we can cap thinking at a maximum (append the end token early).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import MOD, N_OPS


class S1Reasoner(nn.Module):
    def __init__(self, max_len: int, hidden: int = 64) -> None:
        super().__init__()
        self.max_len = max_len
        self.step_net = nn.Sequential(
            nn.Linear(MOD + N_OPS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, MOD),
        )
        # Learned stop/continue logits per step index (0..max_len). Index t is the
        # decision made after t reasoning steps have already been taken.
        self.stop_logits = nn.Parameter(torch.zeros(max_len + 1, 2))

    def step_logits(self, value: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.one_hot(value, MOD).float(),
                       F.one_hot(op, N_OPS).float()], dim=-1)
        return self.step_net(x)

    def wants_stop(self, t: int) -> bool:
        """The model's *natural* stop decision after ``t`` steps (argmax)."""
        t = min(t, self.max_len)
        return bool(self.stop_logits[t].argmax().item() == 1)

    @torch.no_grad()
    def decode(self, start, op_ids, min_think=0, max_think=None):
        """Budget-forced greedy decode.

        Applies operations one at a time. The natural stop decision is honoured
        only once at least ``min_think`` steps have been taken; before that, a
        stop attempt is overridden with a "Wait" (forcing more thinking). Thinking
        is also capped at ``max_think``.

        Returns a dict with the answer, the step-by-step trace (marking forced
        "Wait" continuations), and step/wait counts.
        """
        if max_think is None:
            max_think = len(op_ids)
        v = torch.tensor(start)
        trace = [{"kind": "start", "value": int(v)}]
        applied = waits = 0
        for o in op_ids:
            if applied >= max_think:
                break
            if self.wants_stop(applied):
                if applied >= min_think:
                    break  # honour the natural stop (end token)
                waits += 1
                trace.append({"kind": "wait"})  # budget forcing: keep thinking
            v = self.step_logits(v.unsqueeze(0), torch.tensor([o])).argmax(-1).squeeze(0)
            applied += 1
            trace.append({"kind": "step", "op": o, "value": int(v)})
        return {"answer": int(v), "trace": trace, "steps": applied, "waits": waits}
