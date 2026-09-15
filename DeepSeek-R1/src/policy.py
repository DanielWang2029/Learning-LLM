"""A small *parameterized* policy trained with reinforcement learning.

DeepSeek-R1 RL-trains a large language model. On a CPU we cannot, so we use a
compact parameterized policy (the paper explicitly motivates RL as the driver;
here the policy is a handful of learnable categorical distributions rather than
a deep network). It still emits a full response string of the paper's form

    {think}[answer]          (stands in for <think>…</think><answer>…</answer>)

by making three independent decisions, each a softmax over learnable logits:

* ``use_tags``  — wrap the answer in the reasoning/answer tags, or blurt a bare
  number. Global (shared across prompts): the *format* the model adopts.
* ``think``     — put the actual computation ``a+b`` inside the think block, or
  leave it empty. Global: whether *reasoning* is shown.
* ``answer[p]`` — which digit to answer, one distribution **per prompt**: this is
  where the arithmetic is actually learned.

All logits are ``nn.Parameter``s, so the GRPO policy-gradient update is a plain
autograd step. Because generation factorizes into these decisions, exploration
and credit assignment are clean — exactly what makes the RL signal legible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametrizedReasoner(nn.Module):
    def __init__(self, n_prompts: int, n_digits: int = 10) -> None:
        super().__init__()
        self.n_digits = n_digits
        # Start with no preference (uniform) for every decision.
        self.use_tags = nn.Parameter(torch.zeros(2))          # [no-tags, tags]
        self.think = nn.Parameter(torch.zeros(2))             # [empty, show "a+b"]
        self.answer = nn.Parameter(torch.zeros(n_prompts, n_digits))

    # -- distributions -------------------------------------------------------
    def probs(self):
        return (F.softmax(self.use_tags, -1),
                F.softmax(self.think, -1),
                F.softmax(self.answer, -1))

    @torch.no_grad()
    def sample(self, prompt_idx: torch.Tensor, generator=None):
        """Sample one decision-tuple for each prompt index in ``prompt_idx``.

        Returns tensors ``(use_tags_choice, think_choice, digit_choice)``.
        """
        pu, pt, pa = self.probs()
        B = prompt_idx.numel()
        u = torch.multinomial(pu.expand(B, 2), 1, generator=generator).squeeze(1)
        t = torch.multinomial(pt.expand(B, 2), 1, generator=generator).squeeze(1)
        d = torch.multinomial(pa[prompt_idx], 1, generator=generator).squeeze(1)
        return u, t, d

    @torch.no_grad()
    def greedy(self, prompt_idx: torch.Tensor):
        pu, pt, pa = self.probs()
        return (pu.argmax().expand(prompt_idx.numel()),
                pt.argmax().expand(prompt_idx.numel()),
                pa[prompt_idx].argmax(dim=1))

    # -- scoring (differentiable) -------------------------------------------
    def logprob(self, prompt_idx, u, t, d):
        """Log π of the sampled decisions (the think choice only counts when the
        response actually uses tags)."""
        lpu = F.log_softmax(self.use_tags, -1)[u]
        lpt = F.log_softmax(self.think, -1)[t] * (u == 1).float()
        lpa = F.log_softmax(self.answer[prompt_idx], -1).gather(1, d.unsqueeze(1)).squeeze(1)
        return lpu + lpt + lpa

    def entropy(self):
        """Total entropy of the three decision distributions (exploration bonus)."""
        pu, pt, pa = self.probs()
        h = lambda p: -(p * (p + 1e-9).log()).sum(-1)
        return h(pu) + h(pt) + h(pa).mean()


def compose(u: int, t: int, d: int, a: int, b: int) -> str:
    """Turn a decision-tuple into the response string."""
    if u == 1:
        think = f"{a}+{b}" if t == 1 else ""
        return f"{{{think}}}[{d}]"
    return f"{d}"  # bare answer, no reasoning format
