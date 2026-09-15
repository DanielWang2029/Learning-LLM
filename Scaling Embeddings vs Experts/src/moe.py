"""A small top-k sparse MoE feed-forward block — the "scaling experts" dimension.

Adding experts increases total parameters (capacity) while keeping per-token
compute fixed at top-k. In this reproduction it is the alternative to N-gram
embedding for spending a fixed extra parameter budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, d_hidden: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([_Expert(d_model, d_hidden) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        probs = F.softmax(self.router(flat), dim=-1)
        top_val, top_idx = probs.topk(self.top_k, dim=-1)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        y = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            sel = mask.any(dim=-1)
            w = (top_val * mask).sum(dim=-1)[sel]
            y[sel] += w.unsqueeze(-1) * self.experts[e](flat[sel])
        return y.reshape_as(x)
