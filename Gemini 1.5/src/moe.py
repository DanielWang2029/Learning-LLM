"""Sparse Mixture-of-Experts feed-forward layer (Gemini 1.5, §2).

Gemini 1.5 is described as a *sparse mixture-of-experts* Transformer: each token
is routed to a small number of experts out of a much larger pool, so the model's
total parameter count can grow while the compute *per token* stays roughly fixed
(Shazeer et al. 2017; Lepikhin et al. 2020). The report credits this MoE design,
alongside long-context advances, for the model's efficiency.

This implements a compact **top-k** MoE FFN:

* a linear ``router`` scores the experts for each token,
* the top-k experts are selected and their outputs combined with softmax gates,
* a **load-balancing auxiliary loss** (Switch/GShard style) encourages the router
  to spread tokens evenly instead of collapsing onto a few experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert = a small position-wise MLP."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RouteInfo:
    load: torch.Tensor      # (E,) fraction of token-slots handled by each expert
    importance: torch.Tensor  # (E,) mean router probability per expert
    aux_loss: torch.Tensor  # scalar load-balancing loss


class SparseMoE(nn.Module):
    def __init__(self, dim: int, hidden: int, n_experts: int = 4, top_k: int = 2,
                 alpha: float = 0.01) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.alpha = alpha
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor):
        """x: (batch, seq, dim). Returns (output, RouteInfo)."""
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])          # (T, dim)
        T = tokens.size(0)

        logits = self.router(tokens)               # (T, E)
        probs = F.softmax(logits, dim=-1)
        topk_val, topk_idx = probs.topk(self.top_k, dim=-1)   # (T, k)
        gates = topk_val / topk_val.sum(dim=-1, keepdim=True)  # renormalize top-k

        out = torch.zeros_like(tokens)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]                # (T,) chosen expert per token
            gate = gates[:, slot].unsqueeze(1)     # (T,1)
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += gate[mask] * self.experts[e](tokens[mask])

        # Load-balancing loss: fraction of dispatched slots × mean prob per expert.
        one_hot = F.one_hot(topk_idx, self.n_experts).float().sum(dim=1)  # (T,E) counts
        load = one_hot.mean(dim=0)                 # fraction of slots per expert
        importance = probs.mean(dim=0)             # mean router prob per expert
        aux = self.alpha * self.n_experts * torch.sum(load * importance)

        info = RouteInfo(load=load.detach(), importance=importance.detach(), aux_loss=aux)
        return out.reshape(shape), info
