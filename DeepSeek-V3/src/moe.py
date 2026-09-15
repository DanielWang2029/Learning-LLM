"""DeepSeekMoE feed-forward — fine-grained + shared experts (DeepSeek-V3, §2.2).

DeepSeekMoE refines the standard sparse MoE in two ways:

* **Fine-grained experts.** Instead of a few big experts, use many *small* experts
  and route each token to top-``k`` of them. Finer slicing lets the router compose
  a more precise combination of specialists per token.
* **Shared experts.** A small number of experts are *always on* for every token.
  They absorb common knowledge, freeing the routed experts to specialize and
  reducing redundancy across them.

The layer output is  ``shared_experts(x) + Σ_{routed top-k} gate · expert(x)``.
DeepSeek-V3 also adds an auxiliary-loss-free load balance via per-expert bias
terms on the routing scores; we include that bias here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))  # SwiGLU


@dataclass
class MoEInfo:
    load: torch.Tensor        # (n_routed,) fraction of tokens per routed expert
    aux_loss: torch.Tensor    # scalar (kept ~0; balance is done via router bias)


class DeepSeekMoE(nn.Module):
    def __init__(self, dim: int, expert_hidden: int, n_routed: int = 8,
                 n_shared: int = 1, top_k: int = 2, bias_speed: float = 0.02) -> None:
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        self.bias_speed = bias_speed
        self.router = nn.Linear(dim, n_routed, bias=False)
        # Auxiliary-loss-free balancing: a per-expert bias added to routing scores
        # (updated as a buffer, not by gradient) — DeepSeek-V3's balancing trick.
        self.register_buffer("route_bias", torch.zeros(n_routed))
        self.routed = nn.ModuleList([MLP(dim, expert_hidden) for _ in range(n_routed)])
        self.shared = nn.ModuleList([MLP(dim, expert_hidden) for _ in range(n_shared)])

    def forward(self, x):
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        T = tokens.size(0)

        scores = self.router(tokens).softmax(-1)               # (T, n_routed)
        # Select top-k using the bias-adjusted scores, but gate with the true scores.
        adj = scores + self.route_bias[None]
        topk_idx = adj.topk(self.top_k, dim=-1).indices          # (T, k)
        gate = torch.gather(scores, 1, topk_idx)
        gate = gate / gate.sum(-1, keepdim=True)

        out = torch.zeros_like(tokens)
        counts = torch.zeros(self.n_routed, device=tokens.device)
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]
            g = gate[:, slot].unsqueeze(1)
            for e in range(self.n_routed):
                mask = idx == e
                if mask.any():
                    out[mask] += g[mask] * self.routed[e](tokens[mask])
                    counts[e] += mask.sum()

        # Always-on shared experts.
        for sh in self.shared:
            out += sh(tokens)

        load = counts / max(1, T)
        # Auxiliary-loss-free balancing: nudge the bias down for overloaded experts,
        # up for underloaded ones, so future routing evens out (no gradient needed).
        if self.training:
            with torch.no_grad():
                target = self.top_k / self.n_routed
                self.route_bias += self.bias_speed * (target - load)

        info = MoEInfo(load=load.detach(), aux_loss=torch.zeros((), device=tokens.device))
        return out.reshape(shape), info
