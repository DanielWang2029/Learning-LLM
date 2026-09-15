"""Sparse Mixture-of-Experts layer with top-2 routing (paper §2).

Mixtral replaces the dense feed-forward block with a **Sparse MoE**: a set of
``n_experts`` independent expert MLPs and a small linear **router**.  For every
token the router picks the **top-2** experts, and the token is processed by only
those two — their outputs are combined with the (renormalized) gate weights:

    g = softmax(router(x))
    (w1,e1),(w2,e2) = top-2 of g           # per token
    y = w1 * Expert_e1(x) + w2 * Expert_e2(x)

So each token uses only 2 of the 8 experts: the model has many parameters but a
much smaller *active* parameter count per token — the source of MoE efficiency
(paper §1-2).  A **load-balancing loss** (Switch/Shazeer style) discourages the
router from collapsing onto a few experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert: an independent position-wise MLP (SwiGLU-ish)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


@dataclass
class MoEStats:
    """Diagnostics captured on the last forward pass (for visualization)."""
    expert_indices: torch.Tensor      # (tokens, top_k) which experts were chosen
    gate_weights: torch.Tensor        # (tokens, top_k) their (renormalized) weights
    load_fraction: torch.Tensor       # (n_experts,) fraction of dispatch slots used
    router_probs: torch.Tensor        # (tokens, n_experts) full softmax gate


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8,
                 top_k: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.stats: MoEStats | None = None

    def expert_param_count(self) -> int:
        return sum(p.numel() for p in self.experts[0].parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (output, aux_load_balance_loss).  x: (batch, seq, d_model)."""
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)                       # (T, d),  T = b*t
        T = x_flat.size(0)

        router_logits = self.router(x_flat)             # (T, n_experts)
        probs = F.softmax(router_logits, dim=-1)

        # --- top-2 selection, then renormalize the two gate weights (Mixtral) ---
        top_w, top_i = probs.topk(self.top_k, dim=-1)   # (T, k)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        # --- sparse dispatch: each expert only sees its assigned tokens ---------
        out = torch.zeros_like(x_flat)
        load = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            hit = (top_i == e)                          # (T, k) bool
            if not hit.any():
                continue
            tok_idx, slot = hit.nonzero(as_tuple=True)
            w = top_w[tok_idx, slot].unsqueeze(-1)      # (m, 1)
            y = self.experts[e](x_flat[tok_idx]) * w
            out.index_add_(0, tok_idx, y)
            load[e] = float(tok_idx.numel())
        load = load / (T * self.top_k)                  # fraction of dispatch slots

        # --- load-balancing auxiliary loss (Switch Transformers, eq. 4) --------
        # f_i = fraction of tokens routed to expert i; P_i = mean router prob to i.
        # Minimized when both are uniform -> experts used evenly.
        importance = probs.mean(dim=0)                  # P_i  (n_experts,)
        aux = self.n_experts * torch.sum(load * importance)

        self.stats = MoEStats(
            expert_indices=top_i.detach(),
            gate_weights=top_w.detach(),
            load_fraction=load.detach(),
            router_probs=probs.detach(),
        )
        return out.reshape(b, t, d), aux
