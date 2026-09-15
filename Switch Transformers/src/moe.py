"""Switch Transformer MoE layer (top-1 routing) with a load-balancing loss.

Fedus et al. (2021) replace the dense feed-forward sub-layer of a transformer
with a *sparse* Mixture-of-Experts layer. A lightweight router sends each token
to exactly ONE expert (top-1, the "Switch" simplification, §2.1), so the
per-token compute stays constant even as you add experts (and parameters).

Two pieces are implemented here:

* :class:`SwitchFFN` — router + experts + top-1 dispatch. The chosen expert's
  output is scaled by the router probability so the router receives gradient.
* the differentiable **load-balancing loss** (paper §2.2, Eq. 4-6):

      loss_aux = alpha * E * sum_i  f_i * P_i

  where ``f_i`` is the fraction of tokens dispatched to expert ``i`` and ``P_i``
  is the mean router probability for expert ``i``. It is minimized when both are
  uniform (1/E), which pushes the router to spread tokens evenly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A standard position-wise FFN — one of the experts."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RouteInfo:
    assignments: torch.Tensor  # (n_tokens,) chosen expert id per token
    probs: torch.Tensor        # (n_tokens, E) router softmax probabilities
    aux_loss: torch.Tensor     # scalar load-balancing loss


class SwitchFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int,
                 alpha: float = 0.01) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.alpha = alpha
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor, route_input: torch.Tensor | None = None):
        """x: (..., d_model), processed by the experts.

        ``route_input`` (same leading shape as ``x``) is what the *router* sees
        when choosing an expert. It defaults to ``x`` — the standard Switch
        layer — but letting the router key on a different signal than the
        experts receive is exactly what forces specialization in the demo.

        Returns (output, RouteInfo).
        """
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])  # (T, d)
        T = tokens.size(0)

        route_tokens = tokens if route_input is None else route_input.reshape(-1, route_input.shape[-1])
        logits = self.router(route_tokens)     # (T, E)
        probs = F.softmax(logits, dim=-1)      # (T, E)
        gate, expert_id = probs.max(dim=-1)    # top-1 gate value & expert

        out = torch.zeros_like(tokens)
        for e in range(self.n_experts):
            mask = expert_id == e
            if mask.any():
                # Scale by the router probability (Switch, §2.1) so the router
                # gets a learning signal through the selected expert.
                out[mask] = gate[mask].unsqueeze(1) * self.experts[e](tokens[mask])

        # Load-balancing auxiliary loss (paper Eq. 4-6).
        f_i = torch.zeros(self.n_experts, device=tokens.device)
        counts = torch.bincount(expert_id, minlength=self.n_experts).float()
        f_i = counts / T                       # fraction of tokens per expert
        P_i = probs.mean(dim=0)                # mean router prob per expert
        aux = self.alpha * self.n_experts * torch.sum(f_i * P_i)

        info = RouteInfo(assignments=expert_id.detach(), probs=probs.detach(),
                         aux_loss=aux)
        return out.reshape(shape), info
