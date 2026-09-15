"""The Muon optimizer (DeepSeek-V4, Algorithm 1 / Section 2.4).

Muon updates 2-D weight matrices by *orthogonalizing* the (Nesterov) momentum
before the step. Orthogonalization approximately replaces M = UΣVᵀ by UVᵀ, so
every singular direction is stepped equally — which empirically converges
faster and more stably than Adam on matrix parameters.

DeepSeek-V4 uses **hybrid Newton–Schulz** iterations for the orthogonalization:
10 iterations in two stages (Eq. 28),

    Mₖ = a·Mₖ₋₁ + b·(Mₖ₋₁Mₖ₋₁ᵀ)Mₖ₋₁ + c·(Mₖ₋₁Mₖ₋₁ᵀ)²Mₖ₋₁,

with (a,b,c) = (3.4445, −4.7750, 2.0315) for the first 8 steps (fast, drives
singular values near 1) and (2, −1.5, 0.5) for the last 2 (stabilizes them
exactly at 1). Following the paper, 1-D parameters (biases, norms, embeddings)
are handled by a standard Adam update instead.
"""

from __future__ import annotations

from typing import List

import torch
from torch.optim.optimizer import Optimizer

# Hybrid Newton–Schulz coefficient schedule (Section 2.4).
_NS_FAST = (3.4445, -4.7750, 2.0315)
_NS_FINE = (2.0, -1.5, 0.5)
_NS_SCHEDULE = [_NS_FAST] * 8 + [_NS_FINE] * 2


def newton_schulz(
    grad: torch.Tensor, schedule: List = _NS_SCHEDULE, eps: float = 1e-7
) -> torch.Tensor:
    """Approximately orthogonalize a 2-D matrix via hybrid Newton–Schulz.

    Returns a matrix with (approximately) all singular values equal to 1,
    sharing the singular vectors of ``grad``. Operates in float and handles
    non-square matrices by orthogonalizing the smaller dimension.
    """
    assert grad.dim() == 2, "Newton–Schulz expects a 2-D matrix"
    x = grad.float()
    x = x / (x.norm() + eps)  # normalize so the largest singular value ≤ 1
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.t()
    for a, b, c in schedule:
        gram = x @ x.t()                     # (M Mᵀ)
        poly = b * gram + c * (gram @ gram)  # b·A + c·A²
        x = a * x + poly @ x
    if transpose:
        x = x.t()
    return x


class Muon(Optimizer):
    """Muon for 2-D weights + Adam for everything else (DeepSeek-V4 recipe).

    Args mirror Algorithm 1: learning rate ``lr`` (η), ``momentum`` (μ),
    ``weight_decay`` (λ) and ``rms_scale`` (γ, the update-RMS rescaling). Pass
    ``use_muon=False`` in a param group to force the Adam branch (used by the
    demo for a clean Adam baseline on the same model).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        rms_scale: float = 0.2,
        use_muon: bool = True,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rms_scale=rms_scale,
            use_muon=use_muon,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # noqa: D401
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Muon applies only to 2-D matrices; else use Adam.
                if group["use_muon"] and p.dim() == 2:
                    self._muon_step(p, g, group)
                else:
                    self._adam_step(p, g, group)

    def _muon_step(self, p, g, group):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]
        mu = group["momentum"]

        # Momentum buffer:  Mₜ = μ·Mₜ₋₁ + Gₜ
        buf.mul_(mu).add_(g)
        # Nesterov trick: orthogonalize (μ·Mₜ + Gₜ)
        update = newton_schulz(mu * buf + g)
        # Rescale update RMS by √max(n,m)·γ so it reuses Adam-like learning rates.
        n, m = p.shape
        update = update * (max(n, m) ** 0.5) * group["rms_scale"]
        # Decoupled weight decay, then the step.
        if group["weight_decay"]:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update, alpha=-group["lr"])

    def _adam_step(self, p, g, group):
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(g)
            state["exp_avg_sq"] = torch.zeros_like(g)
        state["step"] += 1
        b1, b2 = group["adam_betas"]
        eps = group["adam_eps"]
        m, v = state["exp_avg"], state["exp_avg_sq"]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1 ** state["step"]
        bc2 = 1 - b2 ** state["step"]
        denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
        if group["weight_decay"]:
            p.mul_(1 - group["lr"] * group["weight_decay"])
        p.addcdiv_(m / bc1, denom, value=-group["lr"])
