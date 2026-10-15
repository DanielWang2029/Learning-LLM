"""Adaptive RMSNorm delay conditioning (Voxtral Realtime, Section 2.3).

A single model is conditioned on the target streaming delay tau (a multiple of
80 ms, i.e. a number of adapter frames). tau is turned into a sinusoidal
embedding, projected by a tiny GELU MLP (inner dim 32) to g(tau) in R^d, and
injected multiplicatively on the FEED-FORWARD branch of every decoder block:

    r_ffn = FFN( RMSNorm(h) * (1 + g(tau)) ),      y = h + r_ffn

The attention branch is left unconditioned. Training samples tau over a range so
one model can run at any delay at inference (trading latency for accuracy).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """tau: (B,) float frame counts -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device) / half)
    a = tau[:, None] * freqs[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class DelayEmbedding(nn.Module):
    """tau -> g(tau) in R^d via sinusoidal embedding + a small GELU MLP."""

    def __init__(self, d: int, sin_dim: int = 32, inner: int = 32) -> None:
        super().__init__()
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(nn.Linear(sin_dim, inner), nn.GELU(), nn.Linear(inner, d))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal(tau.float(), self.sin_dim))  # (B, d)


class AdaRMSNorm(nn.Module):
    """RMSNorm whose output is scaled by (1 + g(tau)) before the FFN."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, h: torch.Tensor, g_tau: torch.Tensor) -> torch.Tensor:
        normed = self.w * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * (1.0 + g_tau[:, None, :])  # broadcast g over time
