"""Rotary Position Embeddings (RoPE) — Llama 3, §3.1.

Llama 3 uses RoPE at every attention layer, but raises the rotary base
frequency from Llama 2's 10,000 to **500,000** so the same rotation scheme
resolves positions cleanly across the 128K-token context window. RoPE rotates
each query/key vector by an angle proportional to its absolute position; because
attention depends on the *difference* of two rotations, the mechanism is
effectively relative.

This uses the real-valued "rotate-half" formulation, numerically equivalent to
the complex formulation and standard in modern LLM codebases.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 500000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
