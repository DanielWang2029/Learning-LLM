"""Rotary Position Embeddings (RoPE), inherited from the LLaMA recipe.

Llama 2 uses RoPE at every layer, exactly like LLaMA (Llama 2 paper,
Section 2.2). RoPE rotates the query and key vectors by an angle proportional to
their absolute position; attention then depends on the rotation difference, so
it behaves relatively. This is the real-valued "rotate-half" formulation.
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
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
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
