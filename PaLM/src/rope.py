"""Rotary Position Embeddings (RoPE).

PaLM uses RoPE (Su et al., 2021) instead of learned or sinusoidal absolute
position embeddings (PaLM paper, Section 2, "RoPE Embeddings"). RoPE encodes
position by *rotating* the query and key vectors by an angle proportional to
their absolute position; because the attention score depends on the rotation
difference between a query and a key, the mechanism is effectively relative.

This module uses the "rotate-half" real-valued formulation, which is
numerically equivalent to the complex formulation in the original paper and is
the common implementation in modern LLM codebases.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotation to ``x`` of shape (batch, heads, seq, head_dim).

    ``cos`` / ``sin`` have shape (seq, head_dim) and are broadcast over the
    batch and head dimensions.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precompute and cache the cos/sin rotation tables for RoPE."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        # Geometrically decreasing rotation frequencies, one per dimension pair.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        return emb.cos(), emb.sin()
