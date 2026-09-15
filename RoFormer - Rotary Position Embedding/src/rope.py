"""Rotary Position Embedding (RoPE), from scratch.

RoPE (Su et al., 2021, "RoFormer") encodes *absolute* position by rotating the
query and key vectors, but does so in a way that makes the attention score
depend only on the *relative* position m - n.

Core idea (paper §3.2). Split a d-dimensional vector into d/2 pairs. Pair i is
rotated in its 2-D plane by an angle ``m * theta_i`` where the position is ``m``
and the per-pair frequency is

    theta_i = base ** (-2i / d),   i = 0 .. d/2 - 1.

Because rotations compose additively, the inner product of a rotated query at
position m and a rotated key at position n reduces to a function of (m - n):

    <R(m) q, R(n) k> = sum_i |q_i||k_i| cos((m - n) theta_i + phi_i)      (§3.4.3)

which is exactly the relative-position property we verify in the demo.
"""

from __future__ import annotations

import torch


def rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Per-pair rotation frequencies theta_i = base^(-2i/d)  (paper Eq. 15)."""
    assert dim % 2 == 0, "RoPE requires an even head dimension"
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return base ** (-i / dim)  # shape (dim/2,)


def rope_cache(seq_len: int, dim: int, base: float = 10000.0):
    """Precompute cos/sin tables of shape (seq_len, dim).

    Each frequency is duplicated so it lines up with the interleaved rotation
    used by :func:`apply_rope`.
    """
    theta = rope_frequencies(dim, base)  # (dim/2,)
    pos = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    angles = torch.outer(pos, theta)  # (seq_len, dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (seq_len, dim)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) on the two halves of the last dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by the position-dependent angles.

    ``x``   : (..., seq_len, dim)
    ``cos`` : (seq_len, dim)   ``sin`` : (seq_len, dim)

    Implements  x' = x * cos + rotate_half(x) * sin,  the standard
    "rotate the pairs" form of R(m) x.
    """
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def rope_dot(q: torch.Tensor, k: torch.Tensor, m: int, n: int, base: float = 10000.0) -> float:
    """Inner product <R(m) q, R(n) k> for single vectors q, k of dim ``d``.

    Convenience helper used by the relative-position verification: it rotates
    ``q`` to position ``m`` and ``k`` to position ``n`` and returns their dot.
    """
    dim = q.shape[-1]
    seq = max(m, n) + 1
    cos, sin = rope_cache(seq, dim, base)
    qr = apply_rope(q.view(1, dim), cos[m : m + 1], sin[m : m + 1])
    kr = apply_rope(k.view(1, dim), cos[n : n + 1], sin[n : n + 1])
    return float((qr * kr).sum())
