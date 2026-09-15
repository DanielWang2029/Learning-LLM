"""Grouped-query attention with sliding-window masking and logit soft-capping.

Three Gemma 2 ingredients live here (paper §2):

* **Grouped-query attention (GQA):** many query heads share a smaller number of
  key/value heads, cutting the KV cache while keeping quality.
* **Alternating local / global attention:** *local* layers may only attend to
  the previous ``sliding_window`` tokens; *global* layers attend to all past
  tokens.  The layer stack alternates between the two.
* **Attention logit soft-capping:** pre-softmax scores are squashed with
  ``cap * tanh(score / cap)`` so no single score can dominate.

Rotary position embeddings (RoPE) are used, as in Gemma.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Gemma2Config


def build_causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular mask: position i may attend to all j <= i (global layer)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


def build_sliding_window_mask(seq_len: int, window: int) -> torch.Tensor:
    """Causal mask restricted to a window: i may attend to j in (i-window, i].

    This is the mask used by Gemma 2's *local* attention layers.
    """
    causal = build_causal_mask(seq_len)
    idx = torch.arange(seq_len)
    within = (idx[:, None] - idx[None, :]) < window
    return causal & within


def _rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """Precompute cos/sin tables for rotary position embeddings."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)              # (seq, half)
    emb = torch.cat([freqs, freqs], dim=-1)         # (seq, head_dim)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotated * sin


class GemmaAttention(nn.Module):
    """One attention block; ``is_local`` picks sliding-window vs global masking."""

    def __init__(self, cfg: Gemma2Config, is_local: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_local = is_local
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # Last attention weights, kept for visualization/inspection.
        self.last_attn: Optional[torch.Tensor] = None
        self.last_max_score: float = 0.0

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: repeat each KV head n_rep times so shapes match the query heads.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Attention logit soft-capping (paper §2): squash scores into (-cap, cap).
        if self.cfg.use_soft_cap:
            cap = self.cfg.attn_logit_softcap
            scores = cap * torch.tanh(scores / cap)

        self.last_max_score = float(scores.abs().max().item())

        # Local (sliding-window) vs global (full causal) masking.
        if self.is_local:
            mask = build_sliding_window_mask(t, self.cfg.sliding_window)
        else:
            mask = build_causal_mask(t)
        scores = scores.masked_fill(~mask.to(scores.device), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()

        out = torch.matmul(attn, v)                       # (b, heads, t, head_dim)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out)
