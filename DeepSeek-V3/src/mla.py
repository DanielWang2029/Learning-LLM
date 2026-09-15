"""Multi-Head Latent Attention (MLA) — DeepSeek-V3, §2.1 (from DeepSeek-V2).

Standard multi-head attention must cache a full key and value vector per head for
every past token, which dominates memory during long-context inference. **MLA**
instead compresses the keys and values into a small **low-rank latent** ``c_KV``
that is what gets cached; keys and values are reconstructed on the fly by
up-projection. RoPE is *decoupled*: a small separate key ``k_R`` (shared across
heads) carries the rotary position information, because RoPE cannot be applied to
a position-independent compressed latent.

Per-token KV cache:
    standard MHA : 2 · n_heads · head_dim          (a full K and V per head)
    MLA          : d_c (kv latent) + d_rope         (shared across all heads)

so MLA is smaller by ``2·n_heads·head_dim / (d_c + d_rope)`` — a big win once the
model is wide, while (as the demo shows) matching full-attention quality.

This module implements MLA and a plain MHA baseline sharing the same interface,
plus helpers to report the per-token cache size of each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_tables(seq_len, dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    # x: (b, h, t, d_rope); cos/sin: (t, d_rope)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


@dataclass
class MLAConfig:
    dim: int = 64
    n_heads: int = 4
    head_dim: int = 16       # content dim per head (v and the non-rope part of k/q)
    kv_latent: int = 32      # d_c: compressed KV latent that gets cached
    q_latent: int = 48       # compressed query latent
    rope_dim: int = 16       # decoupled RoPE dimension (per head for q, shared for k)
    rope_base: float = 10000.0


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: MLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        nh, dh, dr = cfg.n_heads, cfg.head_dim, cfg.rope_dim

        # --- KV path: compress to c_KV (cached), then up-project to K_content, V.
        self.w_dkv = nn.Linear(cfg.dim, cfg.kv_latent, bias=False)
        self.w_uk = nn.Linear(cfg.kv_latent, nh * dh, bias=False)
        self.w_uv = nn.Linear(cfg.kv_latent, nh * dh, bias=False)
        # decoupled key RoPE (shared across heads): one dr-dim key per token (cached).
        self.w_kr = nn.Linear(cfg.dim, dr, bias=False)

        # --- Query path: compress then up-project to Q_content and per-head Q_rope.
        self.w_dq = nn.Linear(cfg.dim, cfg.q_latent, bias=False)
        self.w_uq = nn.Linear(cfg.q_latent, nh * dh, bias=False)
        self.w_qr = nn.Linear(cfg.q_latent, nh * dr, bias=False)

        self.w_o = nn.Linear(nh * dh, cfg.dim, bias=False)

    def forward(self, x, return_latent=False):
        b, t, _ = x.shape
        nh, dh, dr = self.cfg.n_heads, self.cfg.head_dim, self.cfg.rope_dim

        # ---- compress KV to the latent that would be cached ----
        c_kv = self.w_dkv(x)                                   # (b, t, d_c)  <-- cached
        k_c = self.w_uk(c_kv).view(b, t, nh, dh).transpose(1, 2)  # (b, nh, t, dh)
        v = self.w_uv(c_kv).view(b, t, nh, dh).transpose(1, 2)    # (b, nh, t, dh)
        k_r = self.w_kr(x).view(b, t, 1, dr).transpose(1, 2)      # (b, 1, t, dr) <-- cached

        # ---- queries ----
        c_q = self.w_dq(x)
        q_c = self.w_uq(c_q).view(b, t, nh, dh).transpose(1, 2)   # (b, nh, t, dh)
        q_r = self.w_qr(c_q).view(b, t, nh, dr).transpose(1, 2)   # (b, nh, t, dr)

        # ---- decoupled RoPE on the rope parts ----
        cos, sin = _rope_tables(t, dr, self.cfg.rope_base, x.device)
        q_r = _apply_rope(q_r, cos, sin)
        k_r = _apply_rope(k_r, cos, sin)
        k_r = k_r.expand(b, nh, t, dr)                         # shared across heads

        # ---- attention on concatenated [content ; rope] ----
        q = torch.cat([q_c, q_r], dim=-1)                      # (b, nh, t, dh+dr)
        k = torch.cat([k_c, k_r], dim=-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh + dr)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, nh * dh)
        out = self.w_o(out)
        if return_latent:
            return out, c_kv, k_r
        return out

    def cache_floats_per_token(self) -> int:
        """MLA caches only the KV latent + the shared decoupled RoPE key."""
        return self.cfg.kv_latent + self.cfg.rope_dim


class StandardMHA(nn.Module):
    """Plain multi-head attention with RoPE — the MLA baseline for comparison."""

    def __init__(self, cfg: MLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.n_heads * cfg.head_dim
        self.w_q = nn.Linear(cfg.dim, d, bias=False)
        self.w_k = nn.Linear(cfg.dim, d, bias=False)
        self.w_v = nn.Linear(cfg.dim, d, bias=False)
        self.w_o = nn.Linear(d, cfg.dim, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        nh, dh = self.cfg.n_heads, self.cfg.head_dim
        q = self.w_q(x).view(b, t, nh, dh).transpose(1, 2)
        k = self.w_k(x).view(b, t, nh, dh).transpose(1, 2)
        v = self.w_v(x).view(b, t, nh, dh).transpose(1, 2)
        cos, sin = _rope_tables(t, dh, self.cfg.rope_base, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, nh * dh)
        return self.w_o(out)

    def cache_floats_per_token(self) -> int:
        """Standard MHA caches a full key and value per head."""
        return 2 * self.cfg.n_heads * self.cfg.head_dim
