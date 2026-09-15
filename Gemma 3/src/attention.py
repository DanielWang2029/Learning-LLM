"""Local (sliding-window) & global attention with QK-norm (Gemma 3, 2025).

Gemma 3's efficiency trick is to make most attention layers **local** — each
position only attends within a sliding window — and only a few layers **global**.
It also applies **QK-norm**: query and key vectors are normalized before the dot
product, which stabilizes attention logits.

This module implements one multi-head attention layer that can run in either mode.
The only difference between the two modes is the attention *mask*:

* ``global``: every position may attend to every other position   (cost ~ T·T)
* ``local`` : position ``i`` may attend only to ``|i-j| <= window`` (cost ~ T·(2w+1))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_mask(seq_len: int, mode: str, window: int) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask; True = allowed to attend.

    ``global`` allows everything; ``local`` allows a symmetric sliding window of
    radius ``window`` around each position.
    """
    if mode == "global":
        return torch.ones(seq_len, seq_len, dtype=torch.bool)
    idx = torch.arange(seq_len)
    dist = (idx[:, None] - idx[None, :]).abs()
    return dist <= window


def mask_entries(seq_len: int, mode: str, window: int) -> int:
    """Number of allowed (query, key) attention-score entries — the memory proxy."""
    return int(attention_mask(seq_len, mode, window).sum().item())


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mode: str, window: int,
                 qk_norm: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.mode, self.window, self.qk_norm = mode, window, qk_norm
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        if self.qk_norm:  # Gemma 3: normalize q and k before the dot product
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0  # unit-norm vectors already bound the logits
        else:
            scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        allowed = attention_mask(t, self.mode, self.window).to(x.device)
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.w_o(out)
