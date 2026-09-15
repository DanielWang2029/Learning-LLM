"""Sliding Window Attention + Grouped-Query Attention + a rolling KV cache.

These are the three memory/efficiency ideas Mistral 7B (Jiang et al., 2023) is
built around:

* **Sliding Window Attention (SWA)** — each token attends only to the previous
  ``W`` tokens (a banded causal mask), so attention cost per token is O(W), not
  O(sequence length). Stacking layers still lets information propagate far:
  after k layers the receptive field is ~k·W tokens.
* **Grouped-Query Attention (GQA)** — fewer key/value heads than query heads
  (``n_kv_heads < n_heads``); each KV head is shared by a group of query heads,
  shrinking the KV cache by ``n_heads / n_kv_heads``.
* **Rolling-buffer KV cache** — because a token never attends beyond ``W`` steps
  back, the cache only needs to hold the last ``W`` keys/values; older entries
  are overwritten, giving a *fixed* cache size regardless of sequence length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sliding_window_mask(seq_len: int, window: int, device=None) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask: True where query i may attend to key j.

    Allowed iff ``0 <= i - j < window`` (causal AND within the window).
    """
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    delta = i - j
    return (delta >= 0) & (delta < window)


class RollingKVCache:
    """A fixed-capacity KV cache holding only the last ``window`` positions.

    Appending past capacity drops the oldest entry (the "rolling buffer"), so
    memory is bounded by ``window`` no matter how long the sequence gets.
    """

    def __init__(self, window: int) -> None:
        self.window = window
        self.k: torch.Tensor | None = None  # (B, n_kv_heads, <=window, head_dim)
        self.v: torch.Tensor | None = None

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_t, v_t
        else:
            self.k = torch.cat([self.k, k_t], dim=2)[:, :, -self.window:, :]
            self.v = torch.cat([self.v, v_t], dim=2)[:, :, -self.window:, :]
        return self.k, self.v

    def size(self) -> int:
        return 0 if self.k is None else self.k.size(2)


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 window: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.last_attn: torch.Tensor | None = None  # for inspection

    def _shape_q(self, x):
        B, T, _ = x.shape
        return self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, proj, x):
        B, T, _ = x.shape
        return proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def _expand_kv(self, k, v):
        # (B, n_kv_heads, T, hd) -> (B, n_heads, T, hd) by repeating groups.
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
        return k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full parallel forward with the banded sliding-window mask."""
        B, T, _ = x.shape
        q = self._shape_q(x)
        k = self._shape_kv(self.wk, x)
        v = self._shape_kv(self.wv, x)
        k, v = self._expand_kv(k, v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = sliding_window_mask(T, self.window, x.device)  # (T, T)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

    def step(self, x_t: torch.Tensor, cache: RollingKVCache) -> torch.Tensor:
        """Incremental forward for ONE token using the rolling KV cache.

        Produces the same result as the full forward at that position, because
        the cache holds exactly the last ``window`` keys/values — the tokens the
        sliding window allows this query to attend to.
        """
        q = self._shape_q(x_t)                       # (B, n_heads, 1, hd)
        k_t = self._shape_kv(self.wk, x_t)           # (B, n_kv_heads, 1, hd)
        v_t = self._shape_kv(self.wv, x_t)
        k, v = cache.append(k_t, v_t)                # rolling: last <=window
        k, v = self._expand_kv(k, v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(x_t.size(0), 1, -1)
        return self.wo(out)
