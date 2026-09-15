"""Compressed Convolutional Attention (CCA) vs. full attention.

ZAYA1-8B replaces standard attention with CCA (paper §II-A-1): a lightweight
convolutional downprojector compresses the key/value sequence into a shorter
latent sequence before attention. Queries then attend over the *compressed* K/V,
so both the attention matrix and the KV-cache shrink by the compression factor,
while (the paper reports) quality is preserved.

    K, V  = projections of x                       # (B, L, d)
    K_c, V_c = Conv1d_stride_r(K), Conv1d_stride_r(V)   # (B, L/r, d)  -- compress
    attn = softmax(Q · K_cᵀ / √d_head)             # (B, L, L/r)
    out  = attn · V_c

The convolution both mixes local context and downsamples, so the compressed
tokens summarize windows of the original sequence. KV-cache for decoding stores
only ``L/r`` key/value vectors instead of ``L``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHACore:
    """Shared multi-head split/merge helpers."""

    def _split(self, x, n_heads):
        B, L, d = x.shape
        return x.view(B, L, n_heads, d // n_heads).transpose(1, 2)  # (B,h,L,dh)

    def _merge(self, x):
        B, h, L, dh = x.shape
        return x.transpose(1, 2).reshape(B, L, h * dh)


class FullAttention(nn.Module, _MHACore):
    """Standard bidirectional multi-head attention (the baseline)."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.w_q(x), self.n_heads)
        k = self._split(self.w_k(x), self.n_heads)
        v = self._split(self.w_v(x), self.n_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, v)
        return self.w_o(self._merge(out))

    def kv_positions(self, seq_len: int) -> int:
        return seq_len


class CompressedConvAttention(nn.Module, _MHACore):
    """CCA: compress K/V with a strided conv, then attend (paper §II-A-1)."""

    def __init__(self, d_model: int, n_heads: int = 4, compress: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compress = compress
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # Depthwise strided convolutions downsample the sequence by `compress`.
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=compress,
                                stride=compress, groups=d_model)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=compress,
                                stride=compress, groups=d_model)
        self.last_attn = None

    def _compress(self, x, conv):
        # x: (B, L, d) -> (B, L/r, d). Pads so L need not be divisible by r.
        B, L, d = x.shape
        pad = (-L) % self.compress
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        xc = conv(x.transpose(1, 2)).transpose(1, 2)   # (B, Lc, d)
        return xc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.w_q(x), self.n_heads)                 # (B,h,L,dh)
        k_c = self._split(self._compress(self.w_k(x), self.conv_k), self.n_heads)
        v_c = self._split(self._compress(self.w_v(x), self.conv_v), self.n_heads)
        scores = torch.matmul(q, k_c.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)                           # (B,h,L,Lc)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, v_c)
        return self.w_o(self._merge(out))

    def kv_positions(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.compress)


class Classifier(nn.Module):
    """Embed -> one attention layer -> mean-pool -> linear classifier.

    ``attention`` is a FullAttention or CompressedConvAttention instance so the
    two only differ in how keys/values are handled.
    """

    def __init__(self, vocab_size: int, n_classes: int, attention: nn.Module,
                 d_model: int = 48, n_mark: int = 2) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.mark_embed = nn.Embedding(n_mark, d_model)   # marked / unmarked flag
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        L = tokens.size(1)
        x = self.tok_embed(tokens) + self.mark_embed(marks) + self.pos[:, :L]
        x = x + self.attn(self.norm1(x))
        pooled = self.norm2(x).mean(dim=1)
        return self.readout(pooled)


def kv_cache_size(seq_len: int, d_model: int, compress: int = 1) -> int:
    """Number of scalars in the K and V caches for one attention layer."""
    positions = math.ceil(seq_len / compress)
    return 2 * positions * d_model
