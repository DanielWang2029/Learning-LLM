"""Shared Conformer encoder with dynamic chunk masking (U2/U2++).

The single most important property for U2++ is that **one** encoder works both
full-context (offline) and streaming, achieved by training with *dynamic chunk
masking*: each batch uses a randomly chosen chunk size, and attention is
restricted so a frame may only see up to the end of its own chunk (plus all
left history). Convolutions are causal so they never peek into the future.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def chunk_mask(T: int, chunk_size: int, device=None) -> torch.Tensor:
    """(T, T) bool mask; query i may attend to key j iff j <= end-of-chunk(i).

    Left history is unlimited; look-ahead is limited to the current chunk. A
    ``chunk_size >= T`` gives full (offline) context.
    """
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    chunk_end = (i // chunk_size) * chunk_size + chunk_size - 1
    return j <= chunk_end


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.h, self.d_k = num_heads, d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / self.d_k ** 0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalConv(nn.Module):
    """Conformer conv module with a causal (left-padded) depthwise conv."""

    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.k = kernel_size
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.glu(self.pw1(x.transpose(1, 2)), dim=1)
        y = F.pad(y, (self.k - 1, 0))
        y = self.dw(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(F.silu(y))
        return self.dropout(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attn = SelfAttention(d_model, num_heads, dropout)
        self.conv = CausalConv(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.n_attn = nn.LayerNorm(d_model)
        self.n_conv = nn.LayerNorm(d_model)
        self.n_out = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.n_attn(x), mask)
        x = x + self.conv(self.n_conv(x))
        x = x + 0.5 * self.ffn2(x)
        return self.n_out(x)


class ConformerEncoder(nn.Module):
    """Shared encoder used by both the CTC and attention branches."""

    def __init__(self, n_mels=40, d_model=64, num_heads=2, d_ff=128,
                 num_blocks=3, kernel_size=9, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        )

    def forward(self, feats, chunk_size: Optional[int] = None):
        x = self.proj(feats)
        cs = chunk_size if chunk_size else x.size(1)
        mask = chunk_mask(x.size(1), cs, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return x
