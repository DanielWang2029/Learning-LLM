"""A tiny decoder-only model that can use NSA or full attention.

Used to check that Native Sparse Attention matches full-attention quality on a
long-range associative-recall task while attending to far fewer key positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .nsa import NSAAttention, FullAttention


class Block(nn.Module):
    def __init__(self, d_model, attn, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, collect=False):
        x = x + self.attn(self.norm1(x), collect=collect)
        x = x + self.ff(self.norm2(x))
        return x


class RecallModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        d_ff: int = 128,
        max_len: int = 96,
        attn: str = "nsa",
        block: int = 4,
        n_select: int = 2,
        window: int = 8,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        def make_attn():
            if attn == "nsa":
                return NSAAttention(d_model, num_heads, block, n_select, window)
            return FullAttention(d_model, num_heads)

        self.blocks = nn.ModuleList([Block(d_model, make_attn(), d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.attn_kind = attn
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, collect=False):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x, collect=collect)
        return self.head(self.norm(x))

    def attn_stats(self):
        stats = [b.attn.last_stats for b in self.blocks if hasattr(b.attn, "last_stats") and b.attn.last_stats]
        if not stats:
            return {}
        keys = stats[0].keys()
        return {k: sum(s[k] for s in stats) / len(stats) for k in keys}
