"""A tiny transformer with a 5:1 local:global layer pattern (Gemma 3, 2025).

Gemma 3 interleaves attention layers in a **5:1 ratio of local to global**: five
sliding-window layers for every one full-attention layer. That keeps the vast
majority of attention cheap while a few global layers still relay long-range
information across the whole sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention, mask_entries


def layer_pattern(n_layers: int, ratio: int = 5) -> list[str]:
    """Return the per-layer attention modes for a ``ratio``:1 local:global stack.

    Every ``ratio+1``-th layer is global (and the final layer is global so
    long-range info can always reach the output). Example (6 layers, ratio 5):
    ``[local, local, local, local, local, global]``.
    """
    pattern = []
    for i in range(1, n_layers + 1):
        pattern.append("global" if i % (ratio + 1) == 0 else "local")
    pattern[-1] = "global"  # ensure the last layer is global
    return pattern


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mode, window, d_ff, qk_norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, mode, window, qk_norm)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    """Small encoder-style transformer for a long-context retrieval task."""

    def __init__(self, vocab_size, n_classes, seq_len, d_model=32, n_heads=2,
                 n_layers=6, d_ff=64, window=4, pattern=None, qk_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.window = window
        self.pattern = pattern if pattern is not None else layer_pattern(n_layers)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, mode, window, d_ff, qk_norm)
            for mode in self.pattern
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens, query_pos):
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Read out the hidden state at each sequence's query position.
        h = x[torch.arange(b), query_pos]
        return self.head(h)

    def attention_entries(self) -> int:
        """Total allowed attention-score entries across all layers (memory proxy)."""
        return sum(mask_entries(self.seq_len, mode, self.window)
                   for mode in self.pattern)
