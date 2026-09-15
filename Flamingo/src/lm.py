"""A tiny decoder-only language model — the *frozen* LM that Flamingo builds on.

Paper Section 2.2: Flamingo keeps a pretrained LM entirely frozen and interleaves
new trainable gated cross-attention layers between its blocks. Here we implement a
small GPT-style LM so we can (a) pretrain it on text, (b) freeze it, and (c) show
that adding vision through the gates is what makes it answer image questions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import FeedForward, MultiHeadAttention


class LMBlock(nn.Module):
    """Pre-norm Transformer decoder block: causal self-attention + FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, causal_mask=causal_mask)
        x = x + self.ff(self.norm2(x))
        return x


class DecoderLM(nn.Module):
    """Minimal GPT-style language model (the part that stays frozen)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
        max_len: int = 16,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [LMBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0)
        return self.token_embed(tokens) + self.pos_embed(pos)

    @staticmethod
    def causal_mask(t: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device)).view(1, 1, t, t)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Text-only forward (used to pretrain the LM). Returns logits."""
        x = self.embed(tokens)
        mask = self.causal_mask(tokens.size(1), tokens.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm_f(x))
