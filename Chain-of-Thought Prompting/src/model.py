"""A tiny decoder-only Transformer (GPT-style) used as the "language model".

The Chain-of-Thought paper studies prompting behaviour in very large models.
We cannot run those on CPU, so this file provides a faithful but *miniature*
autoregressive Transformer that we train from scratch. The point of the demo
is not scale — it is to show that the SAME small model reaches far higher
exact-match accuracy when it is trained to emit an intermediate reasoning
"scratchpad" (chain of thought) instead of jumping straight to the answer.

Architecture is the standard decoder stack from "Attention Is All You Need"
(masked multi-head self-attention + position-wise MLP + residual/LayerNorm),
with a learned positional embedding and a language-model head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention (decoder attention)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = attn @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """One Transformer decoder block: attention + MLP, both residual + norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """A compact decoder-only language model over a character vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, T, vocab)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy (argmax) decoding, stopping when every row emits ``eos_id``."""
        self.eval()
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size :])[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None:
                done = done | (nxt.squeeze(1) == eos_id)
                if bool(done.all()):
                    break
        return idx
