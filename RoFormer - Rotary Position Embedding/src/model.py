"""A tiny causal attention model used to compare RoPE vs. no positional info.

The task in the demo ("predict the token that appeared k positions earlier")
can only be solved if the model knows *relative* positions. This model has one
causal self-attention layer; the ``pos`` flag switches between:

    "rope" : rotary position embedding applied to Q and K   (RoFormer, §3)
    "none" : no positional information at all (pure attention is permutation
             equivariant, so it cannot know which token is k steps back)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, rope_cache


class RoPEAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_head: int = 2,
                 pos: str = "rope", base: float = 10000.0) -> None:
        super().__init__()
        assert d_model % n_head == 0
        assert pos in ("rope", "none")
        self.pos = pos
        self.base = base
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.emb = nn.Embedding(vocab_size, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                ignore_index: int = -100):
        B, T = idx.shape
        x = self.emb(idx)
        q = self.q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if self.pos == "rope":
            # RoPE is applied to Q and K only; note the cache is built for the
            # current sequence length T, so evaluating at a *longer* T than seen
            # in training just extends the same rotation rule (extrapolation).
            cos, sin = rope_cache(T, self.d_head, self.base)
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        y = self.proj(y)
        logits = self.head(self.ln(y))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss
