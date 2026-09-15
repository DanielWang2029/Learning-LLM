"""A tiny decoder-only (GPT-style) language model.

This is the policy network used throughout the RLHF pipeline of InstructGPT
(Ouyang et al., 2022). It is deliberately minimal: a couple of causal
self-attention blocks over a tiny vocabulary, enough to learn the toy
"sort these tokens" task on CPU in a few seconds.

The same architecture backs the supervised fine-tuned (SFT) model of Stage 1
and the RL policy of Stage 3 — in the paper these all share the GPT-3
backbone, here they share this `TinyLM`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (autoregressive) mask."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (b, heads, t, d_k)
        q = q.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = attn @ v  # (b, heads, t, d_k)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm Transformer block: attention + position-wise feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """A minimal GPT-style autoregressive language model."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map token ids (b, t) to next-token logits (b, t, vocab)."""
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        num_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressively extend `prefix` (b, t0) by `num_new_tokens` tokens."""
        idx = prefix
        for _ in range(num_new_tokens):
            logits = self(idx[:, -self.max_len :])[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def sequence_logprob(
        self, full: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Sum of per-token log-probs over the response region.

        `full` is (b, T) containing [BOS, prompt, SEP, response, EOS].
        `response_mask` is (b, T) marking which *target* positions belong to the
        response (see `data.build_response_mask`). Returns (b,) log π(response|prompt).
        """
        logits = self(full[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        targets = full[:, 1:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = response_mask[:, 1:].float()
        return (token_lp * mask).sum(dim=-1)
