"""A tiny 1-layer causal language model with a swappable capacity dimension.

Both configurations share the SAME backbone (token/context embedding, one causal
self-attention layer, a readout) so that the comparison isolates *where* extra
parameters go:

  * "embeddings"  -> add N-gram Embedding tables (larger input capacity).
  * "experts"     -> add a sparse MoE FFN with more experts.

We match the two configurations to (approximately) the same total parameter
budget in the demo, then compare their loss/accuracy on the bigram-grammar task.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ngram_embedding import NGramEmbedding
from .moe import SparseMoE


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _sinusoidal(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    denom = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * denom)
    pe[:, 1::2] = torch.cos(pos * denom)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.w_o(torch.matmul(attn, v))


class TinyLM(nn.Module):
    """1-layer causal LM. ``strategy`` is 'embeddings' or 'experts'."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        strategy: str = "experts",
        max_len: int = 32,
        # embedding-scaling knobs
        ngram_orders: tuple = (2, 3),
        hash_vocab: int = 4096,
        # expert-scaling knobs
        n_experts: int = 1,
        top_k: int = 1,
        d_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        if strategy == "embeddings":
            self.embed = NGramEmbedding(vocab_size, d_model, ngram_orders, hash_vocab)
        elif strategy == "experts":
            self.embed = nn.Embedding(vocab_size, d_model)
        else:
            raise ValueError(strategy)

        self.register_buffer("pos", _sinusoidal(max_len, d_model), persistent=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SparseMoE(d_model, n_experts, top_k, d_hidden)
        self.norm_out = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) + self.pos[: tokens.size(1)]
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.readout(self.norm_out(x))
