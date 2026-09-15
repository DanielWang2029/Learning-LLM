"""A tiny long-context Transformer with sparse MoE layers (Gemini 1.5).

This model ties together Gemini 1.5's two headline documented ideas:

* **Sparse Mixture-of-Experts** feed-forward layers (``moe.py``), and
* **long-context retrieval** — the attention uses RoPE so it can resolve
  positions in contexts far longer than any single training example, which is
  what lets it pass the needle-in-a-haystack test in the demo.

Architecture is a minimal decoder-only Transformer (RMSNorm pre-norm, RoPE
attention, MoE MLP). It is trained on an associative-recall task: read a long list
of key→value pairs, then answer a query about one planted key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import SparseMoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rope(seq_len, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_base: float = 10000.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rope(t, self.head_dim, self.rope_base, x.device)
        cos, sin = cos[None, None], sin[None, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(y)


@dataclass
class GeminiConfig:
    vocab_size: int = 43
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    moe_hidden: int = 128
    n_experts: int = 4
    top_k: int = 2
    rope_base: float = 10000.0
    tie_embeddings: bool = True  # ties unembedding to embedding -> easy value copy


class Block(nn.Module):
    def __init__(self, cfg: GeminiConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.n_heads, cfg.rope_base)
        self.moe_norm = RMSNorm(cfg.dim)
        self.moe = SparseMoE(cfg.dim, cfg.moe_hidden, cfg.n_experts, cfg.top_k)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        moe_out, info = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, info


class GeminiMoE(nn.Module):
    def __init__(self, cfg: GeminiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        x = self.tok(idx)
        aux_total = torch.zeros((), device=idx.device)
        loads = []
        for blk in self.blocks:
            x, info = blk(x)
            aux_total = aux_total + info.aux_loss
            loads.append(info.load)
        logits = self.head(self.norm_f(x))
        return logits, aux_total, loads
