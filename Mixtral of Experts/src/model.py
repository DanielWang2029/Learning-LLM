"""A tiny decoder-only Transformer whose FFN is a sparse MoE (Mixtral-style).

Structure: token+position embedding -> one causal self-attention block ->
one **MoE** feed-forward block -> output head.  Keeping everything else minimal
puts the spotlight on the sparse routing, which is Mixtral's contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoELayer


@dataclass
class MoEConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 64          # experts dominate the params -> strong total/active gap
    n_experts: int = 8
    top_k: int = 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(out)


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoELayer(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        x = x + self.attn(self.ln1(x))
        moe_out, aux = self.moe(self.ln2(x))
        x = x + moe_out
        logits = self.head(self.ln_f(x))
        return logits, aux

    # --- parameter accounting for the "total vs active" comparison ------------
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params_per_token(self) -> int:
        """Params actually used for one token: everything except the unused experts."""
        expert_params = self.moe.expert_param_count()
        total_expert_params = expert_params * self.cfg.n_experts
        active_expert_params = expert_params * self.cfg.top_k
        return self.total_params() - total_expert_params + active_expert_params
