"""A small decoder-only GPT-3-style language model.

Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020). The
architecture is the same decoder-only Transformer as GPT-2; GPT-3's
contribution is the demonstration that, at scale, such a model performs new
tasks *in-context*: you place a few input→output demonstrations in the prompt
and the model infers the pattern and completes a query — with **no gradient
updates**. This repo shows the mechanism at tiny scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 256
    max_len: int = 128
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(self.hidden_states(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits for the token following the given context (last position)."""
        logits, _ = self(idx[:, -self.cfg.max_len :])
        return logits[:, -1]


__all__ = ["GPT", "GPTConfig", "Block"]
