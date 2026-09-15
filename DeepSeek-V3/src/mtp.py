"""Multi-Token Prediction (MTP) — DeepSeek-V3, §2.2 "Multi-Token Prediction".

Instead of only predicting the immediate next token, DeepSeek-V3 adds one or more
**MTP modules** that each predict a *further* future token, densifying the
training signal and enabling speculative decoding at inference. Each MTP module is
a small transformer-style block that takes the previous representation plus the
embedding of the already-known next token and predicts the token one step further
ahead.

Here we implement a single MTP module that predicts token ``t+2`` (the main head
predicts ``t+1``). It combines the trunk hidden state at position ``t`` with the
embedding of the true token at ``t+1`` (teacher-forced during training) and runs a
small feed-forward before sharing the main output head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPModule(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_h = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)
        # combine [trunk hidden ; next-token embedding] -> a refined hidden state
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, hidden: torch.Tensor, next_token_emb: torch.Tensor) -> torch.Tensor:
        """hidden, next_token_emb: (b, t, dim). Returns a hidden state for t+2."""
        combined = torch.cat([self.norm_h(hidden), self.norm_e(next_token_emb)], dim=-1)
        h = self.proj(combined)
        return h + self.ffn(h)
