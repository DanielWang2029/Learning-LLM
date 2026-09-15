"""A tiny hybrid Mamba-Attention stack (Nemotron 3 Super layer pattern).

Nemotron 3 Super interleaves many Mamba-2 (SSM) layers with a few attention
layers. Here we build a small stack from a layer *pattern* string, e.g.
``"MMA"`` = SSM, SSM, Attention, repeated. Each layer is wrapped in a pre-norm
residual, matching the paper's block structure (MoE is omitted for size).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .attention import CausalAttention
from .ssm import SelectiveSSM


class HybridModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        pattern: str = "MMA",
        repeats: int = 2,
        n_heads: int = 4,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layer_kinds: List[str] = list(pattern) * repeats
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for kind in self.layer_kinds:
            if kind == "M":
                self.layers.append(SelectiveSSM(d_model, d_state=d_state))
            elif kind == "A":
                self.layers.append(CausalAttention(d_model, n_heads=n_heads))
            else:
                raise ValueError(f"unknown layer kind {kind!r} (use 'M' or 'A')")
            self.norms.append(nn.LayerNorm(d_model))
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))  # pre-norm residual
        return self.head(self.norm_f(h))

    def describe(self) -> str:
        names = {"M": "Mamba-SSM", "A": "Attention"}
        return " → ".join(names[k] for k in self.layer_kinds)
