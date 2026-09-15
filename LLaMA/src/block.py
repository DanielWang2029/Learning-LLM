"""A LLaMA transformer block: pre-norm attention, then pre-norm feed-forward.

LLaMA uses *pre-normalization* (LLaMA paper, Section 2.2): the normalization is
applied to the input of each sub-layer, and the residual connection carries the
un-normalized signal. This is more stable than the original post-norm design.

    h = x + Attention(RMSNorm(x))
    y = h + FeedForward(RMSNorm(h))

Both the norm class and the feed-forward class are injected so the demo can
ablate them (RMSNorm↔LayerNorm, SwiGLU↔ReLU) without touching this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, attn: nn.Module, ffn: nn.Module, norm_cls) -> None:
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = norm_cls(dim)
        self.ffn_norm = norm_cls(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
