"""The PaLM transformer block with PARALLEL attention + MLP.

Standard ("serialized") transformer blocks run the MLP on the *output* of the
attention sub-layer:

    y = x + Attention(LN(x))
    z = y + MLP(LN(y))

PaLM instead uses the parallel formulation (PaLM paper, Section 2, "Parallel
Layers"): both the attention and the MLP read from the *same* layer-normalized
input, and their outputs are summed onto the residual stream together:

    z = x + Attention(LN(x)) + MLP(LN(x))

A single shared LayerNorm feeds both branches. PaLM reported this is ~15%
faster at scale (the two input projections can be fused and the MLP no longer
waits on attention) with no quality loss at large scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiQueryAttention
from .feedforward import SwiGLU
from .normalization import LayerNormNoBias
from .rope import RotaryEmbedding


class PaLMBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, ffn_hidden: int, rope: RotaryEmbedding
    ) -> None:
        super().__init__()
        # One shared normalization feeds BOTH parallel branches.
        self.norm = LayerNormNoBias(dim)
        self.attn = MultiQueryAttention(dim, n_heads, rope)
        self.mlp = SwiGLU(dim, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.attn(h) + self.mlp(h)
