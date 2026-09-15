"""Gated cross-attention ("GATED XATTN-DENSE", paper Section 2.2, Figure 4).

This is Flamingo's key contribution. Between the frozen LM blocks, we insert new
trainable layers that let text tokens attend to the visual tokens. Crucially,
each cross-attention and feed-forward sub-layer is multiplied by tanh(alpha),
where alpha is a **scalar gate initialized to 0**:

    y = x + tanh(alpha_attn) * cross_attn(x, visual_tokens)
    y = y + tanh(alpha_ff)   * ffn(y)

At initialization tanh(0) = 0, so the layers are the identity and the whole model
is *exactly* the pretrained LM. As training proceeds the gates open (|alpha| grows),
smoothly blending visual information into the language model without destabilizing
the frozen weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import FeedForward, MultiHeadAttention


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        # Tanh gates, initialized to 0 -> the block starts as a no-op (identity).
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        # x: (B, T_text, d) ; visual_tokens: (B, R, d)
        attended = self.attn(self.norm_x(x), self.norm_v(visual_tokens))
        x = x + torch.tanh(self.attn_gate) * attended
        x = x + torch.tanh(self.ff_gate) * self.ff(self.norm_ff(x))
        return x

    def gate_values(self) -> tuple[float, float]:
        return (
            float(torch.tanh(self.attn_gate).item()),
            float(torch.tanh(self.ff_gate).item()),
        )
