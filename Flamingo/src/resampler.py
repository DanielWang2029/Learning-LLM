"""Perceiver Resampler (paper Section 2.1, Figure 3 inset).

A small, fixed set of *learned latent queries* attends to the (variable-length)
visual features from the vision encoder and produces a **fixed** number of
"visual tokens" (R of them), regardless of the input resolution. This keeps the
downstream cross-attention cheap and constant-cost.

    latents (R, d)  --cross-attend to-->  vision features (N, d)  -->  R visual tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import FeedForward, MultiHeadAttention


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_latents: int = 8,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        # Learned latent queries — the "resampled" visual tokens start here.
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn": MultiHeadAttention(d_model, num_heads),
                    "ff": FeedForward(d_model, d_ff),
                    "norm_q": nn.LayerNorm(d_model),
                    "norm_kv": nn.LayerNorm(d_model),
                    "norm_ff": nn.LayerNorm(d_model),
                })
            )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """visual_features: (B, N, d) -> visual tokens (B, R, d)."""
        b = visual_features.size(0)
        x = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, R, d)
        for layer in self.layers:
            # Latents attend to [visual features ++ latents] (Perceiver-style).
            kv = torch.cat([layer["norm_kv"](visual_features), layer["norm_q"](x)], dim=1)
            x = x + layer["attn"](layer["norm_q"](x), kv)
            x = x + layer["ff"](layer["norm_ff"](x))
        return x
