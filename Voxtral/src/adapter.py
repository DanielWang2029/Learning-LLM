"""Audio-language adapter (Voxtral §2.2).

The encoder emits embeddings at 50 Hz, which would make the language decoder's
sequence enormous (a 30-minute audio would be 90k tokens). The adapter fixes
this by **concatenating 4 adjacent frames** and projecting the result with a
small MLP. That is a 4x downsampling: 50 Hz -> 12.5 Hz, the effective frame
rate reported in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Concatenate ``stride`` adjacent frames, then MLP-project to the LLM dim."""

    def __init__(
        self,
        d_encoder: int,
        d_llm: int,
        stride: int = 4,
        d_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        d_hidden = d_hidden or d_llm
        self.mlp = nn.Sequential(
            nn.Linear(d_encoder * stride, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_llm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_encoder) at 50 Hz -> (batch, T/stride, d_llm) at 12.5 Hz
        b, t, d = x.shape
        keep = (t // self.stride) * self.stride
        if keep == 0:
            raise ValueError(
                f"Need at least {self.stride} encoder frames, got {t}."
            )
        x = x[:, :keep].reshape(b, keep // self.stride, d * self.stride)
        return self.mlp(x)
