"""The Conformer block, from scratch (paper Section 2.2).

A Conformer block is a "sandwich" of four modules around a single input, each
applied as a pre-norm residual (Fig. 1):

    x = x + 1/2 * FFN(x)          # half-step feed-forward
    x = x + MHSA(x)               # multi-head self-attention
    x = x + Conv(x)               # convolution module (local modeling)
    x = x + 1/2 * FFN(x)          # half-step feed-forward
    y = LayerNorm(x)

The two feed-forward modules bracket the attention + convolution pair with a
half-step weight (the "Macaron" structure). Attention captures global content
while the convolution module captures local, fine-grained patterns — the paper's
central claim is that combining them beats either alone.

The ``use_attn`` / ``use_conv`` flags on the block let the demo run the small
ablation in Section 3.4 (removing a module and measuring the accuracy drop).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish / SiLU activation, x * sigmoid(x) (paper Section 2.2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    """Half-step position-wise feed-forward (Section 2.2, Fig. 4).

    LayerNorm -> Linear(expand) -> Swish -> Dropout -> Linear -> Dropout.
    The paper uses an expansion factor of 4 and pre-norm residual units.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """Pre-norm multi-head self-attention (Section 2.2, Fig. 3).

    The paper uses relative positional encodings (Transformer-XL style); we use
    standard absolute self-attention here to keep the reference minimal. The
    role in the block — global, content-based mixing across all frames — is
    identical.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y, self.attn_weights = self.attn(y, y, y, need_weights=True, average_attn_weights=True)
        return self.dropout(y)


class ConvolutionModule(nn.Module):
    """Convolution module (Section 2.2, Fig. 2).

    LayerNorm -> Pointwise Conv (expand ×2) -> GLU -> Depthwise Conv ->
    BatchNorm -> Swish -> Pointwise Conv -> Dropout. The depthwise conv gives
    each channel a local temporal receptive field; GLU gates the expansion.
    """

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for 'same' padding"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)  # -> GLU
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T) for 1-D convolutions.
        y = self.norm(x).transpose(1, 2)
        y = self.pointwise1(y)
        y = F.glu(y, dim=1)              # gated linear unit halves the channels
        y = self.depthwise(y)
        y = self.activation(self.batch_norm(y))
        y = self.pointwise2(y)
        y = self.dropout(y)
        return y.transpose(1, 2)         # back to (B, T, C)


class ConformerBlock(nn.Module):
    """One Conformer block (Section 2.2, Fig. 1)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        use_attn: bool = True,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.use_attn = use_attn
        self.use_conv = use_conv
        self.ffn1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        if self.use_attn:
            x = x + self.mhsa(x)
        if self.use_conv:
            x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Log-mel projection + a stack of Conformer blocks (Section 2.1)."""

    def __init__(
        self,
        n_mels: int,
        d_model: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        use_attn: bool = True,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList(
            ConformerBlock(
                d_model, num_heads, 4, conv_kernel, dropout,
                use_attn=use_attn, use_conv=use_conv,
            )
            for _ in range(num_blocks)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B, T, n_mels) -> encoded (B, T, d_model)."""
        x = self.input_proj(feats)
        for block in self.blocks:
            x = block(x)
        return x


class FrameClassifier(nn.Module):
    """Conformer encoder + a linear per-frame head (demo task)."""

    def __init__(self, n_mels: int, num_classes: int, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(n_mels, **encoder_kwargs)
        d_model = self.encoder.input_proj.out_features
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(feats))
