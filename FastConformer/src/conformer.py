"""A minimal Conformer encoder (Gulati et al. 2020) used by FastConformer.

FastConformer keeps the Conformer *block* unchanged and only swaps the
sub-sampling front end. So this file is a small, faithful Conformer block —
FFN / multi-head self-attention / convolution module / FFN with the classic
"half-step" residual feed-forwards — wired behind a pluggable sub-sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .subsampling import ConvSubsampling4x, DepthwiseSeparableSubsampling8x


class FeedForward(nn.Module):
    """Position-wise feed-forward with Swish, used as a half-step residual."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled-dot-product multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out(ctx)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise -> GLU -> depthwise -> pointwise."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise1(x), dim=1)
        x = F.silu(self.norm(self.depthwise(x)))
        x = self.dropout(self.pointwise2(x))
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """One Conformer block (two half-step FFNs around attention + conv)."""

    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout) -> None:
        super().__init__()
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)


class FastConformerEncoder(nn.Module):
    """Conformer encoder with a configurable sub-sampling factor (4x or 8x).

    ``subsampling="8x"`` selects the FastConformer depthwise-separable front
    end; ``"4x"`` selects the baseline Conformer front end. Everything after
    the sub-sampler is identical, so the two settings differ only in how many
    tokens the attention + conv blocks operate over.
    """

    def __init__(
        self,
        n_mels: int = 40,
        d_model: int = 64,
        num_heads: int = 2,
        d_ff: int = 128,
        num_blocks: int = 2,
        kernel_size: int = 9,
        num_classes: int = 4,
        subsampling: str = "8x",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if subsampling == "4x":
            self.subsampling = ConvSubsampling4x(n_mels, d_model)
        elif subsampling == "8x":
            self.subsampling = DepthwiseSeparableSubsampling8x(n_mels, d_model)
        else:
            raise ValueError(f"unknown subsampling {subsampling!r}")
        self.factor = self.subsampling.factor
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, n_mels) -> per-token class logits (B, T//factor, num_classes)."""
        x = self.subsampling(feats)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)
