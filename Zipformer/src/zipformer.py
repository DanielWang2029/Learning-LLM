"""Minimal Zipformer encoder (Yao et al. 2023) — the parts that matter here.

Zipformer's efficiency comes from three ideas reproduced in this file:

  1. A **U-Net-like** encoder where the *middle* stacks operate at a **lower
     frame rate** (downsample → process cheaply → upsample), §3.1.
  2. **BiasNorm**, a simpler LayerNorm replacement that keeps length
     information via a learnable bias, §3.3.
  3. **Bypass** modules that learn a channel-wise blend of a module's input
     and output, §3.2.

Because attention is O(T²), running the middle of the network at half the
frame rate makes it ~4x cheaper there, which is where most of the FLOP savings
come from — with no loss in accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasNorm(nn.Module):
    """BiasNorm (Zipformer §3.3):  x / RMS[x - b] · exp(γ).

    Unlike LayerNorm, there is no mean subtraction, and the RMS is taken over
    the (bias-shifted) channels. The learnable bias ``b`` lets the network keep
    a large constant in one channel so that *length* information survives the
    normalization; the positive scale ``exp(γ)`` avoids the sign oscillation
    that a raw learnable scale can suffer.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.zeros(1))  # γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x - self.bias).pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return x / rms * torch.exp(self.log_scale)


class Bypass(nn.Module):
    """Learned channel-wise blend of module input and output (§3.2).

    ``(1 - c) ⊙ x + c ⊙ y`` with ``c`` initialized near 1.0 so the module is
    almost "straight-through" early in training, which the paper finds aids
    convergence.
    """

    def __init__(self, d_model: int, init: float = 0.9) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.full((d_model,), init))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c.clamp(0.0, 1.0)
        return (1.0 - c) * x + c * y


def downsample(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Average every ``factor`` frames (Zipformer's simplest Downsample)."""
    b, t, d = x.shape
    t2 = (t // factor) * factor
    x = x[:, :t2].reshape(b, t2 // factor, factor, d).mean(dim=2)
    return x


def upsample(x: torch.Tensor, out_len: int, factor: int = 2) -> torch.Tensor:
    """Repeat each frame ``factor`` times, then trim/pad to ``out_len``."""
    b, t, d = x.shape
    x = x.repeat_interleave(factor, dim=1)
    if x.size(1) < out_len:
        pad = x[:, -1:].expand(b, out_len - x.size(1), d)
        x = torch.cat([x, pad], dim=1)
    return x[:, :out_len]


class SelfAttention(nn.Module):
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
        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) / self.d_k ** 0.5, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out(ctx)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=kernel_size // 2, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.pw(F.silu(self.dw(y)))
        return self.dropout(y.transpose(1, 2))


class ZipformerBlock(nn.Module):
    """A block using BiasNorm + Bypass around attention / conv / feed-forward."""

    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout) -> None:
        super().__init__()
        self.norm_attn = BiasNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, dropout)
        self.norm_conv = BiasNorm(d_model)
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.norm_ff = BiasNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.bypass = Bypass(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = x + self.attn(self.norm_attn(x))
        x = x + self.conv(self.norm_conv(x))
        x = x + self.ff(self.norm_ff(x))
        return self.bypass(inp, x)


class ConstantRateEncoder(nn.Module):
    """Baseline: every block runs at the full frame rate."""

    def __init__(self, n_mels, d_model, num_heads, d_ff, num_blocks,
                 kernel_size, num_classes, dropout=0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList(
            ZipformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        )
        self.norm = BiasNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.num_blocks = num_blocks

    def forward(self, feats):
        x = self.proj(feats)
        for blk in self.blocks:
            x = blk(x)
        return self.classifier(self.norm(x))


class ZipformerEncoder(nn.Module):
    """U-Net encoder: a full-rate stem, a low-rate middle, a full-rate head.

    The middle ``num_middle`` blocks run on a sequence downsampled by 2, then
    the result is upsampled and blended back with a Bypass — matching the
    accuracy of the constant-rate encoder at a fraction of the attention FLOPs.
    """

    def __init__(self, n_mels, d_model, num_heads, d_ff, num_middle,
                 kernel_size, num_classes, downsample_factor=2, dropout=0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.stem = ZipformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
        self.middle = nn.ModuleList(
            ZipformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_middle)
        )
        self.head = ZipformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
        self.merge = Bypass(d_model)
        self.norm = BiasNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.factor = downsample_factor
        self.num_middle = num_middle

    def forward(self, feats):
        x = self.proj(feats)
        x = self.stem(x)                         # full rate
        skip = x
        y = downsample(x, self.factor)           # → low rate
        for blk in self.middle:
            y = blk(y)                           # cheap middle
        y = upsample(y, out_len=x.size(1), factor=self.factor)
        x = self.merge(skip, y)                  # blend low-rate result back
        x = self.head(x)                         # full rate
        return self.classifier(self.norm(x))
