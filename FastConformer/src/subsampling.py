"""Convolutional sub-sampling front ends (FastConformer §2.1).

The whole point of FastConformer is the sub-sampling module that sits at the
front of the encoder. The original Conformer uses **regular** convolutions to
downsample the frame rate by **4x** (10 ms -> 40 ms). FastConformer instead
uses **depthwise-separable** convolutions to downsample by **8x** (10 ms ->
80 ms), which (a) halves the number of tokens every later layer must attend
over and (b) makes the sub-sampling block itself far cheaper.

Both classes below map a log-mel feature sequence

    (batch, time, n_mels)  ->  (batch, time // factor, d_model)

so the rest of the encoder is identical and only the token count differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvSubsampling4x(nn.Module):
    """Baseline Conformer sub-sampling: two regular strided convolutions (4x)."""

    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.factor = 4
        self.layers = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T) for Conv1d over time, then back.
        x = self.layers(x.transpose(1, 2))
        return x.transpose(1, 2)


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel) followed by a pointwise 1x1 conv.

    This is the FastConformer replacement for the expensive full convolutions:
    a depthwise spatial filter + a cheap channel mixer, at a fraction of the
    multiply-adds of a dense convolution (FastConformer §2.1, change #2).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_ch,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableSubsampling8x(nn.Module):
    """FastConformer sub-sampling: three depthwise-separable strided convs (8x).

    Kernel size 9 and a modest channel count mirror the paper's design choices
    (FastConformer §2.1, changes #1, #3, #4).
    """

    def __init__(self, n_mels: int, d_model: int, kernel_size: int = 9) -> None:
        super().__init__()
        self.factor = 8
        # First layer lifts n_mels -> d_model with a pointwise mix; each of the
        # three layers strides by 2, for a total of 8x downsampling.
        self.layers = nn.Sequential(
            _DepthwiseSeparableConv(n_mels, d_model, kernel_size, stride=2),
            nn.ReLU(),
            _DepthwiseSeparableConv(d_model, d_model, kernel_size, stride=2),
            nn.ReLU(),
            _DepthwiseSeparableConv(d_model, d_model, kernel_size, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x.transpose(1, 2))
        return x.transpose(1, 2)
