"""Whisper-style audio encoder (Voxtral §2.1).

A log-Mel spectrogram passes through a two-layer convolutional stem that
downsamples the temporal resolution by a factor of two (100 Hz -> 50 Hz), then
through a stack of *bidirectional* Transformer self-attention layers. The
resulting audio embeddings have a 50 Hz frame rate, exactly as in the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    """Fixed sinusoidal positional encoding, reset per 30 s chunk in Voxtral."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    enc = torch.zeros(length, dim, device=device)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])
    enc[:, 1::2] = torch.cos(angle[:, 1::2])
    return enc


class ConvStem(nn.Module):
    """Two 1-D convolutions; the second has stride 2 (temporal /2)."""

    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (batch, n_mels, n_frames) -> (batch, d_model, n_frames/2)
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        return x.transpose(1, 2)  # (batch, n_frames/2, d_model)


class WhisperEncoder(nn.Module):
    """Conv stem + bidirectional Transformer encoder producing 50 Hz frames."""

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.stem = ConvStem(n_mels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)                                  # (B, T50, d)
        pos = sinusoidal_positions(x.size(1), self.d_model, x.device)
        x = x + pos.unsqueeze(0)
        x = self.transformer(x)                             # bidirectional
        return self.norm(x)
