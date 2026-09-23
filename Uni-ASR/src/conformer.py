"""Causal Conformer audio encoder (Uni-ASR §2.1, audio encoder).

Uni-ASR uses a Conformer encoder to extract speech representations. For
*streaming*, the encoder must be causal: a frame may only depend on the audio
that has already arrived, never the future. We therefore make every time-mixing
op causal:

* self-attention uses a causal (lower-triangular) mask, and
* the convolution module uses a causal depthwise conv (left padding only).

With a causal encoder, encoding the whole clip once and then slicing the frames
is identical to encoding each streaming chunk as it arrives — which is exactly
what the streaming demo relies on.

Each Conformer block is the standard macaron sandwich:
    ½·FFN → MHSA → ConvModule → ½·FFN → LayerNorm
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    enc = torch.zeros(length, dim, device=device)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])
    enc[:, 1::2] = torch.cos(angle[:, 1::2])
    return enc


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalConvModule(nn.Module):
    """Conformer conv module with a causal depthwise convolution."""

    def __init__(self, d_model: int, kernel_size: int = 15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)   # + GLU
        self.pad = kernel_size - 1                             # left-pad only
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)          # (B, d, T)
        y = nn.functional.glu(self.pointwise1(y), dim=1)
        y = nn.functional.pad(y, (self.pad, 0))   # causal: pad past only
        y = self.depthwise(y)
        y = self.act(self.bn(y))
        y = self.pointwise2(y)
        return y.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, conv_kernel: int = 15) -> None:
        super().__init__()
        self.ffn1 = FeedForward(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.conv = CausalConvModule(d_model, conv_kernel)
        self.ffn2 = FeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        h = self.attn_norm(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class ConvSubsampling(nn.Module):
    """Two stride-2 convolutions -> temporal /4 (100 Hz Mel -> 25 Hz frames)."""

    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        return x.transpose(1, 2)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        conv_kernel: int = 15,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.subsample = ConvSubsampling(n_mels, d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, num_heads, conv_kernel) for _ in range(num_layers)]
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.subsample(mel)                             # (B, T, d)
        pos = sinusoidal_positions(x.size(1), self.d_model, x.device)
        x = x + pos.unsqueeze(0)
        causal = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for blk in self.blocks:
            x = blk(x, causal)
        return x
