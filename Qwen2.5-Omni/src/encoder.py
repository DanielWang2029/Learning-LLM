"""Block-wise streaming audio encoder (Qwen2.5-Omni §2.1 & §2.4).

To support chunked prefill for streaming, Qwen2.5-Omni changes its audio encoder
"from full attention over the entire audio to performing attention in blocks of
2 seconds each" (block-wise / block-diagonal attention along time). This
decouples perception (the encoder) from long-sequence modeling (the LLM): each
~2 s block is encoded independently, so audio can be encoded incrementally with
bounded, block-sized latency.

This module implements one encoder that runs in either mode:

* ``block_frames=None``  -> full attention (the original, O(T²)) behaviour.
* ``block_frames=k``     -> block-diagonal attention over k-frame blocks, which
                            is exactly equivalent to encoding each block on its
                            own (see ``streaming_encode``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import HOP_LENGTH, SAMPLE_RATE

# The conv stem downsamples the 100 Hz Mel frames by 4 -> 25 Hz encoder frames,
# i.e. one encoder frame per 40 ms of audio.
CONV_DOWNSAMPLE = 4
FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE * CONV_DOWNSAMPLE  # 0.04 s


def frames_per_seconds(seconds: float) -> int:
    """Number of encoder frames spanning ``seconds`` of audio (e.g. 2 s -> 50)."""
    return max(1, round(seconds / FRAME_SECONDS))


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    enc = torch.zeros(length, dim, device=device)
    enc[:, 0::2] = torch.sin(angle[:, 0::2])
    enc[:, 1::2] = torch.cos(angle[:, 1::2])
    return enc


def block_diagonal_mask(length: int, block: int, device=None) -> torch.Tensor:
    """(length, length) bool mask; True = *disallowed*. Frames attend only
    within their own block (block-diagonal)."""
    idx = torch.arange(length, device=device)
    same_block = (idx.unsqueeze(0) // block) == (idx.unsqueeze(1) // block)
    return ~same_block


class ConvStem(nn.Module):
    """Two stride-2 convolutions -> temporal /4 (100 Hz Mel -> 25 Hz frames)."""

    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        return x.transpose(1, 2)  # (batch, frames, d_model)


class BlockwiseAudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.stem = ConvStem(n_mels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _frames(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        pos = sinusoidal_positions(x.size(1), self.d_model, x.device)
        return x + pos.unsqueeze(0)

    def forward(self, mel: torch.Tensor, block_frames: int | None = None) -> torch.Tensor:
        """Encode a spectrogram. ``block_frames=None`` -> full attention."""
        x = self._frames(mel)
        mask = None
        if block_frames is not None:
            mask = block_diagonal_mask(x.size(1), block_frames, x.device)
        x = self.transformer(x, mask=mask)
        return self.norm(x)

    @torch.no_grad()
    def streaming_encode(self, mel: torch.Tensor, block_frames: int):
        """Encode block by block, as in streaming prefill.

        Returns (embeddings, per_block_frame_counts). Because attention is
        block-diagonal, encoding each block alone is mathematically identical to
        one masked full pass — the point of block-wise attention.
        """
        x = self._frames(mel)                       # (1, T, d)
        outs, counts = [], []
        for start in range(0, x.size(1), block_frames):
            chunk = x[:, start : start + block_frames]
            outs.append(self.transformer(chunk))    # independent block, no mask
            counts.append(chunk.size(1))
        return self.norm(torch.cat(outs, dim=1)), counts
