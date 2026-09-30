"""Hand-rolled STFT utilities (no torchaudio / librosa).

Mimi is trained with a multi-scale spectral reconstruction loss on top of the
time-domain loss. We implement a tiny magnitude-STFT from ``torch.stft`` (a core
torch op, not torchaudio) and a multi-resolution spectral loss built from it.
The same STFT produces the spectrogram baked into the visualization.
"""

from __future__ import annotations

from typing import List

import torch


def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Magnitude STFT of a waveform ``(B, L)`` -> ``(B, F, frames)``."""
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, center=True, return_complex=True,
    )
    return spec.abs()


def multi_scale_spectral_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: List[int] = (128, 256, 512),
) -> torch.Tensor:
    """Sum of L1 log-magnitude STFT distances over several window sizes.

    ``x``/``y`` are waveforms shaped ``(B, 1, L)`` or ``(B, L)``.
    """
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    loss = x.new_zeros(())
    for n_fft in scales:
        hop = n_fft // 4
        sx = stft_mag(x, n_fft, hop)
        sy = stft_mag(y, n_fft, hop)
        loss = loss + (sx - sy).abs().mean()
        loss = loss + (torch.log(sx + 1e-4) - torch.log(sy + 1e-4)).abs().mean()
    return loss / len(scales)
