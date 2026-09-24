"""Hand-rolled log-Mel spectrogram front end (no torchaudio / librosa).

Conformer (Section 3.1) consumes "80-channel filterbanks features computed
from a 25 ms window with a stride of 10 ms". We reproduce exactly that kind of
front end from first principles: a short-time Fourier transform (``torch.stft``)
followed by a triangular Mel filterbank and a log. Everything here is plain
torch + math so it runs on CPU with only torch + numpy installed.
"""

from __future__ import annotations

import math

import torch


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # HTK mel scale (same convention as most speech toolkits).
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None
) -> torch.Tensor:
    """Triangular Mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    # Equally spaced points on the Mel scale, mapped back to Hz.
    m_min, m_max = _hz_to_mel(torch.tensor(f_min)), _hz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        up = (fft_freqs - left) / (center - left + 1e-9)
        down = (right - fft_freqs) / (right - center + 1e-9)
        fb[m - 1] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class MelFrontend:
    """Callable STFT -> Mel -> log front end with cached filterbank."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,       # 25 ms at 16 kHz
        hop_length: int = 160,  # 10 ms at 16 kHz
        n_mels: int = 40,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = torch.hann_window(n_fft)
        self.fb = mel_filterbank(n_mels, n_fft, sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (..., samples) -> log-mel (..., n_mels, frames)."""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.real**2 + spec.imag**2          # (..., n_freqs, frames)
        mel = torch.matmul(self.fb, power)           # (..., n_mels, frames)
        return torch.log(mel + 1e-6)


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
) -> torch.Tensor:
    """Convenience one-shot wrapper around :class:`MelFrontend`."""
    return MelFrontend(sample_rate, n_fft, hop_length, n_mels)(waveform)
