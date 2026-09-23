"""Hand-rolled log-Mel front end for Whisper (no torchaudio / librosa).

Whisper (Section 2.2) uses "an 80-channel log-magnitude Mel spectrogram
representation on 25-millisecond windows with a stride of 10 milliseconds". We
reproduce exactly that: an STFT via ``torch.stft``, a triangular Mel filterbank,
a log, and Whisper's normalization (clamp to the dynamic range, then scale to
roughly [-1, 1]).
"""

from __future__ import annotations

import torch


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    mel_pts = torch.linspace(
        _hz_to_mel(torch.tensor(0.0)).item(),
        _hz_to_mel(torch.tensor(sample_rate / 2)).item(),
        n_mels + 2,
    )
    hz_pts = _mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up = (fft_freqs - left) / (center - left + 1e-9)
        down = (right - fft_freqs) / (right - center + 1e-9)
        fb[m - 1] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class MelFrontend:
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = torch.hann_window(n_fft)
        self.fb = mel_filterbank(n_mels, n_fft, sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (..., samples) -> log-mel (..., n_mels, frames)."""
        spec = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True, return_complex=True,
        )
        power = spec.real**2 + spec.imag**2
        mel = torch.matmul(self.fb, power)
        log_spec = torch.clamp(mel, min=1e-10).log10()
        # Whisper normalization: clip 8 dB below the max, then scale to ~[-1, 1].
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0


def log_mel_spectrogram(waveform, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
    return MelFrontend(sample_rate, n_fft, hop_length, n_mels)(waveform)
