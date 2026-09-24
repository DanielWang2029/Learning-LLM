"""From-scratch log-Mel spectrogram front-end (Uni-ASR audio encoder input).

Uni-ASR uses a Conformer audio encoder over log-Mel features. We implement the
log-Mel front-end with numpy only — no torchaudio, no librosa:

    waveform -> framing (Hann, 25 ms) -> rFFT -> power -> Mel filterbank -> log

At 16 kHz with a 10 ms hop this gives 100 Mel frames/second; the encoder's conv
stem then downsamples by 4 to 25 Hz (one frame per 40 ms).
"""

from __future__ import annotations

import numpy as np

N_FFT = 400          # 25 ms window at 16 kHz
HOP_LENGTH = 160     # 10 ms hop -> 100 frames/second
N_MELS = 80          # Conformer ASR encoders commonly use 80 Mel bins
SAMPLE_RATE = 16_000


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular Mel filterbank of shape (n_mels, n_fft//2 + 1)."""
    fmax = fmax or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    mel_min, mel_max = _hz_to_mel(np.array([fmin])), _hz_to_mel(np.array([fmax]))
    mel_points = np.linspace(mel_min[0], mel_max[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        fb[m - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


_FILTERBANK = mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Return a (n_mels, n_frames) log-Mel spectrogram for a mono waveform."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.shape[0] < N_FFT:
        x = np.pad(x, (0, N_FFT - x.shape[0]))

    n_frames = 1 + (x.shape[0] - N_FFT) // HOP_LENGTH
    frames = np.stack(
        [x[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] for i in range(n_frames)]
    )
    frames = frames * _WINDOW

    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

    mel = power @ _FILTERBANK.T
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel.T.astype(np.float32)       # (n_mels, n_frames)
