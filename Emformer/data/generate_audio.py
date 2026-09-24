"""Synthesize an acoustic feature sequence for the Emformer demo (numpy only).

Emformer is an acoustic *encoder* layer, so its input is a sequence of feature
frames. We synthesize a waveform of drifting formant-like tones (numpy sine
stacks), then compute a small log-Mel spectrogram from scratch with ``numpy.fft``
— no torchaudio / librosa. The result is a structured ``(T, n_mels)`` sequence
where nearby frames are correlated and distant frames differ, so limiting the
attention context has a measurable effect.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels, n_fft, sr):
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mel_pts = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up = (fft_freqs - l) / (c - l + 1e-9)
        down = (r - fft_freqs) / (r - c + 1e-9)
        fb[m - 1] = np.clip(np.minimum(up, down), 0, None)
    return fb


def synth_waveform(rng: np.random.Generator, duration: float = 1.0) -> np.ndarray:
    """Two formants that sweep *monotonically* across the utterance + noise.

    A monotonic (non-repeating) sweep makes acoustically similar frames also be
    temporally close, so full-context attention is dominated by nearby frames —
    exactly the regime where Emformer's bounded left/right context + memory
    closely approximates full attention, and the gap shrinks as context grows.
    """
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    p = t / (t[-1] + 1e-9)                              # 0 -> 1 progress
    f1 = 250 + 500 * p                                 # rising first formant
    f2 = 1800 - 700 * p                                # falling second formant
    wave = np.sin(2 * np.pi * np.cumsum(f1) / SAMPLE_RATE)
    wave += 0.6 * np.sin(2 * np.pi * np.cumsum(f2) / SAMPLE_RATE)
    wave += 0.03 * rng.standard_normal(n)
    return (wave / (np.max(np.abs(wave)) + 1e-8)).astype(np.float32)


def log_mel(wave: np.ndarray, n_mels: int = 32, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """numpy STFT -> Mel -> log, returns (T, n_mels)."""
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(wave) - n_fft) // hop
    spec = np.empty((n_fft // 2 + 1, frames), np.float32)
    for i in range(frames):
        seg = wave[i * hop : i * hop + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(seg)) ** 2
    fb = _mel_filterbank(n_mels, n_fft, SAMPLE_RATE)
    mel = fb @ spec
    logmel = np.log(mel + 1e-6).T                      # (T, n_mels)
    # per-dimension standardization
    logmel = (logmel - logmel.mean(0)) / (logmel.std(0) + 1e-5)
    return logmel.astype(np.float32)


def make_feature_sequence(rng: np.random.Generator, n_mels: int = 32, duration: float = 1.0):
    return log_mel(synth_waveform(rng, duration), n_mels=n_mels)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    feats = make_feature_sequence(rng)
    print(f"feature sequence: {feats.shape} (frames, mels)  sample_rate={SAMPLE_RATE}")
