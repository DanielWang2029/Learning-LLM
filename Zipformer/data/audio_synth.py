"""Synthetic audio + hand-rolled log-mel features (numpy only).

No real audio is downloaded. We synthesize short waveforms from speech-like
primitives — sine tones, linear chirps, stacked "formant" resonances and white
noise — and convert them to log-mel spectrograms with a from-scratch STFT.
This keeps the pipeline dependency-free (numpy + stdlib) and CPU-only.

For Zipformer we use a frame-classification task at the *full* frame rate:
every frame belongs to one acoustic class, and the encoder must label them all.
Because Zipformer's middle stacks operate at a lower frame rate, the task also
exercises the downsample→low-rate→upsample structure directly.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

CLASS_NAMES = ["low_tone", "high_tone", "chirp", "noise"]
NUM_CLASSES = len(CLASS_NAMES)


def sine_tone(freq: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def chirp(f0: float, f1: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    k = (f1 - f0) / max(dur, 1e-6)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def formant_stack(freqs, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    wave = np.zeros_like(t)
    for i, f in enumerate(freqs):
        wave += (0.8 ** i) * np.sin(2 * np.pi * f * t)
    return wave


def white_noise(dur: float, sr: int = SAMPLE_RATE, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.standard_normal(int(dur * sr))


def _segment_for_class(cls: int, dur: float, rng) -> np.ndarray:
    if cls == 0:
        base = rng.uniform(180, 320)
        wave = formant_stack([base, 2 * base, 3 * base], dur)
    elif cls == 1:
        base = rng.uniform(1400, 2200)
        wave = formant_stack([base, 1.5 * base], dur)
    elif cls == 2:
        wave = chirp(rng.uniform(300, 600), rng.uniform(2500, 3800), dur)
    else:
        wave = white_noise(dur, rng=rng)
    wave = wave * rng.uniform(0.7, 1.0) + 0.02 * white_noise(dur, rng=rng)
    return wave.astype(np.float32)


def synth_utterance(num_segments: int, seg_dur: float, rng):
    waves, labels = [], []
    for _ in range(num_segments):
        cls = int(rng.integers(0, NUM_CLASSES))
        seg = _segment_for_class(cls, seg_dur, rng)
        waves.append(seg)
        labels.append(np.full(len(seg), cls, dtype=np.int64))
    return np.concatenate(waves), np.concatenate(labels)


# --------------------------- from-scratch log-mel ---------------------------
def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_power(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = _hann(n_fft)
    n_frames = 1 + max(0, (len(wave) - n_fft) // hop)
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = wave[i * hop : i * hop + n_fft] * window
        spec = np.fft.rfft(seg)
        out[i] = spec.real ** 2 + spec.imag ** 2
    return out


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


def log_mel(wave: np.ndarray, n_fft: int = 400, hop: int = 160, n_mels: int = 40):
    power = stft_power(wave, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft)
    return np.log(power @ fb.T + 1e-6).astype(np.float32), hop


def frame_labels(sample_labels: np.ndarray, n_fft: int, hop: int, n_frames: int):
    out = np.empty(n_frames, dtype=np.int64)
    for i in range(n_frames):
        seg = sample_labels[i * hop : i * hop + n_fft]
        out[i] = np.bincount(seg, minlength=NUM_CLASSES).argmax()
    return out
