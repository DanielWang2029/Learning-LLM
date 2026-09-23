"""Synthetic audio + hand-rolled log-mel features (numpy only).

No real audio is downloaded. Waveforms are synthesized from speech-like
primitives — sine tones, linear chirps, stacked "formant" resonances and white
noise — and converted to log-mel spectrograms with a from-scratch STFT, so the
whole pipeline is dependency-free (numpy + stdlib) and CPU-only.

For the cache-aware streaming Conformer we use a frame-classification task.
Frames near a segment boundary are ambiguous, so *look-ahead* (a larger chunk)
genuinely helps accuracy — which is exactly the latency/accuracy trade-off the
paper exposes from a single streaming model.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

CLASS_NAMES = ["low_tone", "high_tone", "chirp", "noise"]
NUM_CLASSES = len(CLASS_NAMES)


def sine_tone(freq, dur, sr=SAMPLE_RATE):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def chirp(f0, f1, dur, sr=SAMPLE_RATE):
    t = np.arange(int(dur * sr)) / sr
    k = (f1 - f0) / max(dur, 1e-6)
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def formant_stack(freqs, dur, sr=SAMPLE_RATE):
    t = np.arange(int(dur * sr)) / sr
    wave = np.zeros_like(t)
    for i, f in enumerate(freqs):
        wave += (0.8 ** i) * np.sin(2 * np.pi * f * t)
    return wave


def white_noise(dur, sr=SAMPLE_RATE, rng=None):
    rng = rng or np.random.default_rng()
    return rng.standard_normal(int(dur * sr))


def _segment_for_class(cls, dur, rng):
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
    wave = wave * rng.uniform(0.7, 1.0) + 0.05 * white_noise(dur, rng=rng)
    return wave.astype(np.float32)


def synth_utterance(num_segments, seg_dur, rng):
    """Random sequence of labeled segments with slightly varied durations."""
    waves, labels = [], []
    for _ in range(num_segments):
        cls = int(rng.integers(0, NUM_CLASSES))
        dur = seg_dur * rng.uniform(0.8, 1.2)
        seg = _segment_for_class(cls, dur, rng)
        waves.append(seg)
        labels.append(np.full(len(seg), cls, dtype=np.int64))
    return np.concatenate(waves), np.concatenate(labels)


# --------------------------- from-scratch log-mel ---------------------------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_power(wave, n_fft, hop):
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


def mel_filterbank(n_mels, n_fft, sr=SAMPLE_RATE):
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


def log_mel(wave, n_fft=400, hop=160, n_mels=40):
    power = stft_power(wave, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft)
    return np.log(power @ fb.T + 1e-6).astype(np.float32), hop


def frame_labels(sample_labels, n_fft, hop, n_frames):
    out = np.empty(n_frames, dtype=np.int64)
    for i in range(n_frames):
        seg = sample_labels[i * hop : i * hop + n_fft]
        out[i] = np.bincount(seg, minlength=NUM_CLASSES).argmax()
    return out
