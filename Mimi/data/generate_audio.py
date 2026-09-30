"""Synthetic "speech" for the Mimi demo (numpy only — no real audio, no downloads).

We need a signal where a *semantic* attribute (which "vowel" is spoken) is cleanly
separable from *acoustic* detail (pitch / loudness / noise), so we can show that
Mimi's split RVQ puts the vowel identity into the distilled **semantic** codebook
while the **acoustic** RVQ levels carry the rest.

We use a tiny source-filter vowel model:

* Source: a harmonic series at a random fundamental ``f0`` (the "speaker pitch").
* Filter: a formant envelope — a sum of Gaussian bumps at class-specific formant
  frequencies. The *pattern* of formants defines the vowel **class**; the pitch
  and loudness are acoustic detail independent of the class.

Everything is at a scaled-down 2.4 kHz "24 kHz-style" rate so the codec's 192x
downsampling lands on exactly a 12.5 Hz token rate (2400 / 192 = 12.5), matching
the paper, while staying fast on CPU.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 2400          # scaled-down "24 kHz-style" rate (24000 / 10)
TOTAL_STRIDE = 192          # encoder downsampling (2*4*4*6); 2400/192 = 12.5 Hz
FRAME_RATE = SAMPLE_RATE / TOTAL_STRIDE   # = 12.5 Hz, the paper's token rate
FRAMES_PER_PHONEME = 4      # each vowel spans 4 tokens (320 ms)
PHONEMES_PER_CLIP = 6
FRAMES_PER_CLIP = FRAMES_PER_PHONEME * PHONEMES_PER_CLIP   # 24 tokens / clip
CLIP_SAMPLES = FRAMES_PER_CLIP * TOTAL_STRIDE              # 4608 samples (1.92 s)

NUM_CLASSES = 4             # "vowels": distinct formant patterns

# Formant frequencies (Hz) per class — well separated, below Nyquist (1200 Hz).
CLASS_FORMANTS = np.array(
    [
        [200.0, 750.0],     # class 0
        [350.0, 500.0],     # class 1
        [280.0, 950.0],     # class 2
        [500.0, 850.0],     # class 3
    ],
    dtype=np.float32,
)
FORMANT_BW = 90.0           # Gaussian formant bandwidth (Hz)


def _formant_gain(freqs: np.ndarray, formants: np.ndarray) -> np.ndarray:
    """Vowel filter: sum of Gaussian bumps at the class formant frequencies."""
    gain = np.zeros_like(freqs)
    for fc in formants:
        gain += np.exp(-0.5 * ((freqs - fc) / FORMANT_BW) ** 2)
    return gain + 0.02


def _phoneme(cls: int, f0: float, gain: float, rng: np.random.Generator) -> np.ndarray:
    """One vowel: harmonic source at ``f0`` shaped by the class formant filter."""
    n = FRAMES_PER_PHONEME * TOTAL_STRIDE
    t = np.arange(n) / SAMPLE_RATE
    wave = np.zeros(n, dtype=np.float32)
    n_harm = int((SAMPLE_RATE / 2) / f0)
    harm_freqs = f0 * np.arange(1, n_harm + 1)
    amps = _formant_gain(harm_freqs, CLASS_FORMANTS[cls]) / (np.arange(1, n_harm + 1) ** 0.5)
    phases = rng.uniform(0, 2 * np.pi, size=n_harm)
    for h, (hf, a, ph) in enumerate(zip(harm_freqs, amps, phases)):
        wave += a * np.sin(2 * np.pi * hf * t + ph)
    # Raised-cosine on/off ramps avoid clicks at phoneme boundaries.
    ramp = TOTAL_STRIDE // 2
    w = np.hanning(2 * ramp)
    wave[:ramp] *= w[:ramp]
    wave[-ramp:] *= w[ramp:]
    wave *= gain / (np.max(np.abs(wave)) + 1e-8)
    return wave.astype(np.float32)


def make_clip(rng: np.random.Generator):
    """Return (waveform (L,), per-frame class labels (FRAMES_PER_CLIP,))."""
    chunks, labels = [], []
    for _ in range(PHONEMES_PER_CLIP):
        cls = int(rng.integers(NUM_CLASSES))
        f0 = float(rng.uniform(90.0, 190.0))          # acoustic: pitch
        gain = float(rng.uniform(0.6, 1.0))           # acoustic: loudness
        chunks.append(_phoneme(cls, f0, gain, rng))
        labels.append(np.full(FRAMES_PER_PHONEME, cls, dtype=np.int64))
    wave = np.concatenate(chunks)
    wave += 0.005 * rng.standard_normal(len(wave)).astype(np.float32)  # acoustic: noise
    wave /= np.max(np.abs(wave)) + 1e-8
    return wave.astype(np.float32), np.concatenate(labels)


def make_batch(batch_size: int, rng: np.random.Generator):
    """Return (waveforms (B, L), frame labels (B, FRAMES_PER_CLIP))."""
    waves, labels = [], []
    for _ in range(batch_size):
        w, y = make_clip(rng)
        waves.append(w)
        labels.append(y)
    return np.stack(waves), np.stack(labels)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wav, y = make_clip(rng)
    print(f"sample_rate={SAMPLE_RATE} Hz  token_rate={FRAME_RATE} Hz")
    print(f"clip: {wav.shape[0]} samples ({wav.shape[0]/SAMPLE_RATE:.2f}s) -> {len(y)} tokens")
    print(f"classes present: {sorted(set(y.tolist()))}")
