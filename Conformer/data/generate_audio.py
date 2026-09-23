"""Synthesize the toy speech task for the Conformer demo (numpy only).

We build a per-frame classification task that *needs both* local and global
modeling, so the ablation in the demo is meaningful:

* Each utterance begins with a "key" tone (one of ``K`` frequencies). The key
  is a single global fact stated once, at the start of the utterance.
* After the key comes a sequence of short "content" tones, each one of the same
  ``K`` frequencies.
* The per-frame label for a content tone is ``(tone_id + key) % K``. To label a
  frame correctly the model must (a) identify the *local* tone at that frame and
  (b) recall the *global* key from the very start of the utterance.

To make the *local* identification genuinely need temporal context, the whole
waveform is passed through a short decaying FIR filter (a mild "reverb" smear),
so a single frame is ambiguous and a few neighbouring frames disambiguate it —
exactly what the Conformer convolution module is good at.

No real audio is used: every sample comes from ``numpy`` sine tones + noise.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
HOP = 160                 # 10 ms frames, matches the front end
NUM_TONES = 6             # K distinct "phonemes"
IGNORE_INDEX = -100       # frames that carry no label (key/silence)

# K log-spaced tone frequencies (Hz), well separated on the Mel scale.
TONE_FREQS = np.geomspace(200.0, 2600.0, NUM_TONES)


def _tone(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    # A couple of harmonics + a light vibrato make it "formant-like".
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.35 * np.sin(2 * np.pi * 2 * freq * t)
    wave += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    # Raised-cosine on/off ramps to avoid clicks.
    ramp = min(80, n // 4)
    if ramp > 0:
        w = np.hanning(2 * ramp)
        wave[:ramp] *= w[:ramp]
        wave[-ramp:] *= w[ramp:]
    return wave.astype(np.float32)


def _reverb(x: np.ndarray) -> np.ndarray:
    """Short decaying FIR: smears energy across ~5 ms to create local context."""
    kernel = np.exp(-np.arange(80) / 25.0).astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def make_utterance(rng: np.random.Generator, num_content: int = 12):
    """Return (waveform, sample_labels) for one utterance."""
    key = int(rng.integers(NUM_TONES))
    key_dur = int(0.16 * SAMPLE_RATE)
    tone_dur = int(0.10 * SAMPLE_RATE)
    gap = int(0.03 * SAMPLE_RATE)

    chunks, labels = [], []

    def add(wave, label):
        chunks.append(wave)
        labels.append(np.full(len(wave), label, dtype=np.int64))

    # Key segment (global cue) — carries no label of its own.
    add(_tone(TONE_FREQS[key], key_dur, rng), IGNORE_INDEX)
    add(np.zeros(gap, np.float32), IGNORE_INDEX)

    for _ in range(num_content):
        tone_id = int(rng.integers(NUM_TONES))
        target = (tone_id + key) % NUM_TONES
        add(_tone(TONE_FREQS[tone_id], tone_dur, rng), target)
        add(np.zeros(gap, np.float32), IGNORE_INDEX)

    wave = np.concatenate(chunks)
    sample_labels = np.concatenate(labels)
    wave = _reverb(wave)
    wave += 0.01 * rng.standard_normal(len(wave)).astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-8
    return wave.astype(np.float32), sample_labels, key


def frame_labels(sample_labels: np.ndarray, num_frames: int) -> np.ndarray:
    """Down-sample per-sample labels to per-frame labels (center-aligned)."""
    idx = np.clip(np.arange(num_frames) * HOP, 0, len(sample_labels) - 1)
    return sample_labels[idx]


def make_batch(batch_size: int, rng: np.random.Generator, num_content: int = 12):
    """Return padded (waveforms, sample_labels list, keys) for a batch."""
    waves, labels, keys = [], [], []
    for _ in range(batch_size):
        w, sl, k = make_utterance(rng, num_content)
        waves.append(w)
        labels.append(sl)
        keys.append(k)
    max_len = max(len(w) for w in waves)
    wav = np.zeros((batch_size, max_len), np.float32)
    for i, w in enumerate(waves):
        wav[i, : len(w)] = w
    return wav, labels, keys


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wave, sl, key = make_utterance(rng)
    print(f"sample_rate={SAMPLE_RATE}  tones={NUM_TONES}  key={key}")
    print(f"waveform: {wave.shape[0]} samples ({wave.shape[0]/SAMPLE_RATE:.2f}s)")
    print(f"labelled frames (10ms): {(sl >= 0).sum() // HOP}")
