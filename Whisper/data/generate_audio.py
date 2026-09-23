"""Synthesize an audio->text transcription task for Whisper (numpy only).

Each utterance is a spoken "sentence": a sequence of tone "words" drawn from a
vocabulary of ``NUM_TONES`` distinct frequencies. The ground-truth transcript is
simply the sequence of tone ids. The encoder-decoder must listen to the audio
and emit the correct token sequence — a miniature speech-recognition task.

No real audio: every sample is a numpy sine-tone stack (with harmonics) + noise.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
NUM_TONES = 10                                   # vocabulary of "words"
TONE_FREQS = np.geomspace(200.0, 3200.0, NUM_TONES)


def _tone(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.35 * np.sin(2 * np.pi * 2 * freq * t)   # formant-like harmonics
    wave += 0.18 * np.sin(2 * np.pi * 3 * freq * t)
    ramp = min(120, n // 4)
    if ramp:
        w = np.hanning(2 * ramp)
        wave[:ramp] *= w[:ramp]
        wave[-ramp:] *= w[ramp:]
    return wave.astype(np.float32)


def make_utterance(rng: np.random.Generator, min_len: int = 5, max_len: int = 8):
    length = int(rng.integers(min_len, max_len + 1))
    ids = rng.integers(0, NUM_TONES, size=length)
    tone_dur = int(0.12 * SAMPLE_RATE)
    gap = int(0.03 * SAMPLE_RATE)
    chunks = []
    for i in ids:
        chunks.append(_tone(TONE_FREQS[i], tone_dur))
        chunks.append(np.zeros(gap, np.float32))
    wave = np.concatenate(chunks)
    wave += 0.01 * rng.standard_normal(len(wave)).astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-8
    return wave.astype(np.float32), ids.tolist()


def make_batch(batch_size: int, rng: np.random.Generator):
    waves, id_lists = [], []
    for _ in range(batch_size):
        w, ids = make_utterance(rng)
        waves.append(w)
        id_lists.append(ids)
    max_samp = max(len(w) for w in waves)
    wav = np.zeros((batch_size, max_samp), np.float32)
    for i, w in enumerate(waves):
        wav[i, : len(w)] = w
    return wav, id_lists


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wave, ids = make_utterance(rng)
    print(f"sample_rate={SAMPLE_RATE}  vocab={NUM_TONES} tones")
    print(f"waveform: {wave.shape[0]} samples ({wave.shape[0]/SAMPLE_RATE:.2f}s)")
    print(f"transcript (tone ids): {ids}")
