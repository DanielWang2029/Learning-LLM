"""Synthesize structured "speech" for wav2vec 2.0 pretraining (numpy only).

Self-supervised contrastive learning only works if the signal has temporal
structure: a masked latent frame must be *predictable* from its neighbours. We
therefore build each utterance as a sequence of held "phone" tones, each lasting
several latent frames, drawn from a small vocabulary. When we mask a few frames
in the middle of a tone, the surrounding (unmasked) frames reveal which
codeword should be there — the model can only solve the contrastive task by
learning meaningful representations.

No real audio: everything is numpy sine tones + light noise.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
VOCAB = np.geomspace(180.0, 3000.0, 8)  # 8 "phone" frequencies (Hz)


def _tone(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.4 * np.sin(2 * np.pi * 2 * freq * t)   # a formant-like harmonic
    wave += 0.2 * np.sin(2 * np.pi * 3 * freq * t)
    ramp = min(60, n // 4)
    if ramp:
        w = np.hanning(2 * ramp)
        wave[:ramp] *= w[:ramp]
        wave[-ramp:] *= w[ramp:]
    return wave.astype(np.float32)


def make_utterance(rng: np.random.Generator, num_phones: int = 12, phone_dur: float = 0.09):
    n = int(phone_dur * SAMPLE_RATE)
    phones = rng.integers(0, len(VOCAB), size=num_phones)
    wave = np.concatenate([_tone(VOCAB[p], n) for p in phones])
    wave += 0.01 * rng.standard_normal(len(wave)).astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-8
    return wave.astype(np.float32), phones


def make_batch(batch_size: int, rng: np.random.Generator, num_phones: int = 12):
    waves, phones = [], []
    for _ in range(batch_size):
        w, p = make_utterance(rng, num_phones)
        waves.append(w)
        phones.append(p)
    L = min(len(w) for w in waves)
    wav = np.stack([w[:L] for w in waves]).astype(np.float32)
    return wav, phones


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    wave, phones = make_utterance(rng)
    print(f"sample_rate={SAMPLE_RATE}  vocab={len(VOCAB)} phones")
    print(f"waveform: {wave.shape[0]} samples ({wave.shape[0]/SAMPLE_RATE:.2f}s)")
    print(f"phone sequence: {phones.tolist()}")
