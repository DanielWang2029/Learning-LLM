"""Synthetic streaming-ASR data with Delayed Streams Modeling (DSM) targets.

A toy "speech" stream: a sequence of spoken WORDS, each spanning L audio frames,
separated by G silence frames. Crucially, a word's identity is AMBIGUOUS in its
first frames and becomes clear only later (the per-frame noise decays across the
word). So emitting a word early (small delay) must happen with little acoustic
evidence -> lower accuracy; waiting longer (larger delay tau) sees the clean
later frames -> higher accuracy. This reproduces Voxtral Realtime's latency <->
accuracy trade-off.

Target stream (one token per 80 ms frame, DSM, Section 3.1):
    [P] (pad / "wait") everywhere, except for each word, at its emission frame
    a + tau we place [W] (word-boundary), then the word token at a + tau + 1.
tau is the configurable streaming delay in frames (multiples of 80 ms).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

PAD, WB = 0, 1          # [P] padding/wait, [W] word-boundary
NUM_WORDS = 6
FIRST_WORD = 2          # word token ids: 2 .. 2+NUM_WORDS-1
VOCAB_SIZE = FIRST_WORD + NUM_WORDS
FEAT_DIM = 12

L = 7                   # frames per word (~560 ms at 80 ms/frame)
G = 8                   # silence frames between words
PERIOD = L + G
N_WORDS_PER_UTT = 5
UTT_FRAMES = N_WORDS_PER_UTT * PERIOD
STD_HI, STD_LO = 1.4, 0.12   # per-frame noise: high at word onset -> low at end
FRAME_MS = 80


def word_prototypes(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((NUM_WORDS, FEAT_DIM)).astype(np.float32)
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


_PROTO = word_prototypes()


def _frame_std(j: int) -> float:
    """Noise std at within-word frame j (0=onset, L-1=end): decays over the word."""
    return STD_LO + STD_HI * (1.0 - j / (L - 1))


def make_utterance(rng: np.random.Generator, tau: int) -> Dict:
    feats = np.zeros((UTT_FRAMES, FEAT_DIM), dtype=np.float32)
    target = np.full(UTT_FRAMES, PAD, dtype=np.int64)
    words, tok_frames = [], []
    for i in range(N_WORDS_PER_UTT):
        w = int(rng.integers(NUM_WORDS))
        a = i * PERIOD
        for j in range(L):                                   # word frames (decaying noise)
            feats[a + j] = _PROTO[w] + _frame_std(j) * rng.standard_normal(FEAT_DIM).astype(np.float32)
        for j in range(L, PERIOD):                           # silence frames
            feats[a + j] = 0.05 * rng.standard_normal(FEAT_DIM).astype(np.float32)
        e = a + tau                                          # emission frame
        if e + 1 < UTT_FRAMES:
            target[e] = WB
            target[e + 1] = FIRST_WORD + w
            words.append(w)
            tok_frames.append(e + 1)
    return {"feats": feats, "target": target, "tau": tau,
            "words": words, "tok_frames": tok_frames}


def make_batch(bs: int, rng: np.random.Generator, tau_choices: List[int]) -> Tuple:
    us = [make_utterance(rng, int(rng.choice(tau_choices))) for _ in range(bs)]
    feats = np.stack([u["feats"] for u in us])
    target = np.stack([u["target"] for u in us])
    taus = np.array([u["tau"] for u in us], dtype=np.int64)
    return feats, target, taus, us


def tau_frames_to_ms(tau: int) -> int:
    return tau * FRAME_MS
