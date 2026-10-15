"""Synthetic audio-conditioned task for the M2-OPD foundation recipe (§3).

Qwen-Audio-3.1-Realtime's foundation training distils a strong *Text Teacher*
(which reads clean text x_t) into an *Audio Student* (which reads audio x_a),
on the student's OWN generated trajectories (on-policy), while a frozen *Audio
Reference* preserves behaviour that text cannot supervise (emotion / acoustic
scene / pragmatics -- "information beyond the transcript", §3.2).

This toy reproduces exactly that structure with two skills in one token stream:

  output = [ STYLE , s_1 , s_2 , ... , s_L ]

  * position 0 -- STYLE: an audio-only attribute (think "emotion/acoustic
    scene"). It is injected into the audio features but is NOT present in the
    text x_t, so the Text Teacher cannot supervise it; only an audio signal can.
  * positions 1..L -- a running STATE MACHINE  s_j = T[s_{j-1}, c_j]  over the
    spoken content tokens c (a random transition table T, s_0 = base). This is
    text-representable, so the Text Teacher scores it well -- but ONLY from the
    student's current state, which is what makes on-policy supervision matter
    (off-policy SFT never sees the states the student actually drifts into).

Everything is numpy; no audio files, no downloads.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

V = 8                      # number of content / state values (0..V-1)
L = 12                     # spoken content length
N_STYLES = 2               # audio-only attribute classes
BASE = V                   # the "no previous state" marker (s_0)
STYLE0 = V                 # output-token id for style class 0
# unified output vocabulary: {0..V-1} state tokens, {STYLE0, STYLE0+1} styles
OUT_VOCAB = V + N_STYLES

FEAT_DIM = 16
STYLE_GAIN = 1.6           # how strongly the style tints the audio
NOISE_STD = 0.3            # mild acoustic noise -> content stays readable


# The gold transition maps into a small set of "attractor" states, so correct
# (gold) trajectories only ever visit a thin slice of the state space. The other
# states are reachable only when the policy makes a mistake -- precisely where
# off-policy SFT (gold states only) is blind and on-policy OPD can recover. This
# is what makes exposure bias, and the on-policy advantage, visible.
ATTRACTORS = (2, 4, 7)     # |gold states| << V


def _rng_tables(seed: int = 0):
    rng = np.random.default_rng(seed)
    attr = np.array(ATTRACTORS, dtype=np.int64)
    # random transition table whose OUTPUTS lie in the attractor set
    T = attr[rng.integers(0, len(attr), size=(V, V))].astype(np.int64)
    A = rng.standard_normal((V, FEAT_DIM)).astype(np.float32)      # content -> audio
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    S = rng.standard_normal((N_STYLES, FEAT_DIM)).astype(np.float32)  # style tint
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-8
    return T, A, S


T, A, S = _rng_tables(0)


def gold_states(content: np.ndarray) -> np.ndarray:
    """content (L,) ints -> gold next-states s_1..s_L via s_j = T[s_{j-1}, c_j]."""
    out = np.empty(L, dtype=np.int64)
    s = 0  # base state
    for j in range(L):
        s = int(T[s, content[j]])
        out[j] = s
    return out


def make_utterance(rng: np.random.Generator) -> Dict:
    content = rng.integers(0, V, size=L).astype(np.int64)
    style = int(rng.integers(0, N_STYLES))
    feats = A[content] + STYLE_GAIN * S[style] + NOISE_STD * rng.standard_normal(
        (L, FEAT_DIM)
    ).astype(np.float32)
    sums = gold_states(content)
    target = np.concatenate([[STYLE0 + style], sums]).astype(np.int64)  # (L+1,)
    return {
        "feats": feats.astype(np.float32),
        "content": content,          # x_t: the clean text the Text Teacher reads
        "style": style,
        "target": target,            # [STYLE, s_1..s_L]
        "style_token": STYLE0 + style,
    }


def make_batch(bs: int, rng: np.random.Generator) -> Dict:
    us = [make_utterance(rng) for _ in range(bs)]
    return {
        "feats": np.stack([u["feats"] for u in us]),
        "content": np.stack([u["content"] for u in us]),
        "style": np.array([u["style"] for u in us], dtype=np.int64),
        "target": np.stack([u["target"] for u in us]),
        "style_token": np.array([u["style_token"] for u in us], dtype=np.int64),
    }
