"""Generate trajectory segments and preference labels.

Following Christiano et al. (2017): a policy (here a random walk) produces short
trajectory *segments*. A human is shown two segments and says which they prefer.
We simulate that human with the HIDDEN true reward — the segment that collects
more true reward is labeled preferred. The reward model never sees these true
rewards, only the binary preference labels.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .gridworld import ACTIONS, GridWorld


def random_segment(
    env: GridWorld, length: int, rng: np.random.Generator
) -> List[int]:
    """A length-`length` random walk (list of visited states, ignoring terminals)."""
    s = int(rng.integers(env.num_states))
    states = [s]
    for _ in range(length):
        a = int(rng.integers(len(ACTIONS)))
        s = env.step(s, a)
        states.append(s)
    return states


def segment_true_return(env: GridWorld, seg: List[int]) -> float:
    """Sum of the hidden true reward over the states in a segment."""
    return float(sum(env.true_reward[s] for s in seg))


def make_preference_dataset(
    env: GridWorld, n_pairs: int, seg_len: int, rng: np.random.Generator
) -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
    """Return (seg_a, seg_b, label) where label=1 means seg_a is preferred.

    Pairs whose two segments tie in true return are discarded so every label
    carries signal (mirrors the paper dropping indistinguishable comparisons).
    """
    seg_a, seg_b, labels = [], [], []
    while len(labels) < n_pairs:
        a = random_segment(env, seg_len, rng)
        b = random_segment(env, seg_len, rng)
        ra, rb = segment_true_return(env, a), segment_true_return(env, b)
        if abs(ra - rb) < 1e-9:
            continue
        seg_a.append(a)
        seg_b.append(b)
        labels.append(1.0 if ra > rb else 0.0)
    return seg_a, seg_b, np.array(labels, dtype=np.float32)
