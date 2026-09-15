"""A toy response generator that (deliberately) sometimes violates the rules.

Stands in for the initial "helpful-only" model of Constitutional AI, which can
produce harmful or low-quality responses. Each sampled response may inject a
toxic word, an adjacent repetition, and/or be over-length — exactly the failure
modes the constitution targets.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .words import MAX_LEN, SAFE, TOXIC

# A little flavor: the (adversarial) prompts the assistant is responding to.
PROMPTS = [
    "tell me what you think",
    "answer honestly",
    "give me your opinion",
    "respond to my message",
]


def sample_response(rng: np.random.Generator) -> List[str]:
    """Sample a response that may violate one or more constitutional rules."""
    length = int(rng.integers(3, MAX_LEN + 3))  # can exceed MAX_LEN
    resp = [SAFE[int(rng.integers(len(SAFE)))] for _ in range(length)]

    if rng.random() < 0.45:  # inject harmful language
        pos = int(rng.integers(len(resp)))
        resp[pos] = TOXIC[int(rng.integers(len(TOXIC)))]
    if rng.random() < 0.35:  # inject an adjacent repetition
        pos = int(rng.integers(max(1, len(resp) - 1)))
        resp[pos + 1 if pos + 1 < len(resp) else pos] = resp[pos]
    return resp


def sample_dataset(n: int, rng: np.random.Generator) -> List[List[str]]:
    return [sample_response(rng) for _ in range(n)]
