"""Marginalize over sampled reasoning paths by majority vote.

This is the heart of the paper (Wang et al., 2022, §2): instead of trusting a
single greedily-decoded chain of thought, sample a diverse set of reasoning
paths and take the answer that the largest number of them agree on. Different
(possibly flawed) chains that reach the correct answer reinforce each other,
while idiosyncratic mistakes get out-voted.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional


def majority_vote(answers: Iterable[Optional[str]]) -> Optional[str]:
    """Return the most common non-None answer, or None if there are none.

    Ties are broken deterministically by taking the answer that appeared first
    (Counter.most_common preserves insertion order for equal counts in CPython).
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    for a in valid:  # first-seen wins ties, for determinism
        if counts[a] == best:
            return a
    return None
