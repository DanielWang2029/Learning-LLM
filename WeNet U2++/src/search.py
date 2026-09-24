"""CTC decoding: greedy (first-pass) and prefix beam search (for n-best).

The CTC vocabulary is: content tokens 0..3 (A, B, C, D) and BLANK = 4.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np

BLANK = 4


def greedy_decode(log_probs: np.ndarray) -> List[int]:
    """Frame-synchronous argmax, collapse repeats, drop blanks. (T, V) -> tokens."""
    best = log_probs.argmax(axis=-1)
    out, prev = [], -1
    for b in best:
        if b != prev and b != BLANK:
            out.append(int(b))
        prev = int(b)
    return out


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def prefix_beam_search(log_probs: np.ndarray, beam_size: int = 8,
                       n_best: int = 4) -> List[Tuple[Tuple[int, ...], float]]:
    """Standard CTC prefix beam search. Returns [(tokens, log_prob), ...].

    Each beam entry tracks two probabilities: ending in blank (pb) and ending in
    a non-blank symbol (pnb). This is the classic algorithm (Hannun et al.).
    """
    T, V = log_probs.shape
    beams = {(): (0.0, -math.inf)}  # prefix -> (p_blank, p_non_blank)

    for t in range(T):
        next_beams = defaultdict(lambda: (-math.inf, -math.inf))
        # Consider only the most promising symbols this frame for speed.
        top = np.argsort(log_probs[t])[::-1][: beam_size]
        for prefix, (pb, pnb) in beams.items():
            prev_total = _logsumexp(pb, pnb)
            for s in top:
                p = float(log_probs[t, s])
                if s == BLANK:
                    nb, nnb = next_beams[prefix]
                    nb = _logsumexp(nb, prev_total + p)
                    next_beams[prefix] = (nb, nnb)
                else:
                    last = prefix[-1] if prefix else None
                    if s == last:
                        # repeat: extend from blank-ended prob, or stay via pnb
                        nb, nnb = next_beams[prefix]
                        nnb = _logsumexp(nnb, pnb + p)
                        next_beams[prefix] = (nb, nnb)
                        new = prefix + (int(s),)
                        xb, xnb = next_beams[new]
                        xnb = _logsumexp(xnb, pb + p)
                        next_beams[new] = (xb, xnb)
                    else:
                        new = prefix + (int(s),)
                        xb, xnb = next_beams[new]
                        xnb = _logsumexp(xnb, prev_total + p)
                        next_beams[new] = (xb, xnb)
        # prune to beam_size by total probability
        scored = sorted(next_beams.items(),
                        key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(scored[: beam_size])

    final = [(prefix, _logsumexp(pb, pnb)) for prefix, (pb, pnb) in beams.items()]
    final.sort(key=lambda kv: kv[1], reverse=True)
    return final[: n_best]


def edit_distance(a: List[int], b: List[int]) -> int:
    """Levenshtein distance between two token lists."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[m]
