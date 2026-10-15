"""High-Entropy Sum (HES) and the baseline metrics it is compared against.

Paper reference: "Unified Data Selection for LLM Reasoning" (arXiv:2605.22389),
Section 3.1 ("High-Entropy Sum").

The core idea: a long chain-of-thought response contains a few *critical*,
high-entropy tokens (the "forks" where the model genuinely decides something).
Rather than averaging entropy over the whole response (which dilutes the signal)
or summing over all tokens (which is confounded by length), HES sums the entropy
of ONLY the top-p% highest-entropy tokens of the sample:

    HES_relative(y) = sum_{ t : rank(H_t) in top p% } H_t         (Eq. 1, p=0.5%)

A larger HES means the response "successfully navigates more numerous and intense
forks", which the paper shows correlates with higher reasoning quality.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


def token_entropies(logits: torch.Tensor) -> torch.Tensor:
    """Per-position predictive entropy H_t = -sum_v p_v log p_v (natural log).

    ``logits``: (T, vocab) next-token logits at each position of a response.
    Returns a 1-D tensor of length T with the entropy (in nats) at each token.
    """
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def _top_p_count(n: int, p: float) -> int:
    """Number of tokens in the top-p percentile of a length-n sample (>=1)."""
    return max(1, math.ceil(n * p))


def hes(entropies: torch.Tensor, p: float = 0.005) -> float:
    """HES_relative: sum of the entropies of the top-p% highest-entropy tokens.

    This is the paper's recommended metric (relative / percentile threshold),
    robust to differences in sample length because the threshold is per-sample.
    """
    if entropies.numel() == 0:
        return 0.0
    k = _top_p_count(entropies.numel(), p)
    top = torch.topk(entropies, k).values
    return float(top.sum().item())


def hes_absolute(entropies: torch.Tensor, tau: float) -> float:
    """HES_absolute: sum of entropies of tokens whose entropy exceeds tau.

    The paper's simpler variant with a fixed threshold; it underperforms the
    relative version because a fixed cutoff cannot adapt across samples/models.
    """
    mask = entropies >= tau
    return float(entropies[mask].sum().item())


# ---- Baseline metrics the paper compares against (Section 4, Table 4) ----

def avg_entropy(entropies: torch.Tensor) -> float:
    """AvgE: mean entropy over ALL tokens (dilated by many low-entropy tokens)."""
    return float(entropies.mean().item()) if entropies.numel() else 0.0


def avg_high_entropy(entropies: torch.Tensor, p: float = 0.005) -> float:
    """AvgHE: mean entropy over the top-p% tokens (magnitude, not count)."""
    if entropies.numel() == 0:
        return 0.0
    k = _top_p_count(entropies.numel(), p)
    return float(torch.topk(entropies, k).values.mean().item())


def entropy_sum(entropies: torch.Tensor) -> float:
    """ES: sum of entropy over ALL tokens (confounded by sample length)."""
    return float(entropies.sum().item())
