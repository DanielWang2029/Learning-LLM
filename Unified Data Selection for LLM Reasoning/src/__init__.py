"""A faithful, minimal reproduction of HES data selection for LLM reasoning.

Based on "Unified Data Selection for LLM Reasoning" (Li et al., 2026,
arXiv:2605.22389), included here as the PDF in this folder.

Public API mirrors the paper (Section 3):

- `token_entropies`         per-token predictive entropy under a reference LM
- `hes`, `hes_absolute`     High-Entropy Sum (sum of the top-p% token entropies)
- `avg_entropy`, `avg_high_entropy`, `entropy_sum`   the baseline metrics the
                            paper compares against (AvgE, AvgHE, ES)
- `TinyLM`                  a tiny causal Transformer used as the reference LM
                            and as the SFT student
- data generation for a synthetic long-CoT arithmetic task with quality tiers
"""

from .hes import (
    token_entropies,
    hes,
    hes_absolute,
    avg_entropy,
    avg_high_entropy,
    entropy_sum,
)
from .model import TinyLM
from . import data

__all__ = [
    "token_entropies",
    "hes",
    "hes_absolute",
    "avg_entropy",
    "avg_high_entropy",
    "entropy_sum",
    "TinyLM",
    "data",
]
