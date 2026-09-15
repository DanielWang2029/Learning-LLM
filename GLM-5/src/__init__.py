"""Minimal, from-scratch reproduction of GLM-5's core efficiency mechanism.

GLM-5 (Zhipu AI, 2026) adopts **DeepSeek Sparse Attention (DSA)** to cut the
O(L^2) cost of long-context attention. Instead of attending densely to every
key, a cheap *lightning indexer* scores how important each key is for each
query, and only the **top-k** keys are attended to (paper Section 2.1.1).

Public API mirrors that mechanism:

- ``LightningIndexer``     the low-cost per-(query,key) importance scorer
- ``full_attention``       the dense O(L^2) reference
- ``dsa_attention``        top-k sparse attention (scores only k << L keys)
- ``attention_mass_recall``  how much true attention mass the top-k captures
"""

from .dsa import (
    LightningIndexer,
    full_attention,
    dsa_attention,
    attention_mass_recall,
)

__all__ = [
    "LightningIndexer",
    "full_attention",
    "dsa_attention",
    "attention_mass_recall",
]
