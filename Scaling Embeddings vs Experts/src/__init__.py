"""Minimal reproduction of "Scaling Embeddings Outperforms Scaling Experts".

Meituan LongCat Team (2026), arXiv:2601.21204.

The paper's claim: at a fixed parameter budget, spending extra capacity on the
**embedding** (via N-gram Embedding, §2) can beat spending it on more MoE
**experts** (§3). This package implements both allocation strategies and a tiny
language model that can use either.

Public API:
- ``NGramEmbedding``  the N-gram (Over-Encoding) embedding layer (Eq. 1-3).
- ``SparseMoE``       a small top-k MoE FFN (the "expert scaling" dimension).
- ``TinyLM``          a 1-layer causal LM that plugs in either strategy.
- ``count_params``    parameter counting for the fixed-budget comparison.
"""

from .ngram_embedding import NGramEmbedding
from .moe import SparseMoE
from .model import TinyLM, count_params

__all__ = ["NGramEmbedding", "SparseMoE", "TinyLM", "count_params"]
