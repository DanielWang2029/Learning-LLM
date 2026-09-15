"""A tiny reproduction of Gemini 1.5's two headline documented ideas.

Based on "Gemini 1.5: Unlocking multimodal understanding across millions of tokens
of context" (Google, 2024, arXiv:2403.05530). The paper PDF lives next to this
package in ``Gemini 1.5/gemini_15.pdf``.

Gemini 1.5 is a closed model, so this reproduces — honestly, at tiny scale — the
two ideas it documents most concretely:

1. a **sparse Mixture-of-Experts** Transformer (moe.py, model.py), and
2. **long-context near-perfect recall**, measured with a needle-in-a-haystack
   associative-recall test (task.py).
"""

from .model import Attention, Block, GeminiConfig, GeminiMoE, RMSNorm
from .moe import Expert, RouteInfo, SparseMoE
from .task import (NEEDLE, N_TOKENS, QUERY, TOKEN_BASE, VOCAB_SIZE, make_batch,
                   make_eval_batch)

__all__ = [
    "Attention", "Block", "GeminiConfig", "GeminiMoE", "RMSNorm",
    "Expert", "RouteInfo", "SparseMoE",
    "NEEDLE", "N_TOKENS", "QUERY", "TOKEN_BASE", "VOCAB_SIZE",
    "make_batch", "make_eval_batch",
]
