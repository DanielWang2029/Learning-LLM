"""Gemma 3 — 5:1 local:global attention + QK-norm (Google, 2025).

Minimal from-scratch reproduction of arXiv:2503.19786's efficiency idea: mostly
sliding-window (local) attention with a 1-in-6 global layer, plus QK-norm. A tiny
transformer is trained on a long-context retrieval task to show the 5:1 pattern
matches all-global accuracy at a fraction of the attention memory.
"""

from .attention import Attention, attention_mask, mask_entries
from .model import TinyTransformer, Block, layer_pattern
from .task import (make_batch, describe_token, VOCAB_SIZE, N_VALUE, MARK, QUERY,
                   PAD)

__all__ = ["Attention", "attention_mask", "mask_entries", "TinyTransformer",
           "Block", "layer_pattern", "make_batch", "describe_token",
           "VOCAB_SIZE", "N_VALUE", "MARK", "QUERY", "PAD"]
