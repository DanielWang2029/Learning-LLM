"""A faithful, minimal PyTorch implementation of the Llama 2 decoder-only LM.

Based on Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models"
(2023, arXiv:2307.09288). The paper PDF lives next to this package in
``Llama 2/llama_2.pdf``.

Llama 2 keeps the LLaMA recipe (RMSNorm pre-norm, RoPE, SwiGLU) and adds one
key architecture change for the larger models (Llama 2 paper, Section 2.2):

- `GroupedQueryAttention` — **Grouped-Query Attention (GQA)**: the query heads
  are split into groups, and each group shares a single key/value head. GQA
  interpolates between standard multi-head attention (one K/V head per query
  head) and multi-query attention (one K/V head total), giving most of MQA's
  KV-cache savings with quality close to full multi-head attention.

The same attention module covers all three regimes via ``n_kv_heads``:
    n_kv_heads == n_heads  -> Multi-Head Attention (MHA)
    1 < n_kv_heads < n_heads -> Grouped-Query Attention (GQA)
    n_kv_heads == 1        -> Multi-Query Attention (MQA)
"""

from .rope import RotaryEmbedding, apply_rotary
from .normalization import RMSNorm
from .feedforward import SwiGLU, swiglu_hidden_dim
from .attention import GroupedQueryAttention, kv_cache_bytes
from .block import TransformerBlock
from .model import Llama2, Llama2Config

__all__ = [
    "RotaryEmbedding",
    "apply_rotary",
    "RMSNorm",
    "SwiGLU",
    "swiglu_hidden_dim",
    "GroupedQueryAttention",
    "kv_cache_bytes",
    "TransformerBlock",
    "Llama2",
    "Llama2Config",
]
