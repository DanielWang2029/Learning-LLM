"""A faithful, minimal PyTorch reproduction of the Llama 3 architecture + tokenizer.

Based on "The Llama 3 Herd of Models" (Meta, 2024, arXiv:2407.21783). The paper
PDF lives next to this package in ``Llama 3/llama_3.pdf``.

Llama 3 is a dense decoder-only Transformer built from four canonical ingredients
(RMSNorm, RoPE with base 500k, Grouped-Query Attention, SwiGLU) paired with a
much larger 128K byte-level BPE tokenizer. All are implemented here at tiny scale.
"""

from .attention import GroupedQueryAttention, repeat_kv
from .block import TransformerBlock
from .feedforward import SwiGLU, swiglu_hidden_dim
from .model import Llama3, Llama3Config
from .normalization import RMSNorm
from .rope import RotaryEmbedding, apply_rotary
from .tokenizer import ByteBPETokenizer

__all__ = [
    "GroupedQueryAttention",
    "repeat_kv",
    "TransformerBlock",
    "SwiGLU",
    "swiglu_hidden_dim",
    "Llama3",
    "Llama3Config",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "ByteBPETokenizer",
]
