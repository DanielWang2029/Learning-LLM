"""A faithful, minimal PyTorch implementation of the PaLM decoder-only LM.

Based on Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways"
(2022, arXiv:2204.02311). The paper PDF lives next to this package in
``PaLM/palm.pdf``.

The implementation isolates PaLM's distinctive architecture choices (Section 2
of the paper) so each one is easy to find in code:

- `RotaryEmbedding` / `apply_rotary`  — RoPE positions (Section 2, "RoPE Embeddings")
- `SwiGLU`                            — SwiGLU MLP activation (Section 2, "SwiGLU Activation")
- `MultiQueryAttention`              — single shared K/V head (Section 2, "Multi-Query Attention")
- `PaLMBlock`                        — the PARALLEL attention+MLP formulation (Section 2, "Parallel Layers")
- `PaLM`                             — the full decoder-only language model
"""

from .rope import RotaryEmbedding, apply_rotary
from .normalization import LayerNormNoBias
from .feedforward import SwiGLU
from .attention import MultiQueryAttention
from .block import PaLMBlock
from .model import PaLM, PaLMConfig

__all__ = [
    "RotaryEmbedding",
    "apply_rotary",
    "LayerNormNoBias",
    "SwiGLU",
    "MultiQueryAttention",
    "PaLMBlock",
    "PaLM",
    "PaLMConfig",
]
