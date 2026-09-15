"""A faithful, minimal PyTorch implementation of the LLaMA decoder-only LM.

Based on Touvron et al., "LLaMA: Open and Efficient Foundation Language Models"
(2023, arXiv:2302.13971). The paper PDF lives next to this package in
``LLaMA/llama.pdf``.

LLaMA is a standard decoder-only Transformer with three specific, now-canonical
modifications (paper Section 2.2, "Architecture"):

- `RMSNorm`                           — pre-normalization with RMSNorm (from GPT-3 + Zhang & Sennrich)
- `RotaryEmbedding` / `apply_rotary`  — RoPE rotary positions (from Su et al.)
- `SwiGLU`                            — SwiGLU feed-forward activation (from Shazeer / PaLM)

Attention here is standard multi-head attention. The ``build_llama`` /
``LLaMAConfig`` helpers also expose toggles so the demo can *ablate* each
component (swap RMSNorm→LayerNorm, RoPE→none, SwiGLU→ReLU) and show it matters.
"""

from .rope import RotaryEmbedding, apply_rotary
from .normalization import RMSNorm
from .feedforward import SwiGLU, ReLUFFN
from .attention import Attention
from .block import TransformerBlock
from .model import LLaMA, LLaMAConfig

__all__ = [
    "RotaryEmbedding",
    "apply_rotary",
    "RMSNorm",
    "SwiGLU",
    "ReLUFFN",
    "Attention",
    "TransformerBlock",
    "LLaMA",
    "LLaMAConfig",
]
