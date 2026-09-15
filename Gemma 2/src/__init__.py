"""From-scratch, CPU-friendly reproduction of Gemma 2's distinctive pieces.

Exposes the model and its building blocks so demos can `from src import ...`.
"""

from .config import Gemma2Config
from .norm import RMSNorm
from .attention import GemmaAttention, build_sliding_window_mask, build_causal_mask
from .model import Gemma2Model, soft_cap

__all__ = [
    "Gemma2Config",
    "RMSNorm",
    "GemmaAttention",
    "Gemma2Model",
    "soft_cap",
    "build_sliding_window_mask",
    "build_causal_mask",
]
