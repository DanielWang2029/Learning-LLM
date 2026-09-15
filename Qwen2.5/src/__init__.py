"""A faithful, minimal PyTorch reproduction of the Qwen2.5 architecture.

Based on the "Qwen2.5 Technical Report" (Alibaba, 2024, arXiv:2412.15115). The
paper PDF lives next to this package in ``Qwen2.5/qwen25.pdf``.

Qwen2.5 is a dense decoder-only Transformer whose distinctive choices are QKV
bias (attention.py) and untied input/output embeddings (model.py), on top of the
standard RMSNorm + RoPE + GQA + SwiGLU recipe (layers.py, attention.py).
"""

from .attention import QwenAttention, repeat_kv
from .layers import RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary
from .model import Qwen25, Qwen25Config, QwenBlock

__all__ = [
    "QwenAttention", "repeat_kv", "RMSNorm", "RotaryEmbedding", "SwiGLU",
    "apply_rotary", "Qwen25", "Qwen25Config", "QwenBlock",
]
