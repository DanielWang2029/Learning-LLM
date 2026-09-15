"""A tiny, faithful reproduction of DeepSeek-V3's signature components.

Based on the "DeepSeek-V3 Technical Report" (2024, arXiv:2412.19437). The paper
PDF lives next to this package in ``DeepSeek-V3/deepseek-v3.pdf``.

Implements, at CPU scale:
- Multi-Head Latent Attention (mla.py) — low-rank compressed KV cache,
- DeepSeekMoE (moe.py)                  — fine-grained routed + shared experts,
- Multi-Token Prediction (mtp.py)       — predict the token two steps ahead,
all combined in a small decoder-only model (model.py).
"""

from .mla import MLAConfig, MultiHeadLatentAttention, StandardMHA
from .model import Block, DeepSeekV3, DeepSeekV3Config, RMSNorm
from .moe import DeepSeekMoE, MoEInfo
from .mtp import MTPModule

__all__ = [
    "MLAConfig", "MultiHeadLatentAttention", "StandardMHA",
    "Block", "DeepSeekV3", "DeepSeekV3Config", "RMSNorm",
    "DeepSeekMoE", "MoEInfo", "MTPModule",
]
