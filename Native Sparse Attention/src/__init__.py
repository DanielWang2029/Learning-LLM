"""Native Sparse Attention (NSA, Yuan et al. 2025) — minimal reproduction.

Exposes the three-branch NSA attention module (compression + selection +
sliding window with a learned gate), a full-attention baseline, and a tiny
decoder-only model that can use either.
"""

from .nsa import NSAAttention, FullAttention
from .model import RecallModel

__all__ = ["NSAAttention", "FullAttention", "RecallModel"]
