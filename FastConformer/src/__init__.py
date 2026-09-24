"""From-scratch FastConformer reference implementation.

Modules:
    subsampling.py  -- 4x regular vs 8x depthwise-separable sub-sampling (§2.1)
    conformer.py    -- a minimal Conformer block + configurable encoder
"""

from .conformer import (
    ConformerBlock,
    ConvModule,
    FastConformerEncoder,
    FeedForward,
    MultiHeadSelfAttention,
)
from .subsampling import (
    ConvSubsampling4x,
    DepthwiseSeparableSubsampling8x,
)

__all__ = [
    "ConformerBlock",
    "ConvModule",
    "FastConformerEncoder",
    "FeedForward",
    "MultiHeadSelfAttention",
    "ConvSubsampling4x",
    "DepthwiseSeparableSubsampling8x",
]
