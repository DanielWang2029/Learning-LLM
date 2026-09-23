"""From-scratch cache-aware streaming Conformer.

streaming_conformer.py provides:
    build_chunk_mask            -- chunk-aware attention mask (§3.1)
    StreamingAttention          -- self-attention with a KV cache (§3.3)
    CausalConvModule            -- causal depthwise conv with a conv-state cache
    StreamingConformerBlock     -- one block, full + streaming forward
    StreamingConformerEncoder   -- forward() vs forward_streaming() (must match)
"""

from .streaming_conformer import (
    CausalConvModule,
    StreamingAttention,
    StreamingConformerBlock,
    StreamingConformerEncoder,
    build_chunk_mask,
)

__all__ = [
    "CausalConvModule",
    "StreamingAttention",
    "StreamingConformerBlock",
    "StreamingConformerEncoder",
    "build_chunk_mask",
]
