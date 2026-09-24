"""From-scratch Zipformer reference implementation.

zipformer.py provides:
    BiasNorm, Bypass          -- §3.3 / §3.2 building blocks
    downsample / upsample     -- the U-Net rate changers (§3.1)
    ZipformerBlock            -- attention + conv + feed-forward with BiasNorm
    ConstantRateEncoder       -- baseline (all blocks at full rate)
    ZipformerEncoder          -- U-Net with a low-rate middle
"""

from .zipformer import (
    BiasNorm,
    Bypass,
    ConstantRateEncoder,
    SelfAttention,
    ZipformerBlock,
    ZipformerEncoder,
    downsample,
    upsample,
)

__all__ = [
    "BiasNorm",
    "Bypass",
    "ConstantRateEncoder",
    "SelfAttention",
    "ZipformerBlock",
    "ZipformerEncoder",
    "downsample",
    "upsample",
]
