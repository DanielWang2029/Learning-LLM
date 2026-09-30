"""A minimal, from-scratch reproduction of Mimi's documented mechanism.

Mimi is the streaming neural audio codec introduced in the Moshi tech report
(Défossez et al., Kyutai, 2024; arXiv:2410.00037), included here as ``mimi.pdf``.

Public API mirrors the paper (Moshi §3.3):

- ``CausalConv1d``      §3.3.1  causal, streamable 1-D convolution
- ``MimiEncoder``       §3.3.1  SEANet-style causal down-sampling encoder
- ``MimiDecoder``       §3.3.1  causal up-sampling decoder
- ``VectorQuantizerEMA``§3.3.2  a single EMA vector-quantizer codebook
- ``ResidualVQ``        §3.3.2  residual VQ stack (scalable bitrate)
- ``SplitRVQ``          §3.3.2  split RVQ: semantic VQ ∥ acoustic RVQ
- ``MimiCodec``         §3.3    the full encoder -> split-RVQ -> decoder codec
"""

from .codec import (
    CausalConv1d,
    ResidualUnit,
    EncoderBlock,
    DecoderBlock,
    MimiEncoder,
    MimiDecoder,
)
from .quantization import VectorQuantizerEMA, ResidualVQ, SplitRVQ
from .model import MimiCodec
from .features import stft_mag, multi_scale_spectral_loss

__all__ = [
    "CausalConv1d",
    "ResidualUnit",
    "EncoderBlock",
    "DecoderBlock",
    "MimiEncoder",
    "MimiDecoder",
    "VectorQuantizerEMA",
    "ResidualVQ",
    "SplitRVQ",
    "MimiCodec",
    "stft_mag",
    "multi_scale_spectral_loss",
]
