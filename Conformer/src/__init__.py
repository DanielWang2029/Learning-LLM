"""A minimal, from-scratch Conformer encoder.

Based on Gulati et al., "Conformer: Convolution-augmented Transformer for
Speech Recognition" (2020), the paper included in this folder as
``conformer.pdf``.

Public API mirrors the paper (Section 2):

- ``FeedForwardModule``            (Section 2.2, Fig. 4 — half-step FFN)
- ``MultiHeadSelfAttentionModule`` (Section 2.2, Fig. 3)
- ``ConvolutionModule``           (Section 2.2, Fig. 2)
- ``ConformerBlock``              (Section 2.2, Fig. 1 — the "sandwich")
- ``ConformerEncoder``            a stack of blocks over log-mel features
- ``log_mel_spectrogram``         hand-rolled log-Mel front end (numpy/torch)
"""

from .features import log_mel_spectrogram, MelFrontend
from .conformer import (
    Swish,
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ConvolutionModule,
    ConformerBlock,
    ConformerEncoder,
    FrameClassifier,
)

__all__ = [
    "log_mel_spectrogram",
    "MelFrontend",
    "Swish",
    "FeedForwardModule",
    "MultiHeadSelfAttentionModule",
    "ConvolutionModule",
    "ConformerBlock",
    "ConformerEncoder",
    "FrameClassifier",
]
