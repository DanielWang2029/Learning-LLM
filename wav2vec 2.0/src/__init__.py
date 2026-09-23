"""A minimal, from-scratch wav2vec 2.0 for self-supervised speech pretraining.

Based on Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning
of Speech Representations" (2020), included here as ``wav2vec_20.pdf``.

Public API mirrors the paper (Section 2):

- ``FeatureEncoder``          (Section 2.1) CNN over the raw waveform -> latents
- ``GumbelVectorQuantizer``   (Section 2.2) product quantization of the latents
- ``ContextTransformer``      (Section 2.1) Transformer context network
- ``Wav2Vec2``               ties them together with masking + the contrastive
                              (InfoNCE) objective of Section 3.2
"""

from .model import (
    FeatureEncoder,
    GumbelVectorQuantizer,
    ContextTransformer,
    Wav2Vec2,
    compute_mask_indices,
)

__all__ = [
    "FeatureEncoder",
    "GumbelVectorQuantizer",
    "ContextTransformer",
    "Wav2Vec2",
    "compute_mask_indices",
]
