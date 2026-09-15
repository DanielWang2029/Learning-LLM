"""From-scratch Flamingo: a visual language model with gated cross-attention.

Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022),
arXiv:2204.14198.
"""

from .gated_xattn import GatedCrossAttentionBlock
from .lm import DecoderLM
from .model import Flamingo
from .resampler import PerceiverResampler
from .vision import VisionEncoder

__all__ = [
    "Flamingo",
    "DecoderLM",
    "VisionEncoder",
    "PerceiverResampler",
    "GatedCrossAttentionBlock",
]
