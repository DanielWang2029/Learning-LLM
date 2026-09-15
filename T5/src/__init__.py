"""A minimal T5-style text-to-text encoder–decoder Transformer.

Based on Raffel et al., "Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer" (T5, 2020), the paper included in this
folder.

Public API:

- ``MultiHeadAttention``          self / cross attention
- ``Encoder`` / ``Decoder``       the two Transformer stacks
- ``T5``                          the full encoder–decoder model
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .layers import Decoder, DecoderLayer, Encoder, EncoderLayer
from .model import T5, PAD, BOS, EOS, subsequent_mask

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "T5",
    "PAD",
    "BOS",
    "EOS",
    "subsequent_mask",
]
