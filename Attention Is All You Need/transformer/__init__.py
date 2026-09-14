"""A faithful, minimal PyTorch implementation of the Transformer.

Based on Vaswani et al., "Attention Is All You Need" (2017), the paper
included in this repository under `Attention Is All You Need/`.

The public API mirrors the structure of the paper:

- `MultiHeadAttention`        (Section 3.2.2)
- `PositionalEncoding`        (Section 3.5)
- `PositionwiseFeedForward`   (Section 3.3)
- `EncoderLayer` / `DecoderLayer` and their stacks (Section 3.1)
- `Transformer`              full encoder-decoder model
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .positional import PositionalEncoding
from .feedforward import PositionwiseFeedForward
from .layers import EncoderLayer, DecoderLayer, Encoder, Decoder
from .model import Transformer

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "Encoder",
    "Decoder",
    "Transformer",
]
