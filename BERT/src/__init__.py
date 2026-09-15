"""A minimal, faithful BERT-style encoder in PyTorch.

Based on Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding" (2018), the paper included in this folder.

Public API:

- ``MultiHeadSelfAttention``     bidirectional self-attention (no causal mask)
- ``EncoderLayer`` / ``Encoder`` the Transformer encoder stack
- ``BertModel``                  embeddings + encoder + masked-LM head
"""

from .attention import MultiHeadSelfAttention, scaled_dot_product_attention
from .feedforward import PositionwiseFeedForward
from .layers import Encoder, EncoderLayer
from .model import BertModel, CLS, MASK, PAD, SEP, NUM_SPECIAL

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadSelfAttention",
    "PositionwiseFeedForward",
    "Encoder",
    "EncoderLayer",
    "BertModel",
    "CLS",
    "MASK",
    "PAD",
    "SEP",
    "NUM_SPECIAL",
]
