"""A minimal decoder-only GPT language model in PyTorch.

Based on Radford et al., "Improving Language Understanding by Generative
Pre-Training" (GPT-1, 2018), the paper included in this folder.

Public API:

- ``CausalSelfAttention``  masked multi-head self-attention
- ``GPT`` / ``GPTConfig``  the decoder-only language model
- ``GPTClassifier``        fine-tuning head for downstream classification
"""

from .attention import CausalSelfAttention
from .model import GPT, GPTConfig, GPTClassifier, Block

__all__ = ["CausalSelfAttention", "GPT", "GPTConfig", "GPTClassifier", "Block"]
