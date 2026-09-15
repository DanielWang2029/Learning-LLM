"""A minimal GPT-2 decoder-only language model in PyTorch.

Based on Radford et al., "Language Models are Unsupervised Multitask Learners"
(GPT-2, 2019), the paper included in this folder.

Public API:

- ``CausalSelfAttention``  masked multi-head self-attention
- ``GPT`` / ``GPTConfig``  the decoder-only language model
"""

from .attention import CausalSelfAttention
from .model import GPT, GPTConfig, Block

__all__ = ["CausalSelfAttention", "GPT", "GPTConfig", "Block"]
