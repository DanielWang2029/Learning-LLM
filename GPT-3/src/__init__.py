"""A minimal GPT-3-style decoder-only language model in PyTorch.

Based on Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020),
the paper included in this folder.

Public API:

- ``CausalSelfAttention``  masked multi-head self-attention
- ``GPT`` / ``GPTConfig``  the decoder-only language model
"""

from .attention import CausalSelfAttention
from .model import GPT, GPTConfig, Block

__all__ = ["CausalSelfAttention", "GPT", "GPTConfig", "Block"]
