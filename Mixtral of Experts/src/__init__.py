"""Sparse Mixture-of-Experts (Mixtral of Experts, Mistral AI 2024).

Exposes the top-2 MoE layer, the tiny MoE transformer, and the multi-domain
toy dataset used to show expert specialization.
"""

from .moe import MoELayer
from .model import MoETransformer, MoEConfig
from .data import make_dataset, N_DOMAINS, VOCAB_SIZE, SEQ_LEN, DOMAIN_SIZE

__all__ = [
    "MoELayer",
    "MoETransformer",
    "MoEConfig",
    "make_dataset",
    "N_DOMAINS",
    "VOCAB_SIZE",
    "SEQ_LEN",
    "DOMAIN_SIZE",
]
