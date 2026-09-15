"""Phi-3 "textbooks are all you need": data quality beats data quantity.

Exposes a tiny decoder-only LM and the clean/noisy dataset generators used to
demonstrate the paper's central thesis at laptop-CPU scale.
"""

from .model import TinyGPT, GPTConfig
from .data import (
    make_clean_dataset,
    make_noisy_dataset,
    next_token_accuracy,
    VOCAB_SIZE,
    SEQ_LEN,
)

__all__ = [
    "TinyGPT",
    "GPTConfig",
    "make_clean_dataset",
    "make_noisy_dataset",
    "next_token_accuracy",
    "VOCAB_SIZE",
    "SEQ_LEN",
]
