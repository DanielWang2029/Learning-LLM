"""Minimal reproduction of the LIMO thesis: *Less Is More for Reasoning*.

Exposes a tiny decoder-only reasoner (``TinyGPT``), a chained-arithmetic
reasoning task with two demonstration qualities, and SFT/eval helpers.
"""

from .model import TinyGPT
from .task import (
    Tokenizer,
    make_problem,
    high_quality_trace,
    low_quality_trace,
    build_dataset,
)

__all__ = [
    "TinyGPT",
    "Tokenizer",
    "make_problem",
    "high_quality_trace",
    "low_quality_trace",
    "build_dataset",
]
