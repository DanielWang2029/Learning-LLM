"""Chain of Draft (Xu et al. 2025) — minimal reproduction.

Exposes a tiny decoder-only LM and the multi-step arithmetic task rendered in
three prompting styles: Standard (direct), Chain of Thought, and Chain of Draft.
"""

from .model import TinyGPT
from .task import (
    Tokenizer,
    STYLES,
    STYLE_NAME,
    STANDARD,
    COT,
    COD,
    make_problem,
    prompt_of,
    full_example,
    parse_answer,
    RENDER,
)

__all__ = [
    "TinyGPT",
    "Tokenizer",
    "STYLES",
    "STYLE_NAME",
    "STANDARD",
    "COT",
    "COD",
    "make_problem",
    "prompt_of",
    "full_example",
    "parse_answer",
    "RENDER",
]
