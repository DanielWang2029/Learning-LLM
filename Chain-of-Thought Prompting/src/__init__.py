"""Chain-of-Thought Prompting — minimal, from-scratch reproduction.

Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language
Models" (2022), the paper included in this folder.

The public API mirrors the pieces the demo needs:

- ``TinyGPT``            a tiny decoder-only Transformer (the "model").
- ``CharVocab``          character-level tokenizer.
- arithmetic helpers     build DIRECT vs CHAIN-OF-THOUGHT training targets.
"""

from .model import TinyGPT
from .tokenizer import CharVocab, VOCAB_CHARS
from .arithmetic import (
    cot_scratchpad,
    format_example,
    parse_answer,
    sample_problem,
    make_dataset,
)

__all__ = [
    "TinyGPT",
    "CharVocab",
    "VOCAB_CHARS",
    "cot_scratchpad",
    "format_example",
    "parse_answer",
    "sample_problem",
    "make_dataset",
]
