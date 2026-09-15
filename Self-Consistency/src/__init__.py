"""Self-Consistency — minimal, from-scratch reproduction.

Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language
Models" (2022), the paper included in this folder.

Public API:

- ``Problem`` / ``generate_problems``   the toy multi-path reasoning task.
- ``greedy_answer`` / ``sample_answer`` a stochastic chain-of-thought reasoner.
- ``majority_vote``                     marginalize sampled paths onto one answer.
"""

from .reasoner import (
    Problem,
    Path,
    generate_problems,
    paths_for,
    greedy_answer,
    sample_answer,
)
from .self_consistency import majority_vote

__all__ = [
    "Problem",
    "Path",
    "generate_problems",
    "paths_for",
    "greedy_answer",
    "sample_answer",
    "majority_vote",
]
