"""s1 — Simple Test-Time Scaling via budget forcing (Muennighoff et al., 2025).

Minimal from-scratch reproduction of arXiv:2501.19393: a tiny curated SFT set
teaches a reasoning format, and **budget forcing** ("Wait" to think more, an end
token to stop) controls test-time compute — more forced thinking, more accuracy.
"""

from .task import (OPS, N_OPS, MOD, apply_op, solve, make_problems, render,
                   sample_length)
from .reasoner import S1Reasoner

__all__ = ["OPS", "N_OPS", "MOD", "apply_op", "solve", "make_problems",
           "render", "sample_length", "S1Reasoner"]
