"""Qwen3 — hybrid thinking / non-thinking reasoning with a thinking budget.

Minimal from-scratch reproduction of the Qwen3 Technical Report (Alibaba, 2025),
arXiv:2505.09388. A single task is solved two ways — a *thinking* path that
executes the problem step by step under a controllable budget, and a
*non-thinking* path that answers in one shot.
"""

from .task import (OPS, N_OPS, MOD, apply_op, solve, make_problems, render,
                   render_think)
from .reasoner import StepReasoner, DirectAnswerer, PAD_OP

__all__ = [
    "OPS", "N_OPS", "MOD", "apply_op", "solve", "make_problems", "render",
    "render_think", "StepReasoner", "DirectAnswerer", "PAD_OP",
]
