"""Gemini 2.5 — a thinking model with test-time compute scaling (small-scale).

Honest, minimal reproduction of the documented headline of Gemini 2.5 (Google,
2025, arXiv:2507.06261): a thinking model whose accuracy scales with test-time
compute. Gemini is closed, so we reproduce the *mechanism* — parallel sampled
reasoning paths aggregated by majority vote (self-consistency) — at tiny scale.
"""

from .task import OPS, N_OPS, MOD, apply_op, solve, make_problems, render
from .reasoner import StepReasoner

__all__ = ["OPS", "N_OPS", "MOD", "apply_op", "solve", "make_problems",
           "render", "StepReasoner"]
