"""Minimal reproduction of Kimi k1.5's RL-with-length-penalty for reasoning.

Exposes a tiny policy model, the toy reasoning task/tokenizer, and the
GRPO-style RL update with Kimi's length-penalty reward.
"""

from .model import TinyPolicy
from .task import answer_of, prompt_of, render_completion, MAX_THINK, THINK_NEED
from .rl import rollout, grpo_step, compute_rewards, evaluate

__all__ = [
    "TinyPolicy",
    "answer_of",
    "prompt_of",
    "render_completion",
    "MAX_THINK",
    "THINK_NEED",
    "rollout",
    "grpo_step",
    "compute_rewards",
    "evaluate",
]
