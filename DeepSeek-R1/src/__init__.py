"""DeepSeek-R1 — minimal, from-scratch reproduction of GRPO-incentivized reasoning.

DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning" (2025), the paper included in this folder.

Public API:

- ``TinyPolicy``                the RL-trained decoder-only policy.
- ``Vocab`` / task helpers      the toy ``{think}[answer]`` arithmetic task + reward.
- ``group_advantages`` / ``grpo_loss``   the GRPO update.
"""

from .policy import ParametrizedReasoner, compose
from .task import (
    Vocab,
    VOCAB_CHARS,
    TAG_LEGEND,
    prompt_str,
    all_prompts,
    reward,
    R_MAX,
)
from .grpo import group_advantages, grpo_loss

__all__ = [
    "ParametrizedReasoner",
    "compose",
    "Vocab",
    "VOCAB_CHARS",
    "TAG_LEGEND",
    "prompt_str",
    "all_prompts",
    "reward",
    "R_MAX",
    "group_advantages",
    "grpo_loss",
]
