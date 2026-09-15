"""A Survey of Large Language Models (Zhao et al., 2023).

A survey, not an algorithm — so this package offers two things: a runnable
*mini lifecycle* that touches every stage the survey covers (pre-training,
SFT, preference optimization), and the model that flows through it. The survey's
taxonomy of techniques lives in the visualization.
"""

from .model import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
