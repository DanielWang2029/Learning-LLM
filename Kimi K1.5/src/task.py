"""The toy reasoning task and its human-readable rendering.

The policy is prompted with a small problem and must output the larger of two
numbers, after a chain of scratch/"think" steps:

    prompt:      "a>b="
    completion:  "T T T … # <answer>"     (the T's are the thinking budget)

The correct answer is ``max(a, b)``. The ``T`` tokens are *scratch* steps. A
**few** think tokens are genuinely useful — the model only "reveals" a reliable
answer once it has thought for at least ``THINK_NEED`` steps (below that it is
forced to guess). But thinking *beyond* ``THINK_NEED`` adds no accuracy: those
extra tokens are pure "overthinking", exactly the overlong chains Kimi k1.5's
length penalty is designed to curb. So the ideal behaviour is to think just
``THINK_NEED`` steps — not zero, and not the maximum. The number of ``T`` tokens
is the **chain length** the RL objective learns to control.
"""

from __future__ import annotations

THINK = "T"
MARK = "#"
MAX_THINK = 8
# Minimum number of think steps needed before the answer is reliably revealed;
# thinking longer than this is wasted ("overthinking").
THINK_NEED = 3


def answer_of(a: int, b: int) -> int:
    """The task's ground-truth answer: the larger of the two numbers."""
    return max(a, b)


def prompt_of(a: int, b: int) -> str:
    # "a>b=" reads as "which of a, b is larger?"; the model outputs max(a, b).
    return f"{a}>{b}="


def render_completion(a: int, b: int, length: int, answer: int) -> str:
    """Render a full generated string for display / logging."""
    return f"{prompt_of(a, b)}{THINK * length}{MARK}{answer}"
