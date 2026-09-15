"""ReAct — minimal, from-scratch reproduction.

Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022),
the paper included in this folder.

Public API:

- ``KB`` / ``lookup`` / ``calc``   the environment and its two real tools.
- ``QUESTIONS``                    the toy multi-hop question set.
- ``react_agent`` / ``reasoning_only`` / ``acting_only`` / ``gold_answer``.
"""

from .environment import KB, lookup, calc, ToolResult
from .questions import QUESTIONS
from .agents import react_agent, reasoning_only, acting_only, gold_answer, AGENTS

__all__ = [
    "KB",
    "lookup",
    "calc",
    "ToolResult",
    "QUESTIONS",
    "react_agent",
    "reasoning_only",
    "acting_only",
    "gold_answer",
    "AGENTS",
]
