"""Tree of Thoughts — minimal, from-scratch reproduction.

Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language
Models" (2023), the paper included in this folder.

Public API:

- ``moves`` / ``evaluate`` / ``can_reach_24``   Game-of-24 generator & evaluator.
- ``search`` / ``Node`` / ``SearchResult``       BFS tree search (beam=1 = greedy CoT).
"""

from .game24 import moves, evaluate, can_reach_24, TARGET
from .search import search, Node, SearchResult

__all__ = [
    "moves",
    "evaluate",
    "can_reach_24",
    "TARGET",
    "search",
    "Node",
    "SearchResult",
]
