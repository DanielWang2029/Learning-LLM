"""Search-R1 — RL for interleaved reasoning + search (Jin et al., 2025).

Minimal from-scratch reproduction of arXiv:2503.09516: a policy is trained by
reinforcement learning (REINFORCE with a GRPO-style group baseline) to interleave
reasoning with real retrieval calls, and learns a multi-hop search procedure that
generalizes to entities it never trained on.
"""

from .corpus import (Corpus, PERSON, CITY, COUNTRY, YEAR, COLOR, TYPE_NAMES)
from .policy import SearchPolicy, NoSearchClassifier
from .grpo import group_advantages

__all__ = ["Corpus", "PERSON", "CITY", "COUNTRY", "YEAR", "COLOR", "TYPE_NAMES",
           "SearchPolicy", "NoSearchClassifier", "group_advantages"]
