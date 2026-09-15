"""Sparks of AGI (Bubeck et al., 2023) — reproduced as a *methodology*.

The paper introduces no new algorithm; it probes a model with a battery of
tasks and reports where capabilities appear. This package provides that
harness: a tiny GPT (``model``), a battery of transparent probes (``tasks``),
and a runner that reports a capability profile per scale (``probe``).
"""

from .model import GPT, GPTConfig
from .probe import PASS_THRESHOLD, capability_profile, evaluate_task

__all__ = ["GPT", "GPTConfig", "capability_profile", "evaluate_task",
           "PASS_THRESHOLD"]
