"""Minimal, CPU-friendly components for the GPT-4 *predictable scaling* demo.

GPT-4 itself is a closed model with no published architecture or training
recipe. This package reproduces the *documented, verifiable* methodology from
the GPT-4 Technical Report (§ "Predictable Scaling"): fit a scaling law on a
family of small models and use it to predict the loss of a larger model
*before* training it.

Public API
----------
- ``GPT`` / ``GPTConfig``   a tiny decoder-only language model (the "GPT" shape)
- ``fit_power_law`` / ``predict``   the scaling-law fit used to extrapolate loss
"""

from .model import GPT, GPTConfig
from .scaling import ScalingLaw, fit_power_law

__all__ = ["GPT", "GPTConfig", "ScalingLaw", "fit_power_law"]
