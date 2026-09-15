"""Miniature RLHF pipeline from InstructGPT (Ouyang et al., 2022).

Modules mirror the paper's three stages:

- `data`          the toy "sort the tokens" task = a reproducible stand-in for
                  human preferences.
- `model`         `TinyLM`, the GPT-style policy (SFT model / RL policy).
- `reward_model`  `RewardModel` + Bradley-Terry loss (Stage 2).
- `rlhf`          best-of-n rejection sampling and REINFORCE helpers (Stage 3).
"""

from . import data
from .model import TinyLM
from .reward_model import RewardModel, bradley_terry_loss
from . import rlhf

__all__ = ["data", "TinyLM", "RewardModel", "bradley_terry_loss", "rlhf"]
