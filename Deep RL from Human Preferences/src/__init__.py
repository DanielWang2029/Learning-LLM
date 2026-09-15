"""Deep RL from Human Preferences (Christiano et al., 2017), in miniature.

- `gridworld`     the environment with a HIDDEN true reward + value iteration.
- `preferences`   segment sampling and preference labeling (the simulated human).
- `reward_model`  the learned reward network + Bradley-Terry loss.
"""

from . import gridworld, preferences, reward_model

__all__ = ["gridworld", "preferences", "reward_model"]
