"""From-scratch reproduction of Qwen-AgentWorld's core idea: a Language World
Model (LWM).

Qwen-AgentWorld (Alibaba, 2026) argues that the missing piece for general
agents is *world modeling*: while a policy maps state → action, a world model
maps (state, action) → next state. A learned world model can then be used as a
**decoupled environment simulator** — the agent plans and rolls out trajectories
in imagination rather than in the (costly, risky, irreversible) real
environment.

Public API:

- ``GridWorld``    a tiny toy environment with deterministic dynamics
- ``WorldModel``   a small model that predicts next-state from (state, action)
- ``plan``         breadth-first planning that runs entirely in imagination
"""

from .env import GridWorld
from .world_model import WorldModel, plan

__all__ = ["GridWorld", "WorldModel", "plan"]
