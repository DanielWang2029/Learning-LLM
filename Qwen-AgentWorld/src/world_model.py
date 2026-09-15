"""The Language World Model (LWM) and imagination-based planning.

The world model predicts the next state from (state, action) — the defining
equation of Qwen-AgentWorld: (states, actions) → subsequent states. We describe
each state by *tokens* (its row and column) rather than one opaque id, so the
model can learn the environment's dynamics (e.g. "right → column+1") as a rule
and generalize to (state, action) pairs it never saw in training — exactly the
point of a world model trained on broad interaction data. Two heads predict the
next row and next column.

Once trained, ``plan`` searches for a route to a goal by rolling out entirely
*in imagination* (only ever calling the model), never touching the real
environment.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

N_ACTIONS = 4


class WorldModel(nn.Module):
    """Predict the next (row, col) from (row, col, action) tokens."""

    def __init__(self, size: int, d_model: int = 64) -> None:
        super().__init__()
        self.size = size
        self.n_cells = size * size
        self.row_emb = nn.Embedding(size, d_model)
        self.col_emb = nn.Embedding(size, d_model)
        self.action_emb = nn.Embedding(N_ACTIONS, d_model)
        self.trunk = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_row = nn.Linear(d_model, size)
        self.head_col = nn.Linear(d_model, size)

    def _rc(self, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return cell // self.size, cell % self.size

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        r, c = self._rc(state)
        h = torch.cat(
            [self.row_emb(r), self.col_emb(c), self.action_emb(action)], dim=-1
        )
        h = self.trunk(h)
        return self.head_row(h), self.head_col(h)

    @torch.no_grad()
    def predict(self, state: int, action: int) -> int:
        """Most likely next state under the world model (one imagined step)."""
        s = torch.tensor([state])
        a = torch.tensor([action])
        lr, lc = self.forward(s, a)
        nr = int(lr.argmax(dim=-1).item())
        nc = int(lc.argmax(dim=-1).item())
        return nr * self.size + nc


def loss_fn(model: WorldModel, state, action, next_state):
    """Cross-entropy on the next-row and next-col heads."""
    lr, lc = model(state, action)
    nr, nc = next_state // model.size, next_state % model.size
    return torch.nn.functional.cross_entropy(lr, nr) + torch.nn.functional.cross_entropy(lc, nc)


def accuracy(model: WorldModel, state, action, next_state) -> float:
    lr, lc = model(state, action)
    pred = lr.argmax(-1) * model.size + lc.argmax(-1)
    return (pred == next_state).float().mean().item()


def plan(model: WorldModel, start: int, goal: int, max_states: int = 400) -> Optional[List[int]]:
    """Breadth-first search for an action sequence, purely in imagination.

    Expands states using ``model.predict`` only — the real environment is never
    queried. Returns the list of actions reaching ``goal``, or None.
    """
    if start == goal:
        return []
    frontier = deque([start])
    came_from = {start: (None, None)}
    while frontier and len(came_from) < max_states:
        s = frontier.popleft()
        for a in range(N_ACTIONS):
            ns = model.predict(s, a)
            if ns not in came_from:
                came_from[ns] = (s, a)
                if ns == goal:
                    return _reconstruct(came_from, goal)
                frontier.append(ns)
    return None


def _reconstruct(came_from, goal) -> List[int]:
    actions = []
    cur = goal
    while came_from[cur][0] is not None:
        prev, act = came_from[cur]
        actions.append(act)
        cur = prev
    actions.reverse()
    return actions
