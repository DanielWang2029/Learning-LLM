"""A tiny deterministic gridworld with a HIDDEN true reward.

Christiano et al. (2017) learn a reward function from human preferences over
trajectory *segments*, without ever observing the environment's true reward.
We reproduce that on the simplest possible environment: an N×N gridworld whose
true reward field the agent (and the reward model) never see directly.

The hidden true reward is a smooth landscape that peaks at a goal cell::

    r*(s) = 1 − 0.2 · manhattan_distance(s, goal)

so the best behavior is to walk to the goal and stay there. There are no
terminal states and a `stay` action exists, i.e. this is an infinite-horizon
task. That matters: preference comparisons over equal-length segments only
identify the reward up to an affine transform, and for a discounted
infinite-horizon MDP the optimal policy is invariant to exactly that (an
overall scale and offset). So the reward we recover from preferences induces
the *same* optimal policy as the true reward.

Default 5×5 world: start = top-left (state 0), goal = bottom-right.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Actions: up, down, left, right, stay
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACTION_NAMES = ["↑", "↓", "←", "→", "•"]


class GridWorld:
    def __init__(self, n: int = 5, gamma: float = 0.9) -> None:
        self.n = n
        self.gamma = gamma
        self.num_states = n * n
        self.start = self.rc_to_s(0, 0)
        self.goal = self.rc_to_s(n - 1, n - 1)

        # Hidden true reward: a smooth bump peaking at the goal.
        self.true_reward = np.zeros(self.num_states, dtype=np.float64)
        gr, gc = self.s_to_rc(self.goal)
        for s in range(self.num_states):
            r, c = self.s_to_rc(s)
            dist = abs(r - gr) + abs(c - gc)
            self.true_reward[s] = 1.0 - 0.2 * dist

    # -- index helpers --------------------------------------------------- #
    def rc_to_s(self, r: int, c: int) -> int:
        return r * self.n + c

    def s_to_rc(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.n)

    def step(self, s: int, a: int) -> int:
        """Deterministic transition, clamped to the grid."""
        r, c = self.s_to_rc(s)
        dr, dc = ACTIONS[a]
        nr = min(max(r + dr, 0), self.n - 1)
        nc = min(max(c + dc, 0), self.n - 1)
        return self.rc_to_s(nr, nc)

    def state_features(self) -> np.ndarray:
        """One-hot features for every state (the reward model's input)."""
        return np.eye(self.num_states, dtype=np.float32)


def value_iteration(
    env: GridWorld, reward: np.ndarray, iters: int = 400
) -> np.ndarray:
    """Greedy policy (one action per state) for a reward field, infinite horizon.

    Planning uses `reward` (either the true or the *learned* reward). Because
    the horizon is infinite and discounted, the resulting policy is invariant to
    a positive scale and an overall offset of `reward` — precisely the freedom
    that preference learning leaves undetermined.
    """
    V = np.zeros(env.num_states)
    for _ in range(iters):
        newV = V.copy()
        for s in range(env.num_states):
            best = -1e18
            for a in range(len(ACTIONS)):
                sp = env.step(s, a)
                best = max(best, reward[sp] + env.gamma * V[sp])
            newV[s] = best
        if np.max(np.abs(newV - V)) < 1e-10:
            V = newV
            break
        V = newV

    policy = np.zeros(env.num_states, dtype=np.int64)
    for s in range(env.num_states):
        best, best_a = -1e18, 0
        for a in range(len(ACTIONS)):
            sp = env.step(s, a)
            q = reward[sp] + env.gamma * V[sp]
            if q > best:
                best, best_a = q, a
        policy[s] = best_a
    return policy


def evaluate_true_return(
    env: GridWorld, policy: np.ndarray, max_steps: int = 40
) -> float:
    """Discounted TRUE return of a policy from the start state."""
    s, total, disc = env.start, 0.0, 1.0
    for _ in range(max_steps):
        sp = env.step(s, int(policy[s]))
        total += disc * env.true_reward[sp]
        disc *= env.gamma
        s = sp
    return total


def rollout_states(
    env: GridWorld, policy: np.ndarray, max_steps: int = 20
) -> List[int]:
    """States visited from the start, trimmed once the agent settles (stays)."""
    s = env.start
    path = [s]
    for _ in range(max_steps):
        sp = env.step(s, int(policy[s]))
        if sp == s:  # settled on a cell (stay) — stop drawing repeats
            break
        path.append(sp)
        s = sp
    return path
