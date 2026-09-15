"""A tiny deterministic gridworld — the "real environment" to be modeled.

The agent occupies a cell of an N×N grid. Four actions move it up/down/left/
right; moving into a wall or off the grid leaves it in place. This stands in for
the agentic environments (terminal, web, OS, ...) that Qwen-AgentWorld learns to
simulate — small enough to enumerate, non-trivial enough that its dynamics must
be learned.
"""

from __future__ import annotations

from typing import List, Set, Tuple

# action id -> (d_row, d_col)
ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]


class GridWorld:
    def __init__(self, size: int = 6, walls: Set[int] | None = None) -> None:
        self.size = size
        self.walls: Set[int] = walls or set()

    @property
    def n_cells(self) -> int:
        return self.size * self.size

    def rc(self, cell: int) -> Tuple[int, int]:
        return divmod(cell, self.size)

    def cell(self, r: int, c: int) -> int:
        return r * self.size + c

    def step(self, state: int, action: int) -> int:
        """Deterministic transition: (state, action) -> next_state."""
        r, c = self.rc(state)
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return state  # off-grid -> stay
        nxt = self.cell(nr, nc)
        if nxt in self.walls:
            return state  # wall -> stay
        return nxt

    def transitions(self) -> List[Tuple[int, int, int]]:
        """All (state, action, next_state) triples for non-wall states."""
        out = []
        for s in range(self.n_cells):
            if s in self.walls:
                continue
            for a in range(len(ACTIONS)):
                out.append((s, a, self.step(s, a)))
        return out
