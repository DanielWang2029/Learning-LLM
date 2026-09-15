"""Define the toy environment and dump its layout + transition table.

The layout is a 6×6 grid with a vertical wall (a gap in the last row), so the
shortest route from the top-left start to the top-right goal must detour around
it — a small but non-trivial dynamics the world model has to learn.

Writes ``data/env.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import GridWorld  # noqa: E402

SIZE = 6
# Vertical wall at column 3 for rows 0..4 (row 5 is the gap).
WALLS = sorted({r * SIZE + 3 for r in range(SIZE - 1)})
START = 0            # top-left
GOAL = SIZE - 1      # top-right (5)


def main() -> None:
    env = GridWorld(SIZE, set(WALLS))
    payload = {
        "size": SIZE,
        "walls": WALLS,
        "start": START,
        "goal": GOAL,
        "n_cells": env.n_cells,
        "transitions": env.transitions(),
    }
    path = Path(__file__).resolve().parent / "env.json"
    path.write_text(json.dumps(payload))
    print(f"Wrote {path} : {SIZE}x{SIZE} grid, {len(WALLS)} walls, "
          f"{len(payload['transitions'])} transitions")


if __name__ == "__main__":
    main()
