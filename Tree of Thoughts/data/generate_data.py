"""Write a tiny sample of Game-of-24 instances to data/ for inspection.

The demo generates its own instances procedurally with a fixed seed. Run with:
    python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from src import can_reach_24, search

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    rng = random.Random(1)
    seen, insts = set(), []
    while len(insts) < 8:
        nums = tuple(sorted(rng.randint(1, 13) for _ in range(4)))
        if nums in seen:
            continue
        seen.add(nums)
        if can_reach_24(nums):
            insts.append(nums)

    examples = []
    for nums in insts:
        g = search(nums, beam=1)
        t = search(nums, beam=5)
        examples.append({
            "nums": list(nums),
            "greedy_solved": g.solved,
            "tot_solved": t.solved,
            "tot_solution": t.solution_path,
        })
    payload = {
        "description": "Solvable Game-of-24 instances with greedy vs ToT outcomes.",
        "examples": examples,
    }
    out = DATA_DIR / "sample_instances.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample instances -> {out}")


if __name__ == "__main__":
    main()
