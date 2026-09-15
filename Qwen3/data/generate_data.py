"""Serialize a small sample of the multi-step task to data/sample_dataset.json.

The demo generates its own train/test problems on the fly (seeded); this just
writes a few human-readable examples so you can inspect the task and its
transparent <think> traces. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import OPS, make_problems, render, render_think

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    problems = make_problems(8, min_len=2, max_len=8, generator=gen)
    payload = {
        "description": "Multi-step modular arithmetic: apply a chain of ops to a "
                       "start value (mod 10). Thinking executes it step by step.",
        "ops": OPS,
        "examples": [
            {
                "problem": render(p["start"], p["op_ids"]),
                "think": render_think(p["start"], p["op_ids"]),
                "answer": p["answer"],
                "length": p["length"],
            }
            for p in problems
        ],
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(problems)} example problems -> {out}")


if __name__ == "__main__":
    main()
