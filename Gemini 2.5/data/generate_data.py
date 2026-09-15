"""Serialize a few sample problems to data/sample_dataset.json.

The demo generates its own seeded problems; this writes a few readable examples.
Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import OPS, make_problems, render

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    problems = make_problems(8, length=6, generator=gen)
    payload = {
        "description": "6-step modular arithmetic (mod 10). A thinking model scales "
                       "test-time compute by sampling more reasoning paths and voting.",
        "ops": OPS,
        "examples": [{"problem": render(p["start"], p["op_ids"]),
                      "answer": p["answer"]} for p in problems],
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(problems)} example problems -> {out}")


if __name__ == "__main__":
    main()
