"""Serialize a few sample problems + the curated-vs-test length skew.

The demo builds its own seeded curated/test sets; this writes a few readable
examples and the length histograms. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import OPS, make_problems, render

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    curated = make_problems(1000, 8, gen, skew=True)
    test = make_problems(1000, 8, gen, skew=False)
    curated_hist = dict(sorted(Counter(p["length"] for p in curated).items()))
    test_hist = dict(sorted(Counter(p["length"] for p in test).items()))
    examples = make_problems(8, 8, gen, skew=False)
    payload = {
        "description": "Multi-step modular arithmetic (mod 10). Curated SFT set is "
                       "skewed short; budget forcing ('Wait') extends thinking at test time.",
        "ops": OPS,
        "curated_length_histogram": curated_hist,
        "test_length_histogram": test_hist,
        "examples": [{"problem": render(p["start"], p["op_ids"]),
                      "answer": p["answer"]} for p in examples],
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote curated/test histograms + {len(examples)} examples -> {out}")


if __name__ == "__main__":
    main()
