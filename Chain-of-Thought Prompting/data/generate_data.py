"""Write a tiny, human-readable sample of the addition dataset to data/.

This is purely for inspection — the demo generates its own (larger) training
and evaluation splits procedurally with fixed seeds. Run with:

    python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import format_example, make_dataset

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    problems = make_dataset(12, min_digits=1, max_digits=3, seed=7)
    examples = []
    for a, b in problems:
        dp, dt = format_example(a, b, "direct")
        cp, ct = format_example(a, b, "cot")
        examples.append({
            "a": a, "b": b, "answer": a + b,
            "direct": dp + dt,
            "cot": cp + ct,
        })
    payload = {
        "description": "Multi-digit addition in DIRECT vs CHAIN-OF-THOUGHT format.",
        "format": {
            "direct": "a+b=<answer>$",
            "cot": "a+b=<digit-by-digit steps with carries>#<answer>$",
        },
        "examples": examples,
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample examples -> {out}")


if __name__ == "__main__":
    main()
