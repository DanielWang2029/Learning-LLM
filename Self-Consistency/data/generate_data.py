"""Write a tiny, human-readable sample of the reasoning task to data/.

For inspection only — the demo generates its own problems procedurally with a
fixed seed. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generate_problems, greedy_answer, sample_answer

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    problems = generate_problems(8, k=3, seed=7, trap_frac=0.5)
    rng = random.Random(1)
    examples = []
    for pb in problems:
        g_ans, _, g_kind = greedy_answer(pb)
        samples = [sample_answer(pb, rng, temperature=1.0)[0] for _ in range(9)]
        examples.append({
            "nums": pb.nums, "answer": pb.answer, "trap": pb.trap,
            "lure": pb.lure, "greedy_answer": g_ans, "greedy_path": g_kind,
            "sampled_answers": samples,
        })
    payload = {
        "description": "Add k numbers. Many correct orderings reach the answer; "
                       "trap problems also have a tempting wrong shortcut.",
        "examples": examples,
    }
    out = DATA_DIR / "sample_problems.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample problems -> {out}")


if __name__ == "__main__":
    main()
