"""Synthetic in-context learning tasks for the GPT-3 few-shot demo.

We define a small *family* of rules. Each rule is a permutation of a symbol
alphabet (a fixed input→output table). The permutations are chosen to *overlap*
partially, so a single demonstration is usually ambiguous — it is consistent
with several rules — and more demonstrations are needed to pin the rule down.

A prompt is a run of ``x>y`` demonstration pairs from one rule, followed by a
query ``x>``; the model must output the correct ``y``. Because the query symbol
is not shown among the demonstrations, the model cannot copy the answer: it has
to (a) figure out which rule the demonstrations imply, then (b) apply that rule
to the new symbol. This is exactly in-context learning, and accuracy improves
as the number of shots K grows.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

SYMBOLS = "abcdefgh"  # n = 8
PAIR_SEP = " "
MAP = ">"
END = "#"


def _agreement(p, q) -> float:
    return sum(1 for a, b in zip(p, q) if a == b) / len(p)


def build_tasks(n_tasks: int = 4, seed: int = 7):
    """Return ``n_tasks`` permutations with controlled pairwise overlap.

    Each task is a dict mapping every symbol to its image. Tasks are selected so
    their pairwise agreement lies in a moderate band, making single
    demonstrations only partially informative.
    """
    rng = random.Random(seed)
    base = list(SYMBOLS)
    chosen = []
    attempts = 0
    while len(chosen) < n_tasks and attempts < 20000:
        attempts += 1
        perm = base[:]
        rng.shuffle(perm)
        # No fixed points, so a demo always carries information.
        if any(a == b for a, b in zip(base, perm)):
            continue
        ok = all(0.15 <= _agreement(perm, c) <= 0.5 for c in chosen)
        if ok:
            chosen.append(perm)
    tasks = [dict(zip(base, perm)) for perm in chosen]
    return tasks


def format_pair(task, x) -> str:
    return f"{x}{MAP}{task[x]}"


def make_sequence(task, k, rng: random.Random) -> str:
    """K demo pairs + one query pair, e.g. 'a>c d>f b>e#' (query = last pair)."""
    syms = list(SYMBOLS)
    rng.shuffle(syms)
    demo_inputs = syms[:k]
    remaining = syms[k:] if k < len(syms) else syms
    query = rng.choice(remaining)
    pairs = [format_pair(task, x) for x in demo_inputs] + [format_pair(task, query)]
    return PAIR_SEP.join(pairs) + END


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tasks = build_tasks(args.n_tasks, args.seed)
    rng = random.Random(0)
    preview = [make_sequence(tasks[i % len(tasks)], rng.randint(1, 5), rng) for i in range(6)]

    alphabet = sorted(set(SYMBOLS + MAP + PAIR_SEP + END))
    payload = {
        "description": "In-context few-shot rule family (overlapping permutations).",
        "symbols": SYMBOLS,
        "map_token": MAP,
        "end_token": END,
        "tasks": [{k: v for k, v in t.items()} for t in tasks],
        "alphabet": alphabet,
        "examples_preview": preview,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(tasks)} rules over {len(SYMBOLS)} symbols -> {out}")
    for i, t in enumerate(tasks):
        print(f"  rule {i}: " + " ".join(f"{k}->{v}" for k, v in t.items()))
    print("preview:", "   ".join(preview[:3]))


if __name__ == "__main__":
    main()
