"""End-to-end Tree-of-Thoughts demo (CPU, well under 60s).

Solves a set of Game-of-24 puzzles two ways:

* GREEDY chain of thought — beam width 1: commit to the single best-looking next
  operation at each step, no backtracking (the paper's CoT failure mode).
* TREE OF THOUGHTS — BFS over partial states with a heuristic evaluator, pruning
  impossible branches and keeping a beam of promising ones.

Reports success rates for both, prints a worked search, and writes a real search
tree to ``data/tot_results.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import can_reach_24, search

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_instances(n: int, seed: int, lo: int = 1, hi: int = 13) -> list[tuple[int, ...]]:
    """Sample solvable 4-number Game-of-24 instances (integer-op solvable)."""
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    while len(out) < n:
        nums = tuple(sorted(rng.randint(lo, hi) for _ in range(4)))
        if nums in seen:
            continue
        seen.add(nums)
        if can_reach_24(nums):
            out.append(nums)
    return out


def tree_to_json(result):
    return {
        "solved": result.solved,
        "beam": result.beam,
        "n_expanded": result.n_expanded,
        "solution_path": result.solution_path,
        "nodes": [
            {"id": n.id, "parent": n.parent, "nums": list(n.nums), "op": n.op,
             "depth": n.depth, "label": n.label, "status": n.status}
            for n in result.nodes
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-instances", type=int, default=100)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    start = time.time()
    instances = make_instances(args.n_instances, seed=args.seed)
    print(f"{len(instances)} solvable Game-of-24 instances | "
          f"ToT beam width = {args.beam}\n")

    greedy_solved = 0
    tot_solved = 0
    tot_expanded = 0
    fails_greedy_ok_tot = []
    for nums in instances:
        g = search(nums, beam=1)
        t = search(nums, beam=args.beam)
        greedy_solved += int(g.solved)
        tot_solved += int(t.solved)
        tot_expanded += t.n_expanded
        if t.solved and not g.solved:
            fails_greedy_ok_tot.append(nums)

    elapsed = time.time() - start
    greedy_rate = greedy_solved / len(instances)
    tot_rate = tot_solved / len(instances)

    print("==================== SUCCESS RATES ====================")
    print(f"  greedy chain of thought (beam 1) : {greedy_rate*100:5.1f}%  "
          f"({greedy_solved}/{len(instances)})")
    print(f"  tree of thoughts (beam {args.beam})       : {tot_rate*100:5.1f}%  "
          f"({tot_solved}/{len(instances)})")
    print(f"  avg nodes expanded by ToT        : {tot_expanded/len(instances):.1f}")

    # Pick a showcase instance ToT solves but greedy fails, preferring the
    # cleanest solution (fewest negative intermediate values).
    def solution_negatives(nums):
        return " , ".join(search(nums, beam=args.beam).solution_path).count("(")
    if fails_greedy_ok_tot:
        showcase = min(fails_greedy_ok_tot, key=solution_negatives)
    else:
        showcase = instances[0]
    g = search(showcase, beam=1)
    t = search(showcase, beam=args.beam)
    print(f"\n  Showcase puzzle: {list(showcase)}  ->  reach 24")
    print(f"    greedy (beam 1): {'SOLVED' if g.solved else 'FAILED'}"
          + (f"  {' , '.join(g.solution_path)}" if g.solved else " (committed to a dead end)"))
    print(f"    tree-of-thoughts: {'SOLVED' if t.solved else 'FAILED'}"
          + (f"  {' , '.join(t.solution_path)}" if t.solved else ""))
    print(f"    ToT explored {len(t.nodes)} nodes "
          f"({sum(n.status=='pruned' for n in t.nodes)} pruned).")
    print(f"\nFinished in {elapsed:.2f}s")

    out = {
        "config": {"n_instances": len(instances), "beam": args.beam,
                   "seconds": round(elapsed, 2)},
        "success": {
            "greedy": round(greedy_rate, 4), "tot": round(tot_rate, 4),
            "greedy_solved": greedy_solved, "tot_solved": tot_solved,
            "total": len(instances),
        },
        "showcase": {
            "nums": list(showcase),
            "greedy": tree_to_json(g),
            "tot": tree_to_json(t),
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "tot_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    if tot_rate <= greedy_rate:
        raise SystemExit("Tree-of-thoughts did not beat greedy — retune beam/heuristic.")
    print("\nOK: tree-of-thoughts search beats the greedy chain-of-thought baseline.")


if __name__ == "__main__":
    main()
