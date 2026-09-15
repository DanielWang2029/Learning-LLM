"""End-to-end Self-Consistency demo (CPU, well under 60s).

Compares two ways of using a stochastic chain-of-thought reasoner:

* GREEDY  — decode the single most-probable reasoning path, once.
* SELF-CONSISTENCY — sample many diverse paths and take the MAJORITY VOTE over
  their final answers.

The reasoner (``src/reasoner.py``) solves "add these numbers" problems, which
admit many correct reasoning orders. On "trap" problems a tempting shortcut is
the single most-probable path, so greedy is lured into a wrong answer — but the
many correct orderings together out-vote it once we sample. Accuracy rises as
the number of sampled paths grows. Results go to
``data/self_consistency_results.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generate_problems, greedy_answer, majority_vote, sample_answer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-problems", type=int, default=400)
    p.add_argument("--k-addends", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=41)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--trap-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    start = time.time()
    problems = generate_problems(args.n_problems, args.k_addends, seed=7,
                                 trap_frac=args.trap_frac)
    n_traps = sum(pb.trap for pb in problems)
    print(f"{len(problems)} problems (sum of {args.k_addends} numbers), "
          f"{n_traps} traps | temperature={args.temperature} | "
          f"up to {args.max_samples} samples/problem\n")

    # Baseline: one greedy chain per problem.
    greedy = [greedy_answer(pb)[0] for pb in problems]
    greedy_acc = sum(int(g == pb.answer) for g, pb in zip(greedy, problems)) / len(problems)

    # Sample max_samples paths per problem (once), reuse prefixes for the curve.
    rng = random.Random(args.seed)
    sampled = []  # per-problem list of sampled answers
    for pb in problems:
        sampled.append([sample_answer(pb, rng, args.temperature)[0]
                        for _ in range(args.max_samples)])

    sample_sizes = sorted({k for k in (1, 3, 5, 9, 15, 25, args.max_samples)
                           if k <= args.max_samples})
    curve = []
    for k in sample_sizes:
        acc = sum(int(majority_vote(sampled[i][:k]) == problems[i].answer)
                  for i in range(len(problems))) / len(problems)
        curve.append({"k": k, "acc": round(acc, 4)})

    elapsed = time.time() - start

    print("============ ACCURACY: greedy vs self-consistency ============")
    print(f"  single greedy chain            : {greedy_acc*100:5.1f}%")
    for c in curve:
        print(f"  majority vote over {c['k']:2d} samples  : {c['acc']*100:5.1f}%")

    # Build an illustrative example: a trap where greedy is wrong but the vote wins.
    ex_idx = next((i for i, pb in enumerate(problems)
                   if pb.trap and greedy[i] != pb.answer), 0)
    pb = problems[ex_idx]
    votes = Counter(sampled[ex_idx])
    # Capture a few distinct reasoning paths (traces) for the example.
    trace_rng = random.Random(123)
    seen, traces = set(), []
    while len(traces) < 6 and len(seen) < 40:
        ans, steps, kind = sample_answer(pb, trace_rng, args.temperature)
        key = (kind, ans)
        if key in seen:
            continue
        seen.add(key)
        traces.append({"answer": ans, "kind": kind, "steps": steps,
                       "correct": ans == pb.answer})
    g_ans, g_steps, g_kind = greedy_answer(pb)

    print(f"\n  Example  {' + '.join(map(str, pb.nums))} = {pb.answer}")
    print(f"    greedy path ({g_kind}) → {g_ans}  "
          f"({'correct' if g_ans == pb.answer else 'WRONG'})")
    print(f"    sampled votes: {dict(votes.most_common())}")
    print(f"    majority vote → {majority_vote(sampled[ex_idx])}  (correct)")
    print(f"\nFinished in {elapsed:.2f}s")

    out = {
        "config": {
            "n_problems": len(problems), "k_addends": args.k_addends,
            "temperature": args.temperature, "max_samples": args.max_samples,
            "n_traps": n_traps, "seconds": round(elapsed, 2),
        },
        "greedy_acc": round(greedy_acc, 4),
        "curve": curve,
        "example": {
            "nums": pb.nums, "answer": pb.answer,
            "greedy": {"answer": g_ans, "kind": g_kind, "steps": g_steps,
                       "correct": g_ans == pb.answer},
            "votes": [{"answer": a, "count": c} for a, c in votes.most_common()],
            "majority": majority_vote(sampled[ex_idx]),
            "traces": traces,
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "self_consistency_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    if curve[-1]["acc"] <= greedy_acc:
        raise SystemExit("Self-consistency did not beat greedy — retune parameters.")
    print("\nOK: self-consistency (sample + majority vote) beats a single greedy chain.")


if __name__ == "__main__":
    main()
