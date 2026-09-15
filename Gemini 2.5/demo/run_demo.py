"""End-to-end Gemini 2.5 demo: test-time compute scaling (CPU, <60s).

Trains a small thinking reasoner, then shows that accuracy rises as we spend more
test-time compute — here, more sampled reasoning paths aggregated by majority vote
(self-consistency). This reproduces Gemini 2.5's documented headline: a thinking
model whose accuracy scales with test-time compute, then saturates.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared CPU box: avoid thread oversubscription

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (MOD, N_OPS, StepReasoner, apply_op, make_problems, render)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def train_step_reasoner(steps, lr, gen):
    model = StepReasoner()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        v = torch.randint(0, MOD, (256,), generator=gen)
        o = torch.randint(0, N_OPS, (256,), generator=gen)
        y = torch.tensor([apply_op(int(v[i]), int(o[i])) for i in range(256)])
        loss = loss_fn(model.step_logits(v, o), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def accuracy_at_budget(model, problems, k, temperature, noise, gen):
    correct = 0
    for p in problems:
        ans, _ = model.self_consistency(p["start"], p["op_ids"], k,
                                        temperature, noise, gen)
        correct += int(ans == p["answer"])
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--train-iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=3.5)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    test = make_problems(args.n_test, args.length, gen)
    print(f"Task: {args.length}-step modular arithmetic (mod {MOD}) | "
          f"{len(test)} test problems")
    print("Test-time scaling via self-consistency (parallel paths + majority vote)\n")

    t0 = time.time()
    model = train_step_reasoner(args.train_iters, args.lr, gen)
    print(f"Trained thinking reasoner in {time.time()-t0:.1f}s "
          f"(decoded stochastically: temp={args.temperature}, noise={args.noise})\n")

    curve = []
    for k in args.budgets:
        acc = accuracy_at_budget(model, test, k, args.temperature, args.noise, gen)
        thinking_tokens = k * args.length  # paths x steps ≈ total thinking tokens
        curve.append({"paths": k, "thinking_tokens": thinking_tokens,
                      "accuracy": round(acc, 4)})
        print(f"  paths(compute) {k:3d} | thinking tokens {thinking_tokens:4d} "
              f"| accuracy {acc*100:5.1f}%")

    # A concrete vote example: show the ballot for one problem at the max budget.
    ex = test[0]
    kmax = args.budgets[-1]
    ans, votes = model.self_consistency(ex["start"], ex["op_ids"], kmax,
                                        args.temperature, args.noise, gen)
    ans1, _ = model.self_consistency(ex["start"], ex["op_ids"], 1,
                                     args.temperature, args.noise, gen)
    print(f"\nExample: {render(ex['start'], ex['op_ids'])}  true={ex['answer']}")
    print(f"  single path (compute=1):   {ans1}  "
          f"{'OK' if ans1==ex['answer'] else 'WRONG'}")
    print(f"  majority of {kmax} paths:      {ans}  "
          f"{'OK' if ans==ex['answer'] else 'WRONG'}")
    print(f"  vote tally over answers 0..9: {votes}")

    out = {
        "config": {"length": args.length, "n_test": len(test), "mod": MOD,
                   "temperature": args.temperature, "noise": args.noise,
                   "seconds": round(time.time() - t0, 1)},
        "scaling_curve": curve,
        "single_path_accuracy": curve[0]["accuracy"],
        "max_budget_accuracy": curve[-1]["accuracy"],
        "example": {"problem": render(ex["start"], ex["op_ids"]),
                    "true": ex["answer"], "single_answer": ans1,
                    "majority_answer": ans, "votes": votes, "paths": kmax},
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "gemini25_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if curve[-1]["accuracy"] - curve[0]["accuracy"] < 0.15:
        raise SystemExit("Test-time scaling did not clearly improve accuracy.")
    print("\nOK: accuracy scales with test-time compute (sampled reasoning paths) "
          "and saturates — Gemini 2.5's documented thinking behaviour, at small scale.")


if __name__ == "__main__":
    main()
