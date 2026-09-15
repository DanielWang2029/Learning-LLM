"""End-to-end s1 demo: budget forcing test-time scaling (CPU, <60s).

1. Train the arithmetic step model, and SFT a stop head on a *tiny curated* set
   (~1K examples) skewed toward SHORT chains — so the model learns to stop
   thinking early and under-thinks on long problems.
2. Apply BUDGET FORCING: append "Wait" to override early stops and force more
   reasoning steps. Sweep the minimum forced-thinking budget and show accuracy
   climb — a clean test-time-scaling curve.

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

from src import (MOD, N_OPS, OPS, S1Reasoner, apply_op, make_problems, render)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def train_step(model, iters, lr, gen):
    opt = torch.optim.Adam(model.step_net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(iters):
        v = torch.randint(0, MOD, (256,), generator=gen)
        o = torch.randint(0, N_OPS, (256,), generator=gen)
        y = torch.tensor([apply_op(int(v[i]), int(o[i])) for i in range(256)])
        loss = loss_fn(model.step_logits(v, o), y)
        opt.zero_grad(); loss.backward(); opt.step()


def sft_stop_head(model, curated, iters, lr):
    """SFT the stop head on curated traces: continue at steps 0..L-1, stop at L."""
    # Build (step_index, target) pairs from the curated set.
    idxs, targets = [], []
    for p in curated:
        L = p["length"]
        for t in range(L):
            idxs.append(t); targets.append(0)  # continue
        idxs.append(L); targets.append(1)      # stop at the end
    idxs = torch.tensor(idxs).clamp(max=model.max_len)
    targets = torch.tensor(targets)
    opt = torch.optim.Adam([model.stop_logits], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(iters):
        logits = model.stop_logits[idxs]
        loss = loss_fn(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_budget(model, problems, min_think):
    correct = steps = waits = 0
    for p in problems:
        out = model.decode(p["start"], p["op_ids"], min_think=min_think)
        correct += int(out["answer"] == p["answer"])
        steps += out["steps"]; waits += out["waits"]
    n = len(problems)
    return correct / n, steps / n, waits / n


def trace_to_str(trace):
    parts = []
    for e in trace:
        if e["kind"] == "start":
            parts.append(f"start={e['value']}")
        elif e["kind"] == "wait":
            parts.append("Wait")
        else:
            parts.append(f"{OPS[e['op']]}->{e['value']}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--curated-size", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--step-iters", type=int, default=800)
    ap.add_argument("--sft-iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    curated = make_problems(args.curated_size, args.max_len, gen, skew=True)
    test = make_problems(args.n_test, args.max_len, gen, skew=False)
    avg_curated_len = sum(p["length"] for p in curated) / len(curated)
    print(f"Task: multi-step modular arithmetic (mod {MOD}), chain length 1..{args.max_len}")
    print(f"Curated SFT set: {len(curated)} examples (skewed short, avg length "
          f"{avg_curated_len:.1f}) | test: {len(test)} (uniform lengths)\n")

    t0 = time.time()
    model = S1Reasoner(args.max_len)
    train_step(model, args.step_iters, args.lr, gen)
    sft_stop_head(model, curated, args.sft_iters, args.lr)
    print(f"Trained step model + SFT'd stop head in {time.time()-t0:.1f}s")
    natural_stop = next((t for t in range(args.max_len + 1) if model.wants_stop(t)),
                        args.max_len)
    print(f"Model's natural stopping point: ~{natural_stop} steps "
          f"(it under-thinks on longer chains)\n")

    curve = []
    for b in range(0, args.max_len + 1):
        acc, steps, waits = eval_budget(model, test, b)
        curve.append({"min_think": b, "accuracy": round(acc, 4),
                      "avg_steps": round(steps, 2), "avg_waits": round(waits, 2)})
        print(f"  forced min-thinking {b:2d} | accuracy {acc*100:5.1f}% | "
              f"avg steps {steps:4.1f} | avg 'Wait's {waits:4.1f}")

    print(f"\nNatural decoding (no forcing) accuracy: {curve[0]['accuracy']*100:.1f}%")
    print(f"Budget-forced (min {args.max_len}) accuracy:  {curve[-1]['accuracy']*100:.1f}%")

    # Concrete before/after trace on a long problem the model under-thinks.
    long_probs = [p for p in test if p["length"] == args.max_len]
    ex = long_probs[0]
    natural = model.decode(ex["start"], ex["op_ids"], min_think=0)
    forced = model.decode(ex["start"], ex["op_ids"], min_think=args.max_len)
    print(f"\nExample: {render(ex['start'], ex['op_ids'])}   true={ex['answer']}")
    print(f"  natural : <think> {trace_to_str(natural['trace'])} </think> "
          f"answer={natural['answer']}  {'OK' if natural['answer']==ex['answer'] else 'WRONG'} "
          f"({natural['steps']} steps)")
    print(f"  forced  : <think> {trace_to_str(forced['trace'])} </think> "
          f"answer={forced['answer']}  {'OK' if forced['answer']==ex['answer'] else 'WRONG'} "
          f"({forced['steps']} steps, {forced['waits']} 'Wait's)")

    out = {
        "config": {"max_len": args.max_len, "curated_size": len(curated),
                   "n_test": len(test), "avg_curated_len": round(avg_curated_len, 2),
                   "natural_stop": natural_stop, "seconds": round(time.time() - t0, 1)},
        "curve": curve,
        "natural_accuracy": curve[0]["accuracy"],
        "forced_accuracy": curve[-1]["accuracy"],
        "example": {
            "problem": render(ex["start"], ex["op_ids"]), "true": ex["answer"],
            "natural": {"trace": trace_to_str(natural["trace"]),
                        "answer": natural["answer"], "steps": natural["steps"]},
            "forced": {"trace": trace_to_str(forced["trace"]),
                       "answer": forced["answer"], "steps": forced["steps"],
                       "waits": forced["waits"]},
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "s1_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if curve[-1]["accuracy"] - curve[0]["accuracy"] < 0.15:
        raise SystemExit("Budget forcing did not clearly improve accuracy.")
    print("\nOK: forcing more thinking with 'Wait' extends reasoning and raises "
          "accuracy — s1's budget forcing, at small scale.")


if __name__ == "__main__":
    main()
