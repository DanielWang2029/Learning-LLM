"""End-to-end Qwen3 demo: hybrid thinking vs non-thinking + thinking budget (CPU, <60s).

Trains one *thinking* model (a step reasoner that executes the problem one op at a
time) and one *non-thinking* model (answers in a single shot) on the same
multi-step arithmetic task. Then it sweeps the **thinking budget** — the number
of reasoning steps the thinking path is allowed — and shows that accuracy climbs
from the non-thinking baseline to ~100% as the budget grows, at the cost of more
tokens (steps). This mirrors Qwen3's single-model hybrid thinking with a
controllable thinking budget.

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

from src import (MOD, N_OPS, DirectAnswerer, StepReasoner, make_problems,
                 render, render_think, PAD_OP)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def train_step_reasoner(steps, lr, device, gen):
    """Train the thinking path on random single-op transitions (value, op) -> value."""
    model = StepReasoner().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        v = torch.randint(0, MOD, (256,), generator=gen)
        o = torch.randint(0, N_OPS, (256,), generator=gen)
        from src import apply_op
        y = torch.tensor([apply_op(int(v[i]), int(o[i])) for i in range(256)])
        logits = model.step_logits(v, o)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_direct(model, problems, steps, lr, device, gen):
    """Train the non-thinking path to map whole problems to answers in one shot."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    starts = torch.tensor([p["start"] for p in problems])
    ops = torch.tensor([p["op_ids"] + [PAD_OP] * (model.max_len - len(p["op_ids"]))
                        for p in problems])
    ys = torch.tensor([p["answer"] for p in problems])
    n = len(problems)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), generator=gen)
        logits = model.logits(starts[idx], ops[idx])
        loss = loss_fn(logits, ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def thinking_accuracy(model, problems, budget):
    """Accuracy + average tokens (steps) used at a given thinking budget."""
    correct = tokens = 0
    for p in problems:
        ans, trace = model.run(p["start"], p["op_ids"], budget)
        correct += int(ans == p["answer"])
        tokens += len(trace) - 1
    return correct / len(problems), tokens / len(problems)


@torch.no_grad()
def direct_accuracy(model, problems):
    correct = sum(model.answer(p["start"], p["op_ids"]) == p["answer"]
                  for p in problems)
    return correct / len(problems)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-train-direct", type=int, default=2000)
    ap.add_argument("--step-train-iters", type=int, default=800)
    ap.add_argument("--direct-train-iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # Held-out test problems (mixed lengths 1..max_len) + training pool for direct.
    test = make_problems(args.n_test, 1, args.max_len, gen)
    train_pool = make_problems(args.n_train_direct, 1, args.max_len, gen)

    print(f"Task: multi-step modular arithmetic (mod {MOD}), chain length 1..{args.max_len}")
    print(f"Test problems: {len(test)}  |  ops: {N_OPS}\n")

    t0 = time.time()
    print("Training THINKING path (step reasoner: learns one op at a time)...")
    thinker = train_step_reasoner(args.step_train_iters, args.lr, device, gen)

    print("Training NON-THINKING path (direct one-shot answerer)...")
    direct = DirectAnswerer(args.max_len).to(device)
    direct = train_direct(direct, train_pool, args.direct_train_iters, args.lr, device, gen)
    print(f"Trained both in {time.time()-t0:.1f}s\n")

    non_thinking_acc = direct_accuracy(direct, test)

    # Sweep the thinking budget from 0 (=non-thinking) to max_len.
    curve = []
    for b in range(0, args.max_len + 1):
        acc, toks = thinking_accuracy(thinker, test, b)
        curve.append({"budget": b, "accuracy": round(acc, 4), "avg_tokens": round(toks, 2)})
        print(f"  thinking budget {b:2d} | accuracy {acc*100:5.1f}% | avg tokens {toks:4.1f}")

    print(f"\nNon-thinking (one-shot) accuracy: {non_thinking_acc*100:.1f}%")
    print(f"Thinking @ full budget ({args.max_len}) accuracy: {curve[-1]['accuracy']*100:.1f}%")

    # A concrete side-by-side example on a long problem.
    long_probs = [p for p in test if p["length"] == args.max_len]
    ex = long_probs[0]
    ans_think, trace = thinker.run(ex["start"], ex["op_ids"], args.max_len)
    ans_direct = direct.answer(ex["start"], ex["op_ids"])
    print("\nExample (length "
          f"{ex['length']}): {render(ex['start'], ex['op_ids'])}   true={ex['answer']}")
    print(f"  non-thinking -> {ans_direct}  {'OK' if ans_direct==ex['answer'] else 'WRONG'}")
    print(f"  thinking     -> <think> {render_think(ex['start'], ex['op_ids'])} </think> "
          f"answer={ans_think}  {'OK' if ans_think==ex['answer'] else 'WRONG'}")

    out = {
        "config": {"max_len": args.max_len, "n_test": len(test), "mod": MOD,
                   "n_ops": N_OPS, "seconds": round(time.time() - t0, 1)},
        "non_thinking_accuracy": round(non_thinking_acc, 4),
        "budget_curve": curve,
        "example": {
            "problem": render(ex["start"], ex["op_ids"]),
            "think": render_think(ex["start"], ex["op_ids"]),
            "trace": trace,
            "true": ex["answer"],
            "non_thinking_answer": ans_direct,
            "thinking_answer": ans_think,
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "qwen3_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if curve[-1]["accuracy"] < 0.95 or curve[-1]["accuracy"] - non_thinking_acc < 0.15:
        raise SystemExit("Thinking budget did not clearly beat non-thinking.")
    print("\nOK: enabling thinking and raising the budget lifts accuracy above the "
          "non-thinking baseline (at the cost of more tokens).")


if __name__ == "__main__":
    main()
