"""End-to-end CPU demo for Deep Delta Learning (arXiv:2601.00417).

Part A  Delta-rule associative memory. Store a set of key->value pairs in ONE
        pass, then recall them. Show (1) high retrieval accuracy, (2) that
        rewriting a key REPLACES its value (delta rule, not accumulation), and
        (3) graceful degradation as the number of stored pairs approaches the
        memory's dimension (a capacity curve).

Part B  Deep Delta Learning as a depth-wise residual interface. Verify the
        paper's local error-correction identity e_post = (1-β)·e_pre (Eq. 2.4),
        and show that stacking DDL layers drives a readout of the residual state
        to its target, with the error contracting by (1-β) per layer.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import DeltaAssociativeMemory, ddl_step, readout  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def _unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


# --------------------------------------------------------------------------- #
# Part A: associative memory
# --------------------------------------------------------------------------- #
def retrieval_accuracy(d, n_pairs, seed, beta=1.0):
    g = torch.Generator().manual_seed(seed)
    keys = _unit(torch.randn(n_pairs, d, generator=g))
    values = _unit(torch.randn(n_pairs, d, generator=g))
    mem = DeltaAssociativeMemory(d, d)
    mem.store(keys, values, beta=beta)
    recalled = mem.read_all(keys)                       # (M, d)
    # classify each recall by nearest stored value (cosine).
    sim = _unit(recalled) @ _unit(values).t()           # (M, M)
    pred = sim.argmax(dim=1)
    correct = (pred == torch.arange(n_pairs)).float().mean().item()
    mse = (recalled - values).pow(2).mean().item()
    return correct, mse


def part_a():
    print("=" * 70)
    print("PART A  Delta-rule associative memory: store once, recall")
    print("=" * 70)
    d = 32

    acc8, mse8 = retrieval_accuracy(d, 8, seed=0)
    print(f"  Stored 8 pairs in a {d}-dim memory (one pass, beta=1).")
    print(f"  Retrieval accuracy: {acc8*100:.1f}%   (mean-squared recall error {mse8:.4f})")

    # Overwrite: bind key -> v_old, then rewrite key -> v_new; recall must give v_new.
    g = torch.Generator().manual_seed(3)
    mem = DeltaAssociativeMemory(d, d)
    others_k = _unit(torch.randn(5, d, generator=g))
    others_v = _unit(torch.randn(5, d, generator=g))
    mem.store(others_k, others_v)
    key = _unit(torch.randn(d, generator=g))
    v_old = _unit(torch.randn(d, generator=g))
    v_new = _unit(torch.randn(d, generator=g))
    mem.write(key, v_old)
    before = mem.read(key)
    mem.write(key, v_new)                                # OVERWRITE
    after = mem.read(key)
    sim_old = torch.dot(_unit(after), v_old).item()
    sim_new = torch.dot(_unit(after), v_new).item()
    print(f"\n  Overwrite test: bind key->v_old, then rewrite key->v_new.")
    print(f"    recall vs v_old: cos={sim_old:+.3f}    recall vs v_new: cos={sim_new:+.3f}"
          f"   -> {'latest value recalled' if sim_new > sim_old else 'FAILED'}")

    # Capacity curve: accuracy vs number of stored pairs (avg over seeds).
    sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
    curve = []
    for m in sizes:
        accs = [retrieval_accuracy(d, m, seed=s)[0] for s in range(5)]
        curve.append({"n_pairs": m, "accuracy": sum(accs) / len(accs)})
    print(f"\n  Capacity (retrieval accuracy vs #pairs stored in a {d}-dim memory):")
    for pt in curve:
        bar = "#" * int(pt["accuracy"] * 30)
        print(f"    {pt['n_pairs']:>3} pairs | {pt['accuracy']*100:5.1f}% {bar}")

    return {
        "dim": d,
        "recall_accuracy_8": acc8,
        "recall_mse_8": mse8,
        "overwrite": {"cos_v_old": sim_old, "cos_v_new": sim_new},
        "capacity_curve": curve,
    }


# --------------------------------------------------------------------------- #
# Part B: depth-wise DDL error correction
# --------------------------------------------------------------------------- #
def part_b():
    print("\n" + "=" * 70)
    print("PART B  Deep Delta Learning: depth-wise error correction")
    print("=" * 70)
    d, dv = 24, 4
    g = torch.Generator().manual_seed(1)

    # Identity: e_post = (1 - beta) * e_pre for a single DDL rewrite.
    betas = [round(0.25 * i, 2) for i in range(9)]  # 0.0 .. 2.0
    id_check = []
    max_dev = 0.0
    for beta in betas:
        X = torch.randn(d, dv, generator=g)
        k = torch.randn(d, generator=g)
        v = torch.randn(dv, generator=g)
        e_pre = (readout(X, k) - v).norm().item()
        X2 = ddl_step(X, k, v, beta)
        e_post = (readout(X2, k) - v).norm().item()
        predicted = abs(1 - beta) * e_pre
        max_dev = max(max_dev, abs(e_post - predicted))
        id_check.append({"beta": beta, "e_pre": e_pre, "e_post": e_post,
                         "predicted": predicted})
    print("  Verify e_post = (1 - beta) * e_pre  (Eq. 2.4):")
    print(f"    {'beta':>6}{'e_pre':>10}{'e_post':>10}{'(1-b)*e_pre':>14}")
    for r in id_check:
        print(f"    {r['beta']:>6.2f}{r['e_pre']:>10.4f}{r['e_post']:>10.4f}{r['predicted']:>14.4f}")
    print(f"  max deviation from the identity: {max_dev:.2e}  (β=0 identity, β=1 exact overwrite)")

    # Depth convergence: repeatedly rewrite the SAME (k, v) across layers; the
    # readout error contracts by (1-beta) per layer.
    depth = 8
    conv = {}
    for beta in [0.5, 1.0, 1.5]:
        X = torch.randn(d, dv, generator=torch.Generator().manual_seed(7))
        k = torch.randn(d, generator=torch.Generator().manual_seed(8))
        v = torch.randn(dv, generator=torch.Generator().manual_seed(9))
        errs = [ (readout(X, k) - v).norm().item() ]
        for _ in range(depth):
            X = ddl_step(X, k, v, beta)
            errs.append((readout(X, k) - v).norm().item())
        conv[str(beta)] = [round(e, 5) for e in errs]
    print(f"\n  Readout error vs depth (drive state to target across {depth} DDL layers):")
    print(f"    {'layer':>6}" + "".join(f"{'β='+b:>12}" for b in conv))
    for l in range(depth + 1):
        print(f"    {l:>6}" + "".join(f"{conv[b][l]:>12.5f}" for b in conv))

    return {
        "identity_check": id_check,
        "max_deviation": max_dev,
        "depth_convergence": conv,
        "depth": depth,
        "d_model": d,
        "d_val": dv,
    }


def main() -> None:
    start = time.time()
    a = part_a()
    b = part_b()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Deep Delta Learning (2026)",
        "arxiv": "2601.00417",
        "associative_memory": a,
        "depth_wise_ddl": b,
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/results.json  (elapsed {time.time()-start:.1f}s)")
    print("Open visualization/index.html to explore delta memory + depth-wise DDL.")

    ok_recall = a["recall_accuracy_8"] > 0.95
    ok_overwrite = a["overwrite"]["cos_v_new"] > a["overwrite"]["cos_v_old"]
    ok_identity = b["max_deviation"] < 1e-4
    if not (ok_recall and ok_overwrite and ok_identity):
        raise SystemExit("Demo did not demonstrate the expected DDL behaviour.")
    print("\nOK: delta memory recalls & overwrites correctly; "
          "the depth-wise DDL identity e_post=(1-β)e_pre holds exactly.")


if __name__ == "__main__":
    main()
