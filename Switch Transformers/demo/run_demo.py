"""Switch Transformer demo — expert specialization + load balancing.

Two small experiments, each on the "typed token" vocabulary (see
data/generate_data.py) where every token factors into a type g and a value v.

Experiment A — SPECIALIZATION.
  The label depends on BOTH type and value: label = TABLE[g][v]. Experts only
  see the value, the router only sees the type, so an expert given two types
  hits conflicting targets. The only low-loss routing sends each type to its own
  expert — so a per-type routing histogram becomes (near) one-hot.

Experiment B — LOAD BALANCING.
  The label depends only on the value: label = f(v), the same for every type.
  Now any expert can serve any token, so nothing forces the router to spread
  out. Without the auxiliary loss the router collapses onto a single expert;
  turning the load-balancing loss on spreads tokens evenly across all experts.

Also reported: params-vs-compute — many experts add parameters, but top-1
routing keeps the per-token FFN compute equal to one expert.

Runs on CPU in well under a minute.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
import generate_data as gd

from src.model import TinySwitchModel

DATA_DIR = ROOT / "data"

D_MODEL = 32
D_FF = 64
N_EXPERTS = 4
SEQ_LEN = 16
BATCH = 64
STEPS = 500
SEED = 0


def make_batch(n, seq_len, table, rng):
    tok = torch.randint(0, gd.VOCAB_SIZE, (n, seq_len), generator=rng)
    g = tok % gd.N_TYPES
    v = tok // gd.N_TYPES
    y = torch.tensor(table)[g, v]
    return tok, y


@torch.no_grad()
def evaluate(model, table, rng, iters=8):
    model.eval()
    correct = total = 0
    hist = np.zeros((gd.N_TYPES, N_EXPERTS))
    for _ in range(iters):
        tok, y = make_batch(256, SEQ_LEN, table, rng)
        logits, route, _ = model(tok)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        types = (tok % gd.N_TYPES).reshape(-1).numpy()
        experts = route.assignments.numpy()
        np.add.at(hist, (types, experts), 1)
    return correct / total, hist


def train(alpha, table):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    model = TinySwitchModel(gd.N_TYPES, gd.VALUES_PER_TYPE, gd.N_CLASSES,
                            D_MODEL, D_FF, n_experts=N_EXPERTS, alpha=alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(STEPS):
        tok, y = make_batch(BATCH, SEQ_LEN, table, rng)
        _, _, loss = model(tok, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc, hist = evaluate(model, table, rng)
    return model, acc, hist


def usage(hist):
    return hist.sum(axis=0) / hist.sum()


def main() -> None:
    rng = np.random.default_rng(SEED)
    typed_table = gd.build_table(rng)                       # depends on (g, v)
    shared_row = rng.integers(0, gd.N_CLASSES, size=gd.VALUES_PER_TYPE)
    shared_table = np.tile(shared_row, (gd.N_TYPES, 1))     # depends on v only

    start = time.time()
    print("=" * 68)
    print("Switch Transformer (top-1 MoE)")
    print(f"  vocab={gd.VOCAB_SIZE}  types={gd.N_TYPES}  experts={N_EXPERTS}  "
          f"classes={gd.N_CLASSES}")
    print("=" * 68)

    # ---- Experiment A: specialization
    print("\n[A] SPECIALIZATION  (label depends on type AND value)")
    model, acc, hist = train(alpha=0.05, table=typed_table)
    print(f"    task accuracy = {acc*100:.1f}%")
    print("\n    Per-type routing histogram (rows=type, cols=expert):")
    print("           " + "".join(f" exp{e} " for e in range(N_EXPERTS)))
    peak_per_type = []
    for t in range(gd.N_TYPES):
        frac = hist[t] / hist[t].sum()
        peak_per_type.append(frac.max())
        print(f"    type{t}  " + "".join(f" {f*100:3.0f}% " for f in frac))
    print(f"\n    -> every type is routed almost entirely to ONE expert "
          f"(min peak {min(peak_per_type)*100:.0f}%): experts specialized.")
    u_spec = usage(hist)
    print(f"    expert usage: " +
          "  ".join(f"exp{e}={u_spec[e]*100:.0f}%" for e in range(N_EXPERTS)))

    # ---- Experiment B: load balancing
    print("\n[B] LOAD BALANCING  (label depends on value only -> any expert works)")
    _, acc_off, hist_off = train(alpha=0.0, table=shared_table)
    _, acc_on, hist_on = train(alpha=0.05, table=shared_table)
    u_off, u_on = usage(hist_off), usage(hist_on)
    print(f"    aux OFF: accuracy={acc_off*100:.1f}%  usage=[" +
          " ".join(f"{x*100:.0f}%" for x in u_off) +
          f"]  max load={u_off.max()*100:.0f}%")
    print(f"    aux ON : accuracy={acc_on*100:.1f}%  usage=[" +
          " ".join(f"{x*100:.0f}%" for x in u_on) +
          f"]  max load={u_on.max()*100:.0f}%")
    print(f"    -> without the load-balancing loss the router collapses onto few "
          f"experts;\n       with it, tokens spread evenly "
          f"(ideal {100/N_EXPERTS:.0f}% each).")

    # ---- params vs compute
    expert_params = sum(p.numel() for p in model.switch.experts.parameters())
    per_token = expert_params // N_EXPERTS
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[C] PARAMS vs COMPUTE")
    print(f"    all {N_EXPERTS} experts = {expert_params:,} params, but top-1 "
          f"routing runs only 1 = {per_token:,} params/token.")

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed:.1f}s")

    out = {
        "meta": gd.config(), "n_experts": N_EXPERTS,
        "specialization": {"accuracy": acc, "hist": hist.tolist(),
                           "usage": u_spec.tolist(),
                           "peak_per_type": [float(p) for p in peak_per_type]},
        "load_balancing": {
            "aux_off": {"accuracy": acc_off, "usage": u_off.tolist(),
                        "max_load": float(u_off.max())},
            "aux_on": {"accuracy": acc_on, "usage": u_on.tolist(),
                       "max_load": float(u_on.max())},
        },
        "params": {"expert_params_total": int(expert_params),
                   "per_token_ffn_params": int(per_token),
                   "total_params": int(total)},
    }
    (DATA_DIR / "switch_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {DATA_DIR / 'switch_results.json'}")

    # Evidence.
    assert acc > 0.9, "model failed to learn the typed task"
    assert min(peak_per_type) > 0.8, "experts did not specialize by type"
    assert u_on.max() < u_off.max() - 1e-6, "aux loss did not improve balance"
    assert u_on.max() < 0.45, "usage not balanced enough with aux loss"
    print("\nOK: experts specialize by input type; the load-balancing loss "
          "spreads tokens evenly across experts.")


if __name__ == "__main__":
    main()
