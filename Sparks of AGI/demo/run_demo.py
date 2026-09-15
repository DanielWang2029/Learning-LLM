"""Sparks of AGI — reproducing the *capability-probe methodology* on CPU.

"Sparks of Artificial General Intelligence" (Bubeck et al., 2023) is an
*analysis* paper: it defines no new model or algorithm. Its contribution is a
methodology — probe a model with a broad battery of self-contained tasks and
report a capability profile, then observe how that profile changes with scale.

We reproduce exactly that method, honestly and transparently, at tiny scale:

  1. Define a battery of six probes spanning a difficulty range (copy, reverse,
     pattern, sort, add-with-carry, parity).
  2. Train a family of small GPTs of increasing size on the whole battery.
  3. Report a capability MATRIX: which probes each model passes, and how the
     profile fills in as scale grows.

Runs on CPU in well under a minute. Writes data/capability_profile.json.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared box: avoid CPU oversubscription

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

import numpy as np  # noqa: E402

from src.model import GPT, GPTConfig  # noqa: E402
from src.probe import PASS_THRESHOLD, capability_profile  # noqa: E402
from src.tasks import (SEQ_LEN, TASK_DESC, TASKS, VOCAB_SIZE,  # noqa: E402
                       chance_accuracy, make_batch, make_task_eval)

DATA_DIR = PAPER_DIR / "data"

FAMILY = [
    ("nano", dict(n_layer=1, n_embd=16, n_head=2)),
    ("micro", dict(n_layer=2, n_embd=32, n_head=4)),
    ("mini", dict(n_layer=3, n_embd=48, n_head=4)),
    ("small", dict(n_layer=3, n_embd=64, n_head=8)),
]
STEPS, N_PER_TASK, LR = 600, 20, 3e-3


def train_model(cfg: GPTConfig, seed: int):
    torch.manual_seed(seed)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        seqs, masks = make_batch(TASKS, N_PER_TASK, seed=1000 + step)
        x = torch.tensor(seqs, dtype=torch.long)
        m = torch.tensor(masks[:, 1:], dtype=torch.bool)
        logits = model(x[:, :-1])
        tgt = x[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE)[m.reshape(-1)],
            tgt.reshape(-1)[m.reshape(-1)],
        )
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> None:
    torch.manual_seed(0)
    eval_sets = {t: make_task_eval(t, 200, seed=7) for t in TASKS}

    print("=" * 74)
    print("Sparks of AGI — capability profile vs. scale (tiny CPU reproduction)")
    print("=" * 74)
    print("Analysis paper: no new algorithm — we reproduce the PROBING METHODOLOGY.\n")
    print("Probe battery:")
    for t in TASKS:
        print(f"  - {t:<8} {TASK_DESC[t]}   (chance {chance_accuracy(t)*100:.2f}%)")
    print(f"\nPass threshold: exact-match >= {PASS_THRESHOLD*100:.0f}%\n")

    scales, profiles = [], []
    t0 = time.time()
    for i, (name, arch) in enumerate(FAMILY):
        cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=SEQ_LEN, **arch)
        model = train_model(cfg, seed=100 + i)
        prof = capability_profile(model, eval_sets)
        scales.append({"name": name, "N": model.num_params()})
        profiles.append(prof)

    elapsed = time.time() - t0

    # --- Print the capability matrix (rows = tasks, cols = scales).
    header = f"{'probe':<9}" + "".join(f"{s['name']:>9}" for s in scales)
    print(header)
    print("-" * len(header))
    for t in TASKS:
        row = f"{t:<9}"
        for prof in profiles:
            acc = prof[t]["acc"]
            cell = "✓" if prof[t]["pass"] else "·"
            row += f"{cell}{acc*100:>6.0f}% "
        print(row)
    print("\n(✓ = capability present, · = absent;  number = exact-match accuracy)")

    passes = [sum(p[t]["pass"] for t in TASKS) for p in profiles]
    print("\nCapabilities present: " +
          " -> ".join(f"{s['name']}:{c}/{len(TASKS)}" for s, c in zip(scales, passes)))
    print(f"Params: " +
          " -> ".join(f"{s['name']}:{s['N']:,}" for s in scales))
    print(f"Total train+probe time: {elapsed:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "tasks": TASKS,
        "task_desc": TASK_DESC,
        "chance": {t: chance_accuracy(t) for t in TASKS},
        "pass_threshold": PASS_THRESHOLD,
        "scales": scales,
        "matrix": [
            {"scale": scales[i]["name"], "N": scales[i]["N"],
             "profile": {t: profiles[i][t] for t in TASKS},
             "n_pass": passes[i]}
            for i in range(len(scales))
        ],
    }
    (DATA_DIR / "capability_profile.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DATA_DIR / 'capability_profile.json'}")

    if not (passes[-1] > passes[0]):
        raise SystemExit(
            f"Capabilities did not grow with scale ({passes[0]} -> {passes[-1]})."
        )
    print("OK: the capability profile fills in as the model scales.")


if __name__ == "__main__":
    main()
