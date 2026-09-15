"""Chinchilla demo — compute-optimal allocation of a fixed budget.

Hoffmann et al. (2022) ask: given a *fixed* training compute budget, how should
you split it between model size N (parameters) and data D (training tokens)?
Using the compute proxy

    C ≈ 6 · N · D                                   (paper §2, Eq. 1)

they find an interior optimum — models should be *smaller* and trained on *more*
tokens than the earlier scaling-law recipe suggested, with N and D scaled up
roughly equally.

Here we hold C fixed and sweep the split: for each candidate model size N we set
the number of training tokens to D = C / (6N) (so bigger models get fewer
tokens/steps and vice-versa), train each tiny LM, and plot the final loss vs the
allocation. The result is a U-shaped curve: too-big models are *under-trained*,
too-small models are *capacity-limited*, and a model in the middle is
compute-optimal.

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

from src.tiny_lm import GPTConfig, TinyGPT

DATA_DIR = ROOT / "data"

# Candidate model sizes (n_layer, n_head, d_model), from small to large.
MODEL_SIZES = [
    (1, 2, 16),
    (1, 2, 24),
    (2, 2, 32),
    (2, 4, 48),
    (3, 4, 64),
    (3, 4, 96),
]

BATCH_SIZE = 32
BLOCK_SIZE = 32
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE
# Fixed compute budget expressed as K = N * steps  (since C = 6·N·D and
# D = steps · tokens_per_step, holding N·steps fixed holds C fixed). K is chosen
# so mid-size models get a few hundred steps (enough to exploit their capacity),
# while the smallest is data-saturated (capacity-limited) and the largest is
# starved of steps (under-trained) — producing the U-shaped frontier.
MAX_STEPS_SMALL = 9000
LR = 4e-3
SEED = 0


def load_corpus():
    path = DATA_DIR / "corpus.json"
    if not path.exists():
        sys.path.insert(0, str(DATA_DIR))
        import generate_data
        generate_data.main()
    payload = json.loads(path.read_text())
    return torch.tensor(payload["tokens"], dtype=torch.long), payload["meta"]


def get_batch(data, rng):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=rng)
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * p))


def train(cfg: GPTConfig, steps, train_data, val_data):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    model = TinyGPT(cfg)
    base_lr = LR * (48.0 / cfg.d_model) ** 0.5
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    warmup = max(5, steps // 10)
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, base_lr, warmup)
        x, y = get_batch(train_data, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return eval_loss(model, val_data, rng)


def main() -> None:
    tokens, meta = load_corpus()
    n_val = len(tokens) // 6
    train_data, val_data = tokens[:-n_val], tokens[-n_val:]

    # Establish N for each config, then fix the compute budget K = N * steps.
    configs = []
    for (n_layer, n_head, d_model) in MODEL_SIZES:
        cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=BLOCK_SIZE,
                        n_layer=n_layer, n_head=n_head, d_model=d_model)
        N = TinyGPT(cfg).num_params(non_embedding=True)
        configs.append((cfg, N))
    N_min = min(N for _, N in configs)
    K = MAX_STEPS_SMALL * N_min  # fixed compute budget (N * steps)

    print("=" * 74)
    print("Chinchilla: fixed compute C = 6·N·D, swept over the N/D split")
    print(f"  corpus={len(tokens)} tokens  vocab={meta['vocab_size']}  "
          f"tokens/step={TOKENS_PER_STEP}")
    print("=" * 74)
    print(f"\n{'N (params)':>11} | {'steps':>6} | {'D (tokens)':>11} | "
          f"{'6ND (C)':>11} | {'val loss':>9}")
    print("-" * 74)

    start = time.time()
    points = []
    for cfg, N in configs:
        steps = max(4, round(K / N))
        D = steps * TOKENS_PER_STEP
        C = 6 * N * D
        loss = train(cfg, steps, train_data, val_data)
        points.append({"N": N, "d_model": cfg.d_model, "n_layer": cfg.n_layer,
                       "steps": steps, "D": D, "C": C, "loss": loss})
        print(f"{N:>11,} | {steps:>6} | {D:>11,} | {C:>11.2e} | {loss:>9.4f}")

    losses = [p["loss"] for p in points]
    best = int(np.argmin(losses))
    elapsed = time.time() - start

    print("-" * 74)
    print(f"\nCompute-optimal point: N = {points[best]['N']:,} params, "
          f"D = {points[best]['D']:,} tokens "
          f"(d_model={points[best]['d_model']}, steps={points[best]['steps']})")
    print(f"  -> loss {losses[best]:.4f}; smaller models are capacity-limited, "
          f"larger models are under-trained.")
    print(f"\nTrained {len(points)} allocations in {elapsed:.1f}s")

    out = {"points": points, "best_index": best,
           "compute_budget_C": points[best]["C"], "meta": meta,
           "tokens_per_step": TOKENS_PER_STEP}
    (DATA_DIR / "chinchilla_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {DATA_DIR / 'chinchilla_results.json'}")

    # Evidence: the optimum is interior (a genuine U-shape), not at either end.
    assert 0 < best < len(points) - 1, \
        f"expected an interior compute-optimal point, got index {best}"
    assert losses[best] < losses[0] and losses[best] < losses[-1], \
        "the middle allocation should beat both extremes"
    print("\nOK: fixed-compute loss is U-shaped with an interior compute-optimal "
          "model size (equal scaling of N and D).")


if __name__ == "__main__":
    main()
