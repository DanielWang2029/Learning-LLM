"""Scaling laws demo — train a series of tiny LMs and fit L(N) = (Nc/N)^alpha.

We train several decoder-only language models that differ only in *size*
(width ``d_model`` and depth ``n_layer``) on the *same* synthetic corpus, record
each model's final loss, and then fit a power law to the (N, loss) points.

This reproduces the central empirical finding of Kaplan et al. (2020): the loss
falls off as a smooth power law in the non-embedding parameter count N, i.e. it
is a straight line in log-log space (paper §3, Fig. 1).

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

from src.scaling import fit_power_law
from src.tiny_lm import GPTConfig, TinyGPT

DATA_DIR = ROOT / "data"

# Model sizes, chosen to span ~90x in non-embedding parameter count N while
# staying tiny enough to train (to near-convergence) on CPU in a fixed step
# budget. Each entry is (n_layer, n_head, d_model).
MODEL_SIZES = [
    (1, 2, 12),
    (1, 2, 16),
    (1, 2, 24),
    (2, 2, 32),
    (2, 4, 48),
    (3, 4, 64),
]

STEPS = 600
WARMUP = 60
BATCH_SIZE = 64
LR = 4e-3
SEED = 0


def load_corpus():
    path = DATA_DIR / "corpus.json"
    if not path.exists():
        # Generate on the fly so the demo is self-contained.
        sys.path.insert(0, str(DATA_DIR))
        import generate_data

        generate_data.main()
    payload = json.loads(path.read_text())
    tokens = torch.tensor(payload["tokens"], dtype=torch.long)
    return tokens, payload["meta"]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, rng: torch.Generator):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=rng)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, block_size, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, BATCH_SIZE, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def lr_at(step: int, base_lr: float) -> float:
    """Linear warmup then cosine decay — stabilizes the larger models."""
    if step < WARMUP:
        return base_lr * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))


def train_one(cfg: GPTConfig, train_data, val_data, block_size):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    model = TinyGPT(cfg)
    # Wider models train more stably at a slightly smaller learning rate.
    base_lr = LR * (48.0 / cfg.d_model) ** 0.5
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    model.train()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, base_lr)
        x, y = get_batch(train_data, block_size, BATCH_SIZE, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val = eval_loss(model, val_data, block_size, rng)
    return model.num_params(non_embedding=True), val


def main() -> None:
    tokens, meta = load_corpus()
    block_size = meta["block_size"]

    n_val = len(tokens) // 5
    train_data = tokens[:-n_val]
    val_data = tokens[-n_val:]

    print(f"Corpus: {len(tokens)} tokens | vocab={meta['vocab_size']} "
          f"order={meta['order']} noise={meta['noise']}")
    print(f"Training {len(MODEL_SIZES)} models of increasing size "
          f"({STEPS} steps each)\n")
    print(f"{'N (non-emb)':>12} | {'layers':>6} | {'d_model':>7} | {'val loss':>9}")
    print("-" * 46)

    Ns, losses, records = [], [], []
    start = time.time()
    for (n_layer, n_head, d_model) in MODEL_SIZES:
        cfg = GPTConfig(
            vocab_size=meta["vocab_size"],
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            d_model=d_model,
        )
        N, val = train_one(cfg, train_data, val_data, block_size)
        Ns.append(N)
        losses.append(val)
        records.append({"N": N, "n_layer": n_layer, "d_model": d_model, "loss": val})
        print(f"{N:>12,} | {n_layer:>6} | {d_model:>7} | {val:>9.4f}")

    Ns = np.array(Ns, dtype=float)
    losses = np.array(losses, dtype=float)
    fit = fit_power_law(Ns, losses)
    elapsed = time.time() - start

    print("-" * 46)
    print(f"\nFitted power law  L(N) = (Nc / N) ^ alpha_N")
    print(f"  alpha_N (exponent) = {fit.alpha:.4f}")
    print(f"  Nc                 = {fit.Nc:,.1f}")
    print(f"  log-log R^2        = {fit.r2:.4f}")
    print(f"\nTrained {len(MODEL_SIZES)} models in {elapsed:.1f}s")

    # A dense curve for the visualization.
    grid = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
    fit_curve = fit.predict(grid)

    out = {
        "points": records,
        "fit": {"alpha": fit.alpha, "Nc": fit.Nc, "r2": fit.r2},
        "curve": {"N": grid.tolist(), "loss": fit_curve.tolist()},
        "meta": meta,
        "steps": STEPS,
    }
    (DATA_DIR / "scaling_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {DATA_DIR / 'scaling_results.json'}")

    # Evidence: the loss must fall as N grows, and the fit must be tight.
    assert losses[-1] < losses[0], "loss did not decrease with model size!"
    assert fit.alpha > 0, "expected a positive scaling exponent"
    assert fit.r2 > 0.9, f"power-law fit is poor (R^2={fit.r2:.3f})"
    print("\nOK: loss follows a clean power law in model size (straight line in log-log).")


if __name__ == "__main__":
    main()
