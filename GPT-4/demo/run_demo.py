"""GPT-4 Technical Report — reproducing *predictable scaling* on a laptop CPU.

GPT-4 is closed: no weights, architecture, or data are public. The one
methodological claim in the report that we *can* reproduce is **predictable
scaling** — OpenAI fit a scaling law on much smaller models and predicted
GPT-4's final loss before training it. We do the same at tiny scale:

  1. Train a family of small GPTs of increasing size on a fixed synthetic
     language, recording each one's validation loss.
  2. Fit a power law  L(N) = a * N^-alpha + E  on the *smaller* models only.
  3. PREDICT the loss of the largest, held-out model from that fit, then train
     it and check the prediction.
  4. Track a multiple-choice capability score at every size to show that
     capability (not just loss) improves predictably with scale.

Runs in well under a minute on CPU. Writes data/scaling_results.json for the
visualization.

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

from src.data import MarkovSource  # noqa: E402
from src.eval import build_mc_questions, evaluate_mc  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.scaling import fit_power_law  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

# A family of models of increasing size. The last one is HELD OUT of the fit.
FAMILY = [
    ("XS", dict(n_layer=1, n_embd=8, n_head=2)),
    ("S", dict(n_layer=1, n_embd=16, n_head=2)),
    ("M", dict(n_layer=2, n_embd=24, n_head=3)),
    ("L", dict(n_layer=2, n_embd=32, n_head=4)),
    ("XL (held out)", dict(n_layer=3, n_embd=48, n_head=6)),
]

VOCAB, ORDER, BLOCK = 24, 2, 32
TRAIN_STEPS, BATCH, LR = 500, 64, 3e-3


def get_batch(data: np.ndarray, batch: int, block: int, rng: np.random.Generator):
    rows = rng.integers(0, data.shape[0], size=batch)
    starts = rng.integers(0, data.shape[1] - block - 1, size=batch)
    x = np.stack([data[r, s : s + block] for r, s in zip(rows, starts)])
    y = np.stack([data[r, s + 1 : s + block + 1] for r, s in zip(rows, starts)])
    return torch.from_numpy(x), torch.from_numpy(y)


def train_one(cfg: GPTConfig, train_np, val_x, val_y, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(TRAIN_STEPS):
        x, y = get_batch(train_np, BATCH, cfg.block_size, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, val_loss = model(val_x, val_y)
    return model, float(val_loss)


def main() -> None:
    torch.manual_seed(0)
    source = MarkovSource(vocab_size=VOCAB, order=ORDER, seed=0)
    floor = source.entropy_floor()

    train_np = source.sample(256, BLOCK * 4, seed=1)
    val_np = source.sample(64, BLOCK + 1, seed=2)
    val_x = torch.from_numpy(val_np[:, :BLOCK])
    val_y = torch.from_numpy(val_np[:, 1 : BLOCK + 1])

    mc = build_mc_questions(
        source, n_questions=120, prompt_len=6, completion_len=6, n_choices=4, seed=3
    )
    chance = 1.0 / 4

    print("=" * 70)
    print("GPT-4 Technical Report — predictable scaling (tiny CPU reproduction)")
    print("=" * 70)
    print(f"Synthetic order-{ORDER} Markov language | vocab={VOCAB} block={BLOCK}")
    print(f"Irreducible loss floor E* (source entropy): {floor:.4f} nats\n")
    print(f"{'model':<14}{'params(N)':>12}{'val loss':>12}{'MC acc':>10}")
    print("-" * 48)

    results = []
    t0 = time.time()
    for i, (name, arch) in enumerate(FAMILY):
        cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, **arch)
        model, val_loss = train_one(cfg, train_np, val_x, val_y, seed=100 + i)
        acc = evaluate_mc(model, *mc, device="cpu")
        n = model.num_params()
        results.append({"name": name, "N": n, "loss": val_loss, "mc_acc": acc})
        print(f"{name:<14}{n:>12,}{val_loss:>12.4f}{acc*100:>9.1f}%")
    elapsed = time.time() - t0

    # --- Scaling-law fit on the SMALLER models; predict the held-out largest.
    fit_pts = results[:-1]
    held = results[-1]
    N_fit = np.array([r["N"] for r in fit_pts])
    L_fit = np.array([r["loss"] for r in fit_pts])
    law = fit_power_law(N_fit, L_fit, E_floor=floor)

    predicted = float(law.predict(held["N"]))
    actual = held["loss"]
    rel_err = abs(predicted - actual) / actual * 100

    print("\n" + "-" * 70)
    print("SCALING LAW  L(N) = a * N^(-alpha) + E   fit on the 4 smaller models:")
    print(f"  a = {law.a:.4g}   alpha = {law.alpha:.4f}   E = {law.E:.4f}")
    print("\nExtrapolation to the held-out XL model:")
    print(f"  predicted loss : {predicted:.4f}")
    print(f"  actual   loss : {actual:.4f}")
    print(f"  relative error: {rel_err:.2f}%")
    print(f"\nCapability check: MC accuracy rose "
          f"{results[0]['mc_acc']*100:.1f}% -> {results[-1]['mc_acc']*100:.1f}% "
          f"(chance = {chance*100:.0f}%)")
    print(f"Total train+eval time: {elapsed:.1f}s")

    # Smooth curve for the visualization.
    N_grid = np.logspace(
        np.log10(min(r["N"] for r in results)) - 0.15,
        np.log10(max(r["N"] for r in results)) + 0.15,
        60,
    )
    curve = [{"N": float(n), "loss": float(law.predict(n))} for n in N_grid]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source": {"vocab": VOCAB, "order": ORDER, "block": BLOCK,
                   "entropy_floor": floor},
        "models": results,
        "fit": {"a": law.a, "alpha": law.alpha, "E": law.E},
        "held_out": {"N": held["N"], "predicted": predicted, "actual": actual,
                     "rel_err_pct": rel_err},
        "curve": curve,
        "chance": chance,
    }
    (DATA_DIR / "scaling_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {DATA_DIR / 'scaling_results.json'}")

    ok = rel_err < 12.0 and results[-1]["mc_acc"] > results[0]["mc_acc"]
    if not ok:
        raise SystemExit(
            f"Scaling prediction not convincing (rel err {rel_err:.1f}% "
            f"or capability did not improve)."
        )
    print("OK: the loss of a larger model was predicted before training it.")


if __name__ == "__main__":
    main()
