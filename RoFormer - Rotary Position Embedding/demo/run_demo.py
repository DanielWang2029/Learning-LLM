"""RoPE demo — relative-position property + a relative-position learning task.

Part (a): numerically verify RoPE's defining property (paper §3.4.3): the inner
product <R(m) q, R(n) k> depends only on the offset (m - n). We build a matrix
over positions (m, n) and show it is (nearly) constant along each diagonal
m - n = const.

Part (b): train a tiny one-layer attention model on "predict the token k
positions earlier" with RoPE vs. with no positional encoding, then evaluate at
sequence lengths *longer than seen in training* to show RoPE extrapolates while
the no-position model is stuck at chance.

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

from src.model import RoPEAttentionLM
from src.rope import apply_rope, rope_cache

DATA_DIR = ROOT / "data"

VOCAB = 20
SHIFT_K = 3
D_MODEL = 64
N_HEAD = 2
TRAIN_LEN = 16
EXTRA_LENS = [16, 24, 32, 48]  # evaluate here; >16 tests extrapolation
STEPS = 600
BATCH = 64
SEED = 0


# ----------------------------------------------------------------------------
# Part (a): relative-position property
# ----------------------------------------------------------------------------
def relative_position_check(dim=16, max_pos=8, base=10000.0):
    """Return the matrix S[m, n] = <R(m) q, R(n) k> for fixed random q, k."""
    torch.manual_seed(SEED)
    q = torch.randn(dim)
    k = torch.randn(dim)
    cos, sin = rope_cache(max_pos, dim, base)
    qr = apply_rope(q.expand(max_pos, dim), cos, sin)  # rotate q to each pos m
    kr = apply_rope(k.expand(max_pos, dim), cos, sin)  # rotate k to each pos n
    S = qr @ kr.T  # (max_pos, max_pos), S[m,n] = <R(m)q, R(n)k>
    return S.detach().numpy()


def summarize_diagonals(S: np.ndarray):
    """For each offset d = m - n, report mean and spread of S along that diagonal."""
    n = S.shape[0]
    rows = []
    for d in range(-(n - 1), n):
        vals = np.array([S[m, m - d] for m in range(n) if 0 <= m - d < n])
        rows.append({"offset": int(d), "mean": float(vals.mean()),
                     "std": float(vals.std()), "count": int(vals.size)})
    return rows


# ----------------------------------------------------------------------------
# Part (b): learn a relative-position task, test extrapolation
# ----------------------------------------------------------------------------
def make_batch(n, seq_len, rng):
    x = torch.randint(0, VOCAB, (n, seq_len), generator=rng)
    y = torch.full_like(x, -100)
    y[:, SHIFT_K:] = x[:, : seq_len - SHIFT_K]
    return x, y


@torch.no_grad()
def accuracy(model, seq_len, rng, iters=10):
    model.eval()
    correct = total = 0
    for _ in range(iters):
        x, y = make_batch(256, seq_len, rng)
        logits, _ = model(x)
        pred = logits.argmax(-1)
        mask = y != -100
        correct += (pred[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    return correct / total


def train(pos: str):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    model = RoPEAttentionLM(VOCAB, D_MODEL, N_HEAD, pos=pos)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(BATCH, TRAIN_LEN, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    accs = {L: accuracy(model, L, rng) for L in EXTRA_LENS}
    return accs


def main() -> None:
    start = time.time()

    # ---- Part (a)
    print("=" * 64)
    print("Part (a): relative-position property  <R(m)q, R(n)k> = f(m - n)")
    print("=" * 64)
    S = relative_position_check(dim=16, max_pos=8)
    diag = summarize_diagonals(S)
    print("\nInner product matrix S[m, n]  (rows m = 0..7, cols n = 0..7):")
    for m in range(S.shape[0]):
        print("  " + " ".join(f"{S[m, n]:6.2f}" for n in range(S.shape[1])))
    print("\nAlong each diagonal m - n = const, the value is constant:")
    print(f"  {'offset (m-n)':>12} | {'mean <.,.>':>11} | {'std (spread)':>12}")
    print("  " + "-" * 40)
    max_std = 0.0
    for r in diag:
        print(f"  {r['offset']:>12} | {r['mean']:>11.4f} | {r['std']:>12.2e}")
        max_std = max(max_std, r["std"])
    print(f"\n  Max spread along any diagonal: {max_std:.2e}  (≈0 confirms the "
          f"score depends only on m - n)")

    # ---- Part (b)
    print("\n" + "=" * 64)
    print(f"Part (b): learn 'predict token {SHIFT_K} positions earlier'")
    print(f"          train at length {TRAIN_LEN}, test at {EXTRA_LENS}")
    print("=" * 64)
    rope_acc = train("rope")
    none_acc = train("none")

    print(f"\n  {'seq len':>8} | {'RoPE acc':>9} | {'No-pos acc':>11} | note")
    print("  " + "-" * 52)
    for L in EXTRA_LENS:
        note = "train len" if L == TRAIN_LEN else "extrapolation"
        print(f"  {L:>8} | {rope_acc[L]*100:>8.1f}% | {none_acc[L]*100:>10.1f}% | {note}")

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed:.1f}s")

    out = {
        "relative_property": {
            "matrix": S.tolist(),
            "diagonals": diag,
            "max_diag_std": max_std,
            "dim": 16,
        },
        "task": {
            "vocab": VOCAB, "shift_k": SHIFT_K, "train_len": TRAIN_LEN,
            "eval_lens": EXTRA_LENS,
            "rope_acc": {str(L): rope_acc[L] for L in EXTRA_LENS},
            "none_acc": {str(L): none_acc[L] for L in EXTRA_LENS},
        },
    }
    (DATA_DIR / "rope_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {DATA_DIR / 'rope_results.json'}")

    # Evidence checks.
    assert max_std < 1e-3, "relative-position property violated!"
    assert rope_acc[TRAIN_LEN] > 0.9, "RoPE failed to learn the in-distribution task"
    assert rope_acc[max(EXTRA_LENS)] > 0.8, "RoPE failed to extrapolate"
    assert none_acc[TRAIN_LEN] < 0.5, "no-position model unexpectedly solved the task"
    print("\nOK: RoPE score depends only on (m-n); RoPE learns and extrapolates, "
          "no-position attention stays at chance.")


if __name__ == "__main__":
    main()
