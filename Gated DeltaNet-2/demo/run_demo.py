"""End-to-end CPU demo for Gated DeltaNet-2 (arXiv:2605.22791).

Two parts, both under a minute on CPU:

  Part A  Mechanism check (no training). Run the raw Gated Delta Rule-2 state
          recurrence on hand-built one-hot keys. Overwrite one key, then read
          it back. The decoupled erase/write gates return the LATEST value
          cleanly, whereas plain additive linear attention (no erase) returns a
          scrambled sum of the old and new values.

  Part B  Learned associative recall with overwrite. Train two tiny models on
          the same task — one with DECOUPLED channel-wise erase/write gates
          (the paper),           one with a TIED scalar gate (the KDA-style baseline the
          paper argues against). Report recall accuracy, split by whether the
          queried key was overwritten. Decoupling the erase/write decisions
          gives higher overall recall from the same fixed-size state.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # keep CPU timing predictable

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import GatedDeltaNet2, additive_linear_attention  # noqa: E402
from src.task import RecallVocab, make_batch, make_eval_batch  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0


# --------------------------------------------------------------------------- #
# Part A: deterministic mechanism check
# --------------------------------------------------------------------------- #
def gated_delta_rule2_scan(ops, d):
    """Run the raw recurrence S_t = S̄ + k_t (z_t - r_t)ᵀ over a list of ops.

    Each op is a dict with 'k','v','b','w' (and optional 'alpha'); returns the
    final state S of shape (d, d).
    """
    S = torch.zeros(d, d)
    for op in ops:
        k, v = op["k"], op["v"]
        b, w = op["b"], op["w"]
        alpha = op.get("alpha", torch.ones(d))
        S = alpha.unsqueeze(-1) * S            # channel-wise decay
        e = b * k                              # gated erase direction
        z = w * v                              # gated write target
        r = S.t() @ e                          # read old content: S̄ᵀ e
        S = S + torch.outer(k, z - r)          # rank-1 delta write
    return S


def read(S, q):
    return S.t() @ q


def part_a():
    print("=" * 70)
    print("PART A  Mechanism check: overwrite one key, then read it back")
    print("=" * 70)
    d = 4
    e = torch.eye(d)                # one-hot keys k1=e0, k2=e1
    k1, k2 = e[0], e[1]
    v1 = torch.tensor([1.0, 0.0, 2.0, 0.0])   # original value for key 1
    v2 = torch.tensor([0.0, 3.0, 0.0, 1.0])   # value for key 2
    v1_new = torch.tensor([0.0, 0.0, 0.0, 5.0])  # overwrite value for key 1

    ones = torch.ones(d)
    ops = [
        {"k": k1, "v": v1, "b": ones, "w": ones},       # write (k1, v1)
        {"k": k2, "v": v2, "b": ones, "w": ones},       # write (k2, v2)
        {"k": k1, "v": v1_new, "b": ones, "w": ones},   # OVERWRITE k1 -> v1_new
    ]
    S = gated_delta_rule2_scan(ops, d)
    gd_read = read(S, k1)

    # Additive baseline (Eq. 1): no erase, just accumulate outer products.
    K = torch.stack([k1, k2, k1]).unsqueeze(0)
    V = torch.stack([v1, v2, v1_new]).unsqueeze(0)
    Q = k1.unsqueeze(0).unsqueeze(0)
    add_state = torch.zeros(d, d)
    for t in range(3):
        add_state = add_state + torch.outer(K[0, t], V[0, t])
    add_read = read(add_state, k1)

    print(f"  key 1 original value v1       : {v1.tolist()}")
    print(f"  key 1 overwrite value v1_new  : {v1_new.tolist()}")
    print(f"  Gated Delta Rule-2 read(k1)   : {[round(x, 3) for x in gd_read.tolist()]}"
          f"   <- matches v1_new (old value erased)")
    print(f"  additive linear attn read(k1) : {[round(x, 3) for x in add_read.tolist()]}"
          f"   <- v1 + v1_new (scrambled!)")

    gd_err = (gd_read - v1_new).norm().item()
    add_err = (add_read - v1_new).norm().item()
    print(f"  error vs latest value  |  gated-delta: {gd_err:.3f}   additive: {add_err:.3f}")
    assert gd_err < 1e-5 < add_err, "mechanism check failed"
    print("  OK: decoupled erase+write overwrites cleanly; additive interferes.\n")

    return {
        "v1": v1.tolist(),
        "v2": v2.tolist(),
        "v1_new": v1_new.tolist(),
        "gated_delta_read": [round(x, 3) for x in gd_read.tolist()],
        "additive_read": [round(x, 3) for x in add_read.tolist()],
        "gated_delta_error": round(gd_err, 4),
        "additive_error": round(add_err, 4),
    }


# --------------------------------------------------------------------------- #
# Part B: learned associative recall with overwrite
# --------------------------------------------------------------------------- #
def evaluate(model, vocab, rng, n_overwrite=3, batch=400):
    model.eval()
    tokens, targets, over = make_eval_batch(vocab, batch, n_overwrite, rng)
    with torch.no_grad():
        logits = model(tokens)
    pred = logits.argmax(-1)
    q_mask = targets != -100
    correct = (pred == targets) & q_mask
    overall = correct.sum().item() / q_mask.sum().item()
    ow = over & q_mask
    non = q_mask & ~over
    acc_ow = correct[ow].float().mean().item() if ow.any() else float("nan")
    acc_non = correct[non].float().mean().item() if non.any() else float("nan")
    return overall, acc_ow, acc_non


def train_model(decouple, vocab, steps, rng):
    torch.manual_seed(SEED)
    model = GatedDeltaNet2(vocab.size, d_model=64, d_key=32, d_val=32, decouple=decouple)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, steps + 1):
        model.train()
        tokens, targets = make_batch(vocab, 64, n_overwrite=3, rng=rng)
        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, vocab.size), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % (steps // 4) == 0 or step == 1:
            acc, _, _ = evaluate(model, vocab, torch.Generator().manual_seed(123))
            tag = "decoupled" if decouple else "tied-scalar"
            print(f"  [{tag:11s}] step {step:4d}/{steps} | loss {loss.item():.4f} | acc {acc*100:5.1f}%")
    return model


def part_b():
    print("=" * 70)
    print("PART B  Learned recall with overwrite: decoupled vs tied gates")
    print("=" * 70)
    vocab = RecallVocab(n_keys=6, n_vals=10)
    steps = 600
    train_rng = torch.Generator().manual_seed(SEED)

    print("Training DECOUPLED erase/write gates (Gated DeltaNet-2):")
    dec = train_model(True, vocab, steps, train_rng)
    print("Training TIED scalar gate (KDA-style baseline):")
    tied = train_model(False, vocab, steps, train_rng)

    eval_rng = torch.Generator().manual_seed(999)
    d_all, d_ow, d_non = evaluate(dec, vocab, eval_rng, batch=800)
    eval_rng = torch.Generator().manual_seed(999)
    t_all, t_ow, t_non = evaluate(tied, vocab, eval_rng, batch=800)

    print("\nRecall accuracy on held-out sequences (800 sequences):")
    print(f"  {'model':<24}{'overall':>10}{'overwritten':>14}{'not-overwritten':>18}")
    print(f"  {'Gated DeltaNet-2 (decoupled)':<24}{d_all*100:>9.1f}%{d_ow*100:>13.1f}%{d_non*100:>17.1f}%")
    print(f"  {'KDA-style (tied scalar)':<24}{t_all*100:>9.1f}%{t_ow*100:>13.1f}%{t_non*100:>17.1f}%")

    n_params = sum(p.numel() for p in dec.parameters())
    state_elems = dec.mixer.d_key * dec.mixer.d_val
    print(f"\nFixed-size recurrent state: {dec.mixer.d_key}x{dec.mixer.d_val}"
          f" = {state_elems} scalars (independent of sequence length -> O(L) time, O(1) memory).")
    print(f"Model parameters: {n_params:,}")

    return vocab, dec, tied, {
        "decoupled": {"overall": d_all, "overwritten": d_ow, "not_overwritten": d_non},
        "tied": {"overall": t_all, "overwritten": t_ow, "not_overwritten": t_non},
        "state_shape": [dec.mixer.d_key, dec.mixer.d_val],
        "num_parameters": n_params,
    }


def sample_trace(vocab, model):
    """Grab one sequence + the model's per-step read/write norms for the viz."""
    rng = torch.Generator().manual_seed(7)
    tokens, targets, over = make_eval_batch(vocab, 1, n_overwrite=3, rng=rng)
    model.eval()
    with torch.no_grad():
        x = model.norm(model.embed(tokens))
        _, trace = model.mixer(x, return_trace=True)
        logits = model(tokens)
    pred = logits.argmax(-1)[0].tolist()
    return {
        "tokens": tokens[0].tolist(),
        "targets": targets[0].tolist(),
        "prediction": pred,
        "overwritten_query": over[0].tolist(),
        "erase_read_norm": [round(v, 3) for v in trace["erase_read_norm"][0].tolist()],
        "write_norm": [round(v, 3) for v in trace["write_norm"][0].tolist()],
    }


def main() -> None:
    start = time.time()
    a = part_a()
    vocab, dec, tied, b = part_b()
    trace = sample_trace(vocab, dec)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (2026)",
        "arxiv": "2605.22791",
        "mechanism_check": a,
        "learned_recall": b,
        "sample_trace": trace,
        "vocab": {
            "n_keys": vocab.n_keys,
            "n_vals": vocab.n_vals,
            "size": vocab.size,
            "val_base": vocab.val_base,
            "wkey_base": vocab.wkey_base,
            "qkey_base": vocab.qkey_base,
        },
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/results.json  (elapsed {time.time()-start:.1f}s)")
    print("Open visualization/index.html to explore the erase/write mechanism.")

    d_ow = b["decoupled"]["overwritten"]
    t_ow = b["tied"]["overwritten"]
    if not (d_ow > 0.9 and d_ow > t_ow):
        raise SystemExit(
            f"Demo did not demonstrate the effect (decoupled overwrite acc {d_ow*100:.1f}%,"
            f" tied {t_ow*100:.1f}%)."
        )
    print("\nOK: decoupled erase/write recalls overwritten values; it beats the tied gate.")


if __name__ == "__main__":
    main()
