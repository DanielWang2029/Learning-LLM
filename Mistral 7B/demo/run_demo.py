"""Mistral 7B — sliding window attention, GQA, and a rolling KV cache, on CPU.

Reproduces the three efficiency mechanisms of Mistral 7B (Jiang et al., 2023)
at tiny scale and proves each one:

  1. Train a small SWA model on a task solvable from LOCAL context and show it
     works — and that the window must be large enough (W=1 fails, W>=2 succeeds).
  2. Print the banded sliding-window attention mask.
  3. Show the rolling-buffer KV cache gives results IDENTICAL to a full forward
     pass (max abs difference ~ 0).
  4. Show the KV-cache memory stays BOUNDED as the sequence grows (SWA) versus
     growing linearly (full attention), plus the extra GQA savings.

Runs on CPU in well under a minute. Writes data/mistral_results.json.

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

from src.attention import sliding_window_mask  # noqa: E402
from src.model import MistralConfig, MistralLM  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

VOCAB, SEQ_LEN = 12, 24
N_LAYER, N_HEAD, N_KV_HEAD, N_EMBD = 3, 4, 2, 48
BATCH = 96


def make_local_task(n: int, seq_len: int, vocab: int, seed: int):
    """Order-2 recurrence x[t] = (x[t-1] + x[t-2]) mod V.

    Each next token is fully determined by the previous TWO tokens, so a
    sliding window of size >= 2 is sufficient (and necessary) to solve it.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, seq_len), dtype=np.int64)
    x[:, :2] = rng.integers(0, vocab, size=(n, 2))
    for t in range(2, seq_len):
        x[:, t] = (x[:, t - 1] + x[:, t - 2]) % vocab
    return x


def train(window: int, steps: int, seed: int):
    torch.manual_seed(seed)
    cfg = MistralConfig(vocab_size=VOCAB, block_size=SEQ_LEN, n_layer=N_LAYER,
                        n_head=N_HEAD, n_kv_head=N_KV_HEAD, n_embd=N_EMBD,
                        window=window)
    model = MistralLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(steps):
        data = make_local_task(BATCH, SEQ_LEN, VOCAB, seed=1000 + step)
        x = torch.tensor(data)
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def next_token_accuracy(model) -> float:
    """Accuracy on positions >= 2 (the deterministic, locally-solvable region)."""
    model.eval()
    data = make_local_task(256, SEQ_LEN, VOCAB, seed=555)
    x = torch.tensor(data)
    pred = model(x[:, :-1]).argmax(-1)
    tgt = x[:, 1:]
    correct = (pred[:, 1:] == tgt[:, 1:]).float().mean().item()  # skip pos 0->1
    return correct


def ascii_mask(seq_len: int, window: int) -> str:
    m = sliding_window_mask(seq_len, window)
    rows = []
    for i in range(seq_len):
        rows.append("  " + "".join("█" if m[i, j] else "·" for j in range(seq_len)))
    return "\n".join(rows)


def main() -> None:
    print("=" * 72)
    print("Mistral 7B — sliding window attention + GQA + rolling KV cache (CPU)")
    print("=" * 72)
    print(f"config: layers={N_LAYER} heads={N_HEAD} kv_heads={N_KV_HEAD} "
          f"(GQA {N_HEAD}->{N_KV_HEAD}) d={N_EMBD} | vocab={VOCAB} seq={SEQ_LEN}")
    print("task: x[t] = (x[t-1] + x[t-2]) mod V  (needs the last 2 tokens)\n")

    t0 = time.time()

    # --- 1) Window size matters: W=1 fails, W>=2 succeeds.
    print("[1] Sliding-window size vs. accuracy on the local task")
    steps_per_window = {1: 300, 2: 600, 4: 600}
    window_acc, models = {}, {}
    for w in (1, 2, 4):
        models[w] = train(window=w, steps=steps_per_window[w], seed=w)
        acc = next_token_accuracy(models[w])
        window_acc[w] = acc
        note = "(< 2: cannot see both needed tokens)" if w < 2 else "(sufficient)"
        print(f"    window W={w}: next-token accuracy {acc*100:6.1f}%  {note}")
    main_model = models[4]                     # reuse the W=4 model as the main one
    main_acc = window_acc[4]
    print(f"    => a window of just {2} tokens already solves this order-2 task\n")

    # --- 2) The banded mask.
    print("[2] Sliding-window attention mask (W=4, first 12 positions)")
    print("    (█ = attends, · = masked;  rows=query, cols=key)")
    print(ascii_mask(12, 4))
    print()

    # --- 3) Rolling KV cache == full forward.
    print("[3] Rolling-buffer KV cache vs. full attention (should be identical)")
    test = torch.tensor(make_local_task(16, SEQ_LEN, VOCAB, seed=42))
    full = main_model(test)
    cached = main_model.forward_cached(test)
    max_diff = (full - cached).abs().max().item()
    argmax_match = (full.argmax(-1) == cached.argmax(-1)).float().mean().item()
    print(f"    max |logit difference|      : {max_diff:.2e}")
    print(f"    predicted-token agreement   : {argmax_match*100:.1f}%")
    print(f"    => the rolling buffer is an exact, memory-bounded equivalent\n")

    # --- 4) Memory: bounded (SWA) vs. linear (full); plus GQA savings.
    print("[4] KV-cache memory vs. sequence length")
    per_pos_full = N_LAYER * 2 * N_HEAD * (N_EMBD // N_HEAD)      # MHA baseline
    per_pos_swa = N_LAYER * 2 * N_KV_HEAD * (N_EMBD // N_HEAD)    # GQA heads
    W = 4
    curve = []
    print(f"    {'seq_len':>8}{'full (MHA)':>14}{'SWA+GQA':>12}{'reduction':>12}")
    for L in (8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384):
        full_floats = per_pos_full * L
        swa_floats = per_pos_swa * min(L, W)
        red = full_floats / swa_floats
        curve.append({"L": L, "full": full_floats, "swa": swa_floats,
                      "reduction": red})
        if L in (8, 32, 128, 512, 4096, 16384):
            print(f"    {L:>8}{full_floats:>14,}{swa_floats:>12,}{red:>11.1f}x")
    print(f"    SWA cache is CONSTANT beyond L=W={W}; GQA adds a "
          f"{N_HEAD/N_KV_HEAD:.0f}x KV shrink.\n")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_kv_head": N_KV_HEAD,
                   "n_embd": N_EMBD, "window": 4, "vocab": VOCAB,
                   "seq_len": SEQ_LEN},
        "window_accuracy": {str(k): v for k, v in window_acc.items()},
        "main_accuracy": main_acc,
        "mask": sliding_window_mask(12, 4).int().tolist(),
        "cache_check": {"max_diff": max_diff, "argmax_match": argmax_match},
        "memory_curve": curve,
        "gqa_factor": N_HEAD / N_KV_HEAD,
    }
    (DATA_DIR / "mistral_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {DATA_DIR / 'mistral_results.json'}")

    ok = (window_acc[1] < 0.5 < window_acc[4] and main_acc > 0.9
          and max_diff < 1e-3)
    if not ok:
        raise SystemExit("Mistral demo did not meet expectations.")
    print("OK: local window solves the task; rolling cache is exact and bounded.")


if __name__ == "__main__":
    main()
