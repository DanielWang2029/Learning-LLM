"""FlashAttention demo — exactness + memory advantage.

Part (a): show the tiled/online-softmax output is numerically identical to
naive full attention (max abs difference ~ 1e-6, i.e. floating-point noise).

Part (b): sweep the sequence length N and compare the peak score-matrix memory
and wall-clock time. Naive attention allocates the full N x N score matrix
(O(N²) memory); FlashAttention only ever holds a single block_q x block_k tile
(O(block²) memory), so its peak score memory is flat in N.

Runs on CPU in well under a minute.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.flash_attention import flash_attention, naive_attention

DATA_DIR = ROOT / "data"

SEED = 0
D = 32
BLOCK = 32
SEQ_LENS = [64, 128, 256, 512, 1024]
BYTES_PER_ELEM = 4  # float32 score matrix


def bytes_to_str(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> None:
    torch.manual_seed(SEED)

    # ---- Part (a): numerical equivalence
    print("=" * 66)
    print("Part (a): tiled online-softmax output == naive full attention")
    print("=" * 66)
    N = 128
    Q = torch.randn(N, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    out_naive, _ = naive_attention(Q, K, V)
    out_flash, _ = flash_attention(Q, K, V, block_q=BLOCK, block_k=BLOCK)
    max_diff = (out_naive - out_flash).abs().max().item()
    print(f"\n  sequence length N = {N}, head dim d = {D}, block = {BLOCK}")
    print(f"  max |naive - flash| = {max_diff:.2e}  (floating-point noise)")

    # Also check the causal variant, which the online softmax handles too.
    out_naive_c, _ = naive_attention(Q, K, V, causal=True)
    out_flash_c, _ = flash_attention(Q, K, V, block_q=BLOCK, block_k=BLOCK, causal=True)
    max_diff_c = (out_naive_c - out_flash_c).abs().max().item()
    print(f"  max |naive - flash| (causal) = {max_diff_c:.2e}")

    # ---- Part (b): memory & time sweep
    print("\n" + "=" * 66)
    print("Part (b): peak score-matrix memory & wall-clock vs sequence length")
    print("=" * 66)
    print(f"\n  {'N':>6} | {'naive mem':>11} | {'flash mem':>11} | {'ratio':>8} | "
          f"{'naive s':>8} | {'flash s':>8}")
    print("  " + "-" * 66)

    rows = []
    for N in SEQ_LENS:
        Q = torch.randn(N, D)
        K = torch.randn(N, D)
        V = torch.randn(N, D)

        t0 = time.time()
        _, naive_elems = naive_attention(Q, K, V)
        t_naive = time.time() - t0

        t0 = time.time()
        out_f, flash_elems = flash_attention(Q, K, V, block_q=BLOCK, block_k=BLOCK)
        t_flash = time.time() - t0

        naive_bytes = naive_elems * BYTES_PER_ELEM
        flash_bytes = flash_elems * BYTES_PER_ELEM
        ratio = naive_bytes / flash_bytes
        print(f"  {N:>6} | {bytes_to_str(naive_bytes):>11} | "
              f"{bytes_to_str(flash_bytes):>11} | {ratio:>7.0f}x | "
              f"{t_naive:>8.3f} | {t_flash:>8.3f}")
        rows.append({
            "N": N, "naive_bytes": naive_bytes, "flash_bytes": flash_bytes,
            "ratio": ratio, "naive_time": t_naive, "flash_time": t_flash,
        })

    print("\n  Naive peak memory grows as O(N²); FlashAttention stays flat at "
          f"one {BLOCK}x{BLOCK} tile.")

    out = {
        "equivalence": {"N": 128, "d": D, "block": BLOCK,
                        "max_abs_diff": max_diff, "max_abs_diff_causal": max_diff_c},
        "sweep": rows,
        "block": BLOCK,
        "bytes_per_elem": BYTES_PER_ELEM,
    }
    (DATA_DIR / "flash_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DATA_DIR / 'flash_results.json'}")

    # Evidence checks.
    assert max_diff < 1e-4, "tiled output does not match naive attention!"
    assert max_diff_c < 1e-4, "causal tiled output does not match!"
    assert rows[-1]["ratio"] > 10, "expected large memory savings at long N"
    print("\nOK: tiled attention is numerically identical and uses far less memory.")


if __name__ == "__main__":
    main()
