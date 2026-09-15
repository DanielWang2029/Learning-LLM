"""Generate a tiny (Q, K, V) sample for the FlashAttention demo.

Attention does not need a trained model to demonstrate the tiling result: the
online-softmax identity is exact for *any* inputs. We save a small, seeded set
of query/key/value vectors so the equivalence check is reproducible.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent

N = 8   # sequence length
D = 4   # head dimension
SEED = 0


def main() -> None:
    rng = np.random.default_rng(SEED)
    Q = rng.standard_normal((N, D))
    K = rng.standard_normal((N, D))
    V = rng.standard_normal((N, D))
    payload = {
        "meta": {"N": N, "d": D, "seed": SEED},
        "Q": Q.tolist(), "K": K.tolist(), "V": V.tolist(),
    }
    out = DATA_DIR / "sample_qkv.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote sample Q,K,V of shape ({N},{D}) -> {out}")


if __name__ == "__main__":
    main()
