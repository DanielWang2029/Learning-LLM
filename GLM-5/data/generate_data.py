"""Generate a tiny synthetic sequence whose attention is genuinely sparse.

DSA only helps when dense attention actually concentrates on a few keys. We
plant that structure: tokens belong to a handful of latent *groups*, and a
token is similar mainly to other tokens in its own group. Dense attention then
peaks on same-group keys -- exactly the redundancy DSA is designed to exploit
(the paper notes "90% of attention entries in long contexts are redundant").

Writes ``data/sample_sequence.json`` (token embeddings + group labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate(seq_len: int = 64, d_model: int = 32, n_groups: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    # One random centroid per group; well separated so groups are distinct.
    centroids = rng.standard_normal((n_groups, d_model))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    groups = rng.integers(0, n_groups, size=seq_len)
    # token = its group's centroid (scaled up) + small noise. The large scale
    # makes same-group dot products dominate, so dense attention concentrates
    # almost all of its mass on a token's own (small) group -> genuinely sparse.
    x = 6.0 * centroids[groups] + 0.25 * rng.standard_normal((seq_len, d_model))
    return x.astype(np.float32), groups.astype(int)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    x, groups = generate()
    payload = {
        "seq_len": int(x.shape[0]),
        "d_model": int(x.shape[1]),
        "embeddings": x.tolist(),
        "groups": groups.tolist(),
    }
    path = out_dir / "sample_sequence.json"
    path.write_text(json.dumps(payload))
    print(f"Wrote {path} : seq_len={x.shape[0]} d_model={x.shape[1]}")


if __name__ == "__main__":
    main()
