"""Associative-recall (MQAR-style) task generator for the KDA demo.

Each example is a stream of key→value pairs followed by a query marker and one
of the keys; the model must output that key's value. This is the canonical test
for a *memory* mechanism: the layer has to bind kᵢ to vᵢ when it sees them and
retrieve the right one later — exactly what the delta rule is built for.

Token layout:
    0 .. n_keys-1                 : key symbols
    n_keys .. n_keys+n_vals-1     : value symbols
    QUERY = n_keys+n_vals         : "recall now" marker

Writes ``data/recall_examples.json`` with a few human-readable samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

N_KEYS = 16
N_VALS = 16
QUERY = N_KEYS + N_VALS
VOCAB = N_KEYS + N_VALS + 1


def make_batch(batch_size: int, n_pairs: int, seed_gen: np.random.Generator):
    """Return (tokens, target_value, query_pos).

    tokens: (B, L) with L = 2*n_pairs + 2  (pairs, QUERY marker, query key)
    target_value: (B,) value token id to predict at the last position
    """
    B = batch_size
    L = 2 * n_pairs + 2
    tokens = np.zeros((B, L), dtype=np.int64)
    target = np.zeros((B,), dtype=np.int64)
    for i in range(B):
        keys = seed_gen.permutation(N_KEYS)[:n_pairs]
        vals = seed_gen.integers(0, N_VALS, size=n_pairs) + N_KEYS
        seq = np.empty(2 * n_pairs, dtype=np.int64)
        seq[0::2] = keys
        seq[1::2] = vals
        tokens[i, : 2 * n_pairs] = seq
        tokens[i, 2 * n_pairs] = QUERY
        qi = seed_gen.integers(0, n_pairs)
        tokens[i, 2 * n_pairs + 1] = keys[qi]
        target[i] = vals[qi]
    return torch.from_numpy(tokens), torch.from_numpy(target)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(0)
    toks, tgt = make_batch(5, 6, rng)
    samples = []
    for i in range(toks.size(0)):
        samples.append({
            "tokens": toks[i].tolist(),
            "target_value": int(tgt[i].item()),
        })
    payload = {
        "n_keys": N_KEYS, "n_vals": N_VALS, "query_token": QUERY,
        "vocab": VOCAB, "samples": samples,
    }
    path = out_dir / "recall_examples.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path} : {len(samples)} sample sequences, vocab={VOCAB}")


if __name__ == "__main__":
    main()
