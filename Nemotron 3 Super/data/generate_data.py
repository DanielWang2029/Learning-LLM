"""Long-range recall task for the hybrid Mamba-Attention model.

A single *content* token appears at the very start of the sequence, followed by
a long run of random *distractor* tokens, then a QUERY marker at the end. The
model must reproduce the content token — information it has to carry across the
entire sequence. This is the classic stress test for long-range sequence mixing
(and what the SSM layers' recurrent state is for).

Token layout:
    0 .. C-1            : content symbols
    C .. C+Dd-1         : distractor symbols
    QUERY = C+Dd        : "recall the content" marker

Writes ``data/recall_examples.json`` with a few human-readable samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

N_CONTENT = 10
N_DISTRACT = 10
QUERY = N_CONTENT + N_DISTRACT
VOCAB = N_CONTENT + N_DISTRACT + 1


def make_batch(batch_size: int, seq_len: int, rng: np.random.Generator):
    """Return (tokens, target). tokens: (B, L); target: (B,) content id."""
    B, L = batch_size, seq_len
    content = rng.integers(0, N_CONTENT, size=B)
    distract = rng.integers(0, N_DISTRACT, size=(B, L)) + N_CONTENT
    tokens = distract.astype(np.int64)
    tokens[:, 0] = content            # content at the start
    tokens[:, -1] = QUERY             # query marker at the end
    return torch.from_numpy(tokens), torch.from_numpy(content.astype(np.int64))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(0)
    toks, tgt = make_batch(5, 16, rng)
    samples = [{"tokens": toks[i].tolist(), "content": int(tgt[i])} for i in range(5)]
    payload = {"n_content": N_CONTENT, "n_distract": N_DISTRACT,
               "query_token": QUERY, "vocab": VOCAB, "samples": samples}
    path = out_dir / "recall_examples.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path} : {len(samples)} samples, vocab={VOCAB}")


if __name__ == "__main__":
    main()
