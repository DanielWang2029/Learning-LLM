"""Write a small, human-readable sample of the retrieval task to data/.

Run with:  python data/generate_demo_data.py

Optional: demo/run_demo.py generates all metrics on the fly. This just shows a
few examples of the "find the marked token, return its payload" task and how a
strided convolution compresses the key/value sequence by a factor `r`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.task import MarkedRetrieval  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

VOCAB = 16
SEQ_LEN = 48
COMPRESS = 8


def main() -> None:
    rng = torch.Generator().manual_seed(0)
    task = MarkedRetrieval(VOCAB, SEQ_LEN)
    tokens, marks, labels = task.batch(4, rng)
    examples = []
    for t, m, y in zip(tokens, marks, labels):
        pos = int(m.argmax())
        examples.append({
            "tokens": t.tolist(),
            "marked_pos": pos,
            "payload_at_marked": int(t[pos]),
            "label": int(y),
        })
    payload = {
        "description": "Content-based retrieval: each sequence has one marked "
                       "token; the label is that token's payload id. CCA "
                       f"compresses the {SEQ_LEN} K/V positions to "
                       f"{-(-SEQ_LEN // COMPRESS)} latent tokens (compress={COMPRESS}).",
        "vocab": VOCAB,
        "seq_len": SEQ_LEN,
        "compress": COMPRESS,
        "kv_positions_full": SEQ_LEN,
        "kv_positions_cca": -(-SEQ_LEN // COMPRESS),
        "examples": examples,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "task_sample.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample sequences -> {out}")


if __name__ == "__main__":
    main()
