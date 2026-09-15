"""Write a small, human-readable sample of the associative-recall task to data/.

Run with:  python data/generate_demo_data.py

This is optional: demo/run_demo.py generates data on the fly. The JSON written
here just lets you inspect the exact token layout the model is trained on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.task import RecallVocab, make_example  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def main() -> None:
    rng = torch.Generator().manual_seed(0)
    vocab = RecallVocab(n_keys=6, n_vals=10)
    examples = []
    for _ in range(8):
        tokens, targets, meta = make_example(vocab, n_overwrite=3, rng=rng)
        examples.append({"tokens": tokens, "targets": targets, "meta": meta})

    payload = {
        "description": "Key-value associative recall with overwrite.",
        "vocab": {
            "size": vocab.size,
            "PAD": vocab.PAD,
            "value_tokens": [vocab.val_base, vocab.val_base + vocab.n_vals - 1],
            "write_key_tokens": [vocab.wkey_base, vocab.wkey_base + vocab.n_keys - 1],
            "query_key_tokens": [vocab.qkey_base, vocab.qkey_base + vocab.n_keys - 1],
            "ignore_index": -100,
        },
        "examples": examples,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "recall_sample.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample sequences -> {out}")


if __name__ == "__main__":
    main()
