"""Write a small, human-readable sample of the trigram-grammar task to data/.

Run with:  python data/generate_demo_data.py

Optional: demo/run_demo.py samples the task on the fly. This just lets you see
the sequences and the (a, b, c) -> next structure the models must learn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.task import TrigramGrammar  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def main() -> None:
    vocab = 16
    grammar = TrigramGrammar(vocab, seed=1)
    rng = torch.Generator().manual_seed(0)
    inp, tgt = grammar.batch(6, seq_len=16, rng=rng)
    examples = []
    for row_in, row_tgt in zip(inp.tolist(), tgt.tolist()):
        examples.append({
            "tokens": row_in,
            "targets": row_tgt,  # -100 where no trigram context yet
        })
    payload = {
        "description": "Trigram grammar: next token = T[a, b, c]. Predict the "
                       "grammar-correct next token from the previous 3 tokens.",
        "vocab_size": vocab,
        "ignore_index": -100,
        "examples": examples,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "task_sample.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample sequences -> {out}")


if __name__ == "__main__":
    main()
