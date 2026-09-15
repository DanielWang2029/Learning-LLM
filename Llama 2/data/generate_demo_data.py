"""Generate a tiny, human-readable sample of the copy task used by the demo.

Writes ``data/copy_dataset.json``. Each example is a decoder-only sequence:

    [BOS] c1 c2 ... ck [SEP] c1 c2 ... ck

The model is trained (in ``demo/run_demo.py``) to reproduce the content tokens
after the [SEP] delimiter — a compact test that grouped-query attention still
learns to route information across positions.

Run with:  python data/generate_demo_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_SPECIAL = 4
CONTENT_VOCAB = 12  # content token ids: 4 .. 15
VOCAB_SIZE = NUM_SPECIAL + CONTENT_VOCAB


def make_example(seq_len: int, rng: random.Random):
    content = [rng.randrange(NUM_SPECIAL, VOCAB_SIZE) for _ in range(seq_len)]
    return {"content": content, "tokens": [BOS] + content + [SEP] + content}


def main() -> None:
    rng = random.Random(0)
    seq_len = 6
    examples = [make_example(seq_len, rng) for _ in range(12)]
    out = {
        "task": "copy",
        "description": "[BOS] c1..ck [SEP] c1..ck  — reproduce content after [SEP]",
        "special_tokens": {"PAD": PAD, "BOS": BOS, "SEP": SEP, "EOS": EOS},
        "vocab_size": VOCAB_SIZE,
        "seq_len": seq_len,
        "examples": examples,
    }
    path = Path(__file__).resolve().parent / "copy_dataset.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({len(examples)} examples).")


if __name__ == "__main__":
    main()
