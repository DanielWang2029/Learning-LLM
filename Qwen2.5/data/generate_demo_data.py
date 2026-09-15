"""Write a few human-readable examples of the toy copy task.

The Qwen2.5 demo trains on synthetic copy-task batches generated on the fly, so
no dataset download is needed. This script just materializes a handful of example
sequences to ``data/samples.json`` so the task format is easy to inspect:

    [BOS, content..., SEP, content...]   # the model must reproduce `content`

Run with:  python data/generate_demo_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PAD, BOS, SEP = 0, 1, 2
NUM_SPECIAL = 3


def main() -> None:
    random.seed(0)
    vocab, seq_len, n = 32, 6, 8
    samples = []
    for _ in range(n):
        content = [random.randint(NUM_SPECIAL, vocab - 1) for _ in range(seq_len)]
        samples.append({"input": [BOS] + content + [SEP], "target": content})
    out = {"format": "[BOS, content..., SEP] -> content...",
           "specials": {"PAD": PAD, "BOS": BOS, "SEP": SEP},
           "vocab_size": vocab, "seq_len": seq_len, "samples": samples}
    path = Path(__file__).resolve().parent / "samples.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({n} example copy-task sequences)")


if __name__ == "__main__":
    main()
