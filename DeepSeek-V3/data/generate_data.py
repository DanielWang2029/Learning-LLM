"""Materialize a few examples of the toy copy task used by the demo.

The demo generates copy-task batches on the fly, so no dataset download is needed.
This writes ``data/task.json`` documenting the layout so it is easy to inspect.

    [BOS, content..., SEP, content...]   # model copies `content`; MTP predicts t+2

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PAD, BOS, SEP = 0, 1, 2
NUM_SPECIAL = 3


def main() -> None:
    random.seed(0)
    vocab, n, count = 32, 6, 6
    samples = []
    for _ in range(count):
        content = [random.randint(NUM_SPECIAL, vocab - 1) for _ in range(n)]
        samples.append({"tokens": [BOS] + content + [SEP] + content, "content": content})
    out = {"format": "[BOS, content..., SEP, content...]",
           "specials": {"PAD": PAD, "BOS": BOS, "SEP": SEP},
           "note": "main head predicts token t+1; MTP head predicts token t+2",
           "vocab_size": vocab, "n": n, "samples": samples}
    path = Path(__file__).resolve().parent / "task.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({count} example sequences)")


if __name__ == "__main__":
    main()
