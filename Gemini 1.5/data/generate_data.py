"""Materialize a couple of needle-in-a-haystack examples for inspection.

The demo generates the retrieval task on the fly, so no dataset download is
needed. This writes ``data/task.json`` with a couple of decoded example sequences
so the task format is easy to read.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.task import NEEDLE, PAD, QUERY, TOKEN_BASE, make_eval_batch  # noqa: E402


def decode(tok: int) -> str:
    if tok == PAD:
        return "PAD"
    if tok == NEEDLE:
        return "<NEEDLE>"
    if tok == QUERY:
        return "<QUERY>"
    return f"t{tok - TOKEN_BASE}"


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    examples = []
    for n_fill, frac in [(6, 0.0), (10, 0.5)]:
        tokens, target = make_eval_batch(1, n_fill, frac, gen)
        seq = [decode(int(t)) for t in tokens[0].tolist()]
        examples.append({"n_fill": n_fill, "needle_frac": frac,
                         "sequence": " ".join(seq),
                         "answer": decode(int(target[0]))})
    out = {"task": "needle-in-a-haystack retrieval: find <NEEDLE>, copy the token after it",
           "layout": "d d ... <NEEDLE> v ... d <QUERY>  -> v",
           "examples": examples}
    path = Path(__file__).resolve().parent / "task.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    for e in examples:
        print(f"  {e['sequence']}   =>  {e['answer']}")


if __name__ == "__main__":
    main()
