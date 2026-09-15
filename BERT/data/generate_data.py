"""Generate a tiny synthetic corpus with real, learnable structure.

To make masked-language-modelling meaningful we need a language where a token
can be predicted from its *surrounding context*. We build a simple
"successor grammar":

* the content vocabulary is arranged into one fixed random cycle, so every
  token has a unique successor and a unique predecessor;
* a sentence is a contiguous walk along that cycle starting at a random token.

Because each token is fully determined by either neighbour, a *bidirectional*
model that sees the left and/or right context can recover a masked token —
exactly the signal BERT's masked-LM objective exploits. The grammar and a few
example sentences are written to ``data/`` as human-readable JSON.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.model import NUM_SPECIAL  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def build_grammar(n_content: int, rng: random.Random):
    """Return successor map over content token ids arranged in one cycle."""
    tokens = list(range(NUM_SPECIAL, NUM_SPECIAL + n_content))
    cycle = tokens[:]
    rng.shuffle(cycle)
    succ = {cycle[i]: cycle[(i + 1) % n_content] for i in range(n_content)}
    return succ


def sample_sentence(succ, seq_len: int, rng: random.Random):
    start = rng.choice(list(succ.keys()))
    seq = [start]
    for _ in range(seq_len - 1):
        seq.append(succ[seq[-1]])
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-content", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-examples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    succ = build_grammar(args.n_content, rng)
    examples = [sample_sentence(succ, args.seq_len, rng) for _ in range(args.num_examples)]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Successor-grammar corpus for masked language modelling.",
        "special_tokens": {"PAD": 0, "CLS": 1, "MASK": 2, "SEP": 3},
        "n_content": args.n_content,
        "seq_len": args.seq_len,
        "successor_map": {str(k): v for k, v in succ.items()},
        "examples": examples,
    }
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sentences over {args.n_content} content tokens -> {out}")


if __name__ == "__main__":
    main()
