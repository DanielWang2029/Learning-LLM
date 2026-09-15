"""Vocabulary, toy corpus, span-corruption and the text-to-text tasks for T5.

Token layout (a single shared vocabulary for input *and* output — T5 is
text-to-text)::

    0 PAD   1 BOS   2 EOS
    3..8    sentinel tokens  <X0>..<X5>   (mark corrupted spans)
    9       task: copy
    10      task: reverse
    11      task: sort
    12..     content tokens

Two data recipes are provided:

* **span corruption** (unsupervised pre-training): random spans of a sentence
  are replaced by sentinels in the encoder input; the decoder target is the
  dropped spans, each prefixed by its sentinel.
* **downstream tasks** (supervised): the encoder input is a task token followed
  by a sequence; the decoder target is that sequence transformed (copied,
  reversed or sorted) plus EOS.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

PAD, BOS, EOS = 0, 1, 2
NUM_SENTINELS = 6
SENTINELS = list(range(3, 3 + NUM_SENTINELS))  # 3..8
TASK_COPY, TASK_REVERSE, TASK_SORT = 9, 10, 11
TASK_TOKENS = {"copy": TASK_COPY, "reverse": TASK_REVERSE, "sort": TASK_SORT}
CONTENT_START = 12


def vocab_size(n_content: int) -> int:
    return CONTENT_START + n_content


def build_grammar(n_content: int, seed: int):
    """A fixed successor cycle so masked spans are predictable from context."""
    rng = random.Random(seed)
    toks = list(range(CONTENT_START, CONTENT_START + n_content))
    cycle = toks[:]
    rng.shuffle(cycle)
    return {cycle[i]: cycle[(i + 1) % n_content] for i in range(n_content)}


def sample_text(succ, length: int, rng: random.Random):
    start = rng.choice(list(succ))
    seq = [start]
    for _ in range(length - 1):
        seq.append(succ[seq[-1]])
    return seq


def corrupt_spans(seq, rng: random.Random, mask_prob: float = 0.15, max_span: int = 3):
    """Return (encoder_input, target) in T5 span-corruption format."""
    n = len(seq)
    masked = [False] * n
    i = 0
    while i < n:
        if rng.random() < mask_prob:
            span = rng.randint(1, max_span)
            for j in range(i, min(i + span, n)):
                masked[j] = True
            i += span + 1  # leave at least one visible token between spans
        else:
            i += 1
    if not any(masked):  # guarantee at least one span
        masked[rng.randrange(n)] = True

    enc, target, si = [], [], 0
    i = 0
    while i < n:
        if masked[i]:
            sentinel = SENTINELS[min(si, NUM_SENTINELS - 1)]
            enc.append(sentinel)
            target.append(sentinel)
            while i < n and masked[i]:
                target.append(seq[i])
                i += 1
            si += 1
        else:
            enc.append(seq[i])
            i += 1
    target.append(EOS)
    return enc, target


def task_example(task: str, seq):
    """Return (encoder_input, target) for a downstream text-to-text task."""
    if task == "copy":
        out = list(seq)
    elif task == "reverse":
        out = list(reversed(seq))
    elif task == "sort":
        out = sorted(seq)
    else:
        raise ValueError(task)
    return [TASK_TOKENS[task]] + list(seq), out + [EOS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-content", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    succ = build_grammar(args.n_content, args.seed)
    text = sample_text(succ, 12, rng)
    enc, tgt = corrupt_spans(text, rng)
    demo_tasks = {t: task_example(t, sample_text(succ, 6, rng)) for t in TASK_TOKENS}

    payload = {
        "description": "T5 text-to-text: span corruption + copy/reverse/sort tasks.",
        "vocab_size": vocab_size(args.n_content),
        "special": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
        "sentinels": SENTINELS,
        "task_tokens": TASK_TOKENS,
        "content_start": CONTENT_START,
        "example_text": text,
        "span_corruption_example": {"encoder_input": enc, "target": tgt},
        "task_examples": {t: {"encoder_input": e, "target": g} for t, (e, g) in demo_tasks.items()},
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote vocab={vocab_size(args.n_content)} example -> {out}")
    print("span corruption  enc:", enc)
    print("span corruption  tgt:", tgt)
    for t, (e, g) in demo_tasks.items():
        print(f"  {t:>7}: {e}  ->  {g}")


if __name__ == "__main__":
    main()
