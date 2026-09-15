"""Generate a char-level corpus whose text embeds several *tasks*.

GPT-2's thesis is that many tasks can be learned simply by predicting the next
token, provided the tasks are expressed *in the text itself*. We build a corpus
of lines of the form::

    <task>: <input> = <output>;

for a family of simple string-transform tasks (reverse / upper / sort). During
training the model only ever sees the language-modelling objective. At test
time we prompt it with an unseen ``<task>: <input> =`` and let it generate the
answer *zero-shot* — no task-specific parameters or fine-tuning.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

LETTERS = "abcdefgh"
SEP = ";"  # end-of-example marker (acts as the stop token at generation time)

TASKS = {
    "reverse": lambda s: s[::-1],
    "upper": lambda s: s.upper(),
    "sort": lambda s: "".join(sorted(s)),
}


def make_example(task: str, inp: str) -> str:
    return f"{task}: {inp} = {TASKS[task](inp)}{SEP}"


def random_input(rng: random.Random, min_len=3, max_len=5) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(LETTERS) for _ in range(n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tasks = list(TASKS)
    lines = [make_example(rng.choice(tasks), random_input(rng)) for _ in range(args.num_examples)]
    text = "".join(lines)

    payload = {
        "description": "Char-level multitask corpus for GPT-2 zero-shot evaluation.",
        "letters": LETTERS,
        "sep": SEP,
        "tasks": list(TASKS),
        "alphabet": sorted(set(text)),
        "text": text,
        "examples_preview": lines[:8],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(lines)} task examples ({len(text)} chars) -> {out}")
    print("preview:", " ".join(lines[:4]))


if __name__ == "__main__":
    main()
