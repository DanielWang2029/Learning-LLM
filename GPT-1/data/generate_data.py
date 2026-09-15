"""Generate a tiny char-level corpus for GPT-1 pre-training + fine-tuning.

Two ingredients are produced and written to ``data/corpus.json``:

* ``pretrain_text`` — a long unlabeled string of simple template sentences
  drawn from *two topics* (animals and plants). This is the corpus for the
  self-supervised generative pre-training stage.

* ``classification`` — labeled ``(text, label)`` sentences for the downstream
  supervised task: predict the topic (0 = animals, 1 = plants). Both topics
  share the same sentence template, so the label is only recoverable by
  actually reading the content words — the kind of feature pre-training is
  meant to provide.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

TOPICS = {
    0: {  # animals
        "nouns": ["cat", "dog", "cow", "bird", "fish", "frog", "goat", "duck"],
        "verbs": ["runs", "swims", "jumps", "hides", "rests", "walks"],
    },
    1: {  # plants
        "nouns": ["oak", "fern", "rose", "weed", "pine", "moss", "palm", "vine"],
        "verbs": ["grows", "blooms", "spreads", "sways", "wilts", "climbs"],
    },
}
TEMPLATE = "the {noun} {verb} today . "


def make_sentence(label: int, rng: random.Random) -> str:
    t = TOPICS[label]
    return TEMPLATE.format(noun=rng.choice(t["nouns"]), verb=rng.choice(t["verbs"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-sentences", type=int, default=600)
    parser.add_argument("--labeled", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    pretrain = "".join(make_sentence(rng.randint(0, 1), rng) for _ in range(args.pretrain_sentences))

    labeled = []
    for _ in range(args.labeled):
        label = rng.randint(0, 1)
        labeled.append({"text": make_sentence(label, rng).strip(), "label": label})

    alphabet = sorted(set(pretrain))
    payload = {
        "description": "Char-level two-topic corpus for GPT-1 pretrain + finetune.",
        "alphabet": alphabet,
        "topics": {"0": "animals", "1": "plants"},
        "template": TEMPLATE,
        "pretrain_text": pretrain,
        "classification": labeled,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(pretrain)} pretrain chars and {len(labeled)} labeled sentences -> {out}")
    print(f"alphabet ({len(alphabet)} chars): {''.join(alphabet)!r}")


if __name__ == "__main__":
    main()
