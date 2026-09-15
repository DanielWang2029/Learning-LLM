"""Serialize a slice of the corpus + a sample multi-hop question.

The demo builds its own seeded corpus; this writes a readable slice so you can
inspect the facts and the 2-hop answer path. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import Corpus

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    corpus = Corpus(generator=torch.Generator().manual_seed(0))
    # A few persons and the facts reachable from them.
    sample = []
    for p in corpus.persons[:5]:
        city = corpus.facts[p][0][1]
        sample.append({
            "person": corpus.name_of[p],
            "facts": corpus.fact_str(p),
            "city_facts": corpus.fact_str(city),
            "answer_country": corpus.name_of[corpus.answer_country(p)],
        })
    payload = {
        "description": "Multi-hop QA over a fact corpus. Answer = country the person "
                       "lives in (person -> lives_in city -> located_in country).",
        "n_persons": len(corpus.persons), "n_cities": len(corpus.cities),
        "n_countries": len(corpus.countries),
        "relations": ["lives_in", "located_in", "born_in (distractor)", "likes (distractor)"],
        "examples": sample,
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote corpus slice ({len(sample)} persons) -> {out}")


if __name__ == "__main__":
    main()
