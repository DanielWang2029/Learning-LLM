"""Write a small, human-readable sample of the topic-mapping task to data/.

Run with:  python data/generate_demo_data.py

Optional: demo/run_demo.py samples the task on the fly. This just lets you see
the (symbol, topic) -> target structure the experts must learn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src.task import TopicPermutations  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def main() -> None:
    n_symbols, n_topics = 64, 64
    task = TopicPermutations(n_symbols, n_topics, seed=1)
    rng = torch.Generator().manual_seed(0)
    s, g, y = task.batch(16, rng)
    examples = [
        {"symbol": int(si), "topic": int(gi), "target": int(yi)}
        for si, gi, yi in zip(s.tolist(), g.tolist(), y.tolist())
    ]
    payload = {
        "description": "Topic-conditioned symbol mapping: target = perm_topic(symbol).",
        "n_symbols": n_symbols,
        "n_topics": n_topics,
        "note": "Symbol embeddings are shared across topics, so the answer needs "
                "topic-dependent computation — ideal for expert specialization.",
        "examples": examples,
        "topic0_permutation": task.perms[0].tolist(),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "task_sample.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} sample pairs -> {out}")


if __name__ == "__main__":
    main()
