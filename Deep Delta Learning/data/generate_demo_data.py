"""Write a small, human-readable sample of key-value pairs to data/.

Run with:  python data/generate_demo_data.py

Optional: demo/run_demo.py generates data on the fly. This just shows the kind
of (key, value) bindings the associative memory stores and recalls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import DeltaAssociativeMemory  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def main() -> None:
    d = 8
    g = torch.Generator().manual_seed(0)
    keys = torch.nn.functional.normalize(torch.randn(4, d, generator=g), dim=-1)
    values = torch.nn.functional.normalize(torch.randn(4, d, generator=g), dim=-1)
    mem = DeltaAssociativeMemory(d, d)
    mem.store(keys, values)
    recalled = mem.read_all(keys)
    payload = {
        "description": "Delta-rule associative memory: store 4 key->value pairs, "
                       "then recall. Vectors rounded for readability.",
        "dim": d,
        "pairs": [
            {"key": [round(x, 3) for x in k.tolist()],
             "value": [round(x, 3) for x in v.tolist()],
             "recalled": [round(x, 3) for x in r.tolist()]}
            for k, v, r in zip(keys, values, recalled)
        ],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "memory_sample.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(payload['pairs'])} sample pairs -> {out}")


if __name__ == "__main__":
    main()
