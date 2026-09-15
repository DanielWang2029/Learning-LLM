"""Write a few human-readable samples of the base and new tasks.

The demo generates its own data in-memory; this script just materializes a
handful of labeled examples so the shared-feature setup is inspectable.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import make_task, IN_DIM, HIDDEN, NUM_CLASSES

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    xb, yb = make_task("base", 5, seed=10)
    xn, yn = make_task("new", 5, seed=20)
    payload = {
        "meta": {"in_dim": IN_DIM, "hidden": HIDDEN, "classes": NUM_CLASSES,
                 "note": "base & new tasks share a feature extractor; heads differ"},
        "base_samples": [{"x": [round(v, 3) for v in xb[i].tolist()],
                          "label": int(yb[i])} for i in range(xb.size(0))],
        "new_samples": [{"x": [round(v, 3) for v in xn[i].tolist()],
                         "label": int(yn[i])} for i in range(xn.size(0))],
    }
    out = DATA_DIR / "samples.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print("base labels:", yb.tolist())
    print("new  labels:", yn.tolist())


if __name__ == "__main__":
    main()
