"""Write the toy corpus to disk in a human-readable form.

The demo builds the graph directly from `src/corpus.py`; this script just dumps
the corpus (rendered sentences + underlying triples) so you can read it.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import FACTS, THEMES, render_sentence

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    payload = {
        "themes": THEMES,
        "documents": [
            {"text": render_sentence(f), "subject": f[0], "relation": f[1],
             "object": f[2], "theme": f[3]}
            for f in FACTS
        ],
    }
    out = DATA_DIR / "corpus.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(FACTS)} facts across {len(THEMES)} themes -> {out}")
    for f in FACTS[:5]:
        print("  ", render_sentence(f))


if __name__ == "__main__":
    main()
