"""Write small human-readable samples of the clean and noisy datasets.

The demo generates its own data in-memory; this script just materializes a few
examples to `data/` so you can eyeball the difference between a clean "textbook"
sequence and a corrupted "web" sequence.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import make_clean_dataset, make_noisy_dataset, VOCAB_SIZE, SEQ_LEN

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    clean = make_clean_dataset(6, seed=100).tolist()
    noisy = make_noisy_dataset(6, seed=100).tolist()
    payload = {
        "meta": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "rule": "next = (prev1 + prev2) mod V",
            "note": "clean = pure rule; noisy = ~40% of contexts systematically wrong",
        },
        "clean_samples": clean,
        "noisy_samples": noisy,
    }
    out = DATA_DIR / "samples.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print("clean example:", clean[0])
    print("noisy example:", noisy[0])


if __name__ == "__main__":
    main()
