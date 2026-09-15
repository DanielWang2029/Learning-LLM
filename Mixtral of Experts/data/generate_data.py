"""Write a few human-readable samples of the multi-domain MoE dataset.

The demo generates its own data in-memory; this script just materializes a few
example sequences (one per domain) so you can see the distinct sub-patterns the
experts learn to specialize on.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import make_dataset, N_DOMAINS, DOMAIN_SIZE, VOCAB_SIZE, SEQ_LEN

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    seqs, labels = make_dataset(N_DOMAINS * 2, seed=7)
    payload = {
        "meta": {
            "n_domains": N_DOMAINS,
            "domain_size": DOMAIN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "note": "each domain uses its own token block and arithmetic step",
        },
        "samples": [
            {"domain": int(labels[i]), "sequence": seqs[i].tolist()}
            for i in range(seqs.size(0))
        ],
    }
    out = DATA_DIR / "samples.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    for i in range(N_DOMAINS):
        print(f"  domain {i}: {seqs[i].tolist()}")


if __name__ == "__main__":
    main()
