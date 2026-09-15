"""Serialize a few sample sequences + the layer pattern to data/sample_dataset.json.

The demo generates its own seeded batches; this writes a couple of readable
examples and the 5:1 layer pattern. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import make_batch, describe_token, layer_pattern, mask_entries

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    seq_len, window = 64, 4
    gen = torch.Generator().manual_seed(0)
    tokens, qpos, values, payload = make_batch(3, seq_len, gen)
    pattern = layer_pattern(6, ratio=5)
    payload_examples = []
    for i in range(3):
        p = int(payload[i])
        payload_examples.append({
            "seq_len": seq_len,
            "payload_pos": p,
            "distance_to_query": int(qpos[i]) - p,
            "around_payload": [describe_token(int(t)) for t in tokens[i, max(0, p - 1):p + 3]],
            "true_value": int(values[i]),
        })
    payload_data = {
        "description": "Long-context retrieval: find MARK, output the value token "
                       "right after it, from the QUERY position at the end.",
        "layer_pattern_5to1": pattern,
        "attn_entries_per_layer": {
            "local(window=4)": mask_entries(seq_len, "local", window),
            "global": mask_entries(seq_len, "global", window),
        },
        "examples": payload_examples,
    }
    out = DATA_DIR / "sample_dataset.json"
    out.write_text(json.dumps(payload_data, indent=2))
    print(f"wrote layer pattern + {len(payload_examples)} examples -> {out}")


if __name__ == "__main__":
    main()
