"""Materialize the toy arithmetic task so its format is easy to inspect.

The GRPO demo generates prompts on the fly, so no dataset download is needed.
This writes ``data/task.json`` describing the prompt set and reward rule.

Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.task import ID_TO_STR, all_prompts, encode_prompt  # noqa: E402


def main() -> None:
    max_operand = 6
    prompts = all_prompts(max_operand)
    examples = []
    for a, b in prompts[:12]:
        ids = encode_prompt(a, b)
        examples.append({"a": a, "b": b, "prompt_ids": ids,
                         "prompt_str": "".join(ID_TO_STR[i] for i in ids),
                         "target": a + b})
    out = {
        "task": "single-digit addition, answer generated as digit tokens",
        "reward_rule": "1.0 if answer == a+b, 0.1 if right #digits, else 0.0",
        "max_operand": max_operand,
        "n_prompts": len(prompts),
        "vocab": ID_TO_STR,
        "examples": examples,
    }
    path = Path(__file__).resolve().parent / "task.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({len(prompts)} prompts, showing {len(examples)} examples)")


if __name__ == "__main__":
    main()
