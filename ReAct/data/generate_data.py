"""Write the knowledge base and question set to data/ for inspection.

The environment (KB + tools) is defined in code; this just serializes it so you
can read it. Run with:  python data/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import KB, QUESTIONS, gold_answer

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    payload = {
        "description": "Toy knowledge base and multi-hop questions for the ReAct agent.",
        "knowledge_base": KB,
        "questions": [{"text": q["text"], "kind": q["kind"], "gold": gold_answer(q)}
                      for q in QUESTIONS],
    }
    out = DATA_DIR / "environment.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote KB ({len(KB)} entities) and {len(QUESTIONS)} questions -> {out}")


if __name__ == "__main__":
    main()
