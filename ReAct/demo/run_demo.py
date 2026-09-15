"""End-to-end ReAct demo (CPU, instant).

Runs three agents on the same multi-hop questions:

* ReAct           — interleaves Thought → Action → Observation, chaining tools.
* Reasoning-only  — reasons but cannot call tools (hallucinates).
* Acting-only     — calls tools but cannot chain them (no multi-hop).

Prints the full interleaved reasoning+acting trace for ReAct, compares
accuracy, and writes ``data/react_results.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import QUESTIONS, acting_only, gold_answer, react_agent, reasoning_only

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def format_step(s: dict) -> str:
    if s["type"] == "thought":
        return f"  Thought: {s['text']}"
    if s["type"] == "action":
        args = ", ".join(map(str, s["args"]))
        return f"  Action:  {s['tool']}[{args}]\n  Observation: {s['obs']}"
    if s["type"] == "finish":
        return f"  Finish:  {s['answer']}"
    return ""


def main() -> None:
    start = time.time()
    agents = [("ReAct", react_agent), ("Reasoning-only", reasoning_only),
              ("Acting-only", acting_only)]

    scores = {name: 0 for name, _ in agents}
    per_question = []
    for q in QUESTIONS:
        gold = gold_answer(q)
        row = {"question": q["text"], "gold": gold, "agents": {}}
        for name, fn in agents:
            res = fn(q)
            scores[name] += int(res["correct"])
            row["agents"][name] = res
        per_question.append(row)

    n = len(QUESTIONS)

    # Print full ReAct traces (the paper's key artifact: the interleaved trace).
    print("================= ReAct interleaved traces =================")
    for row in per_question:
        r = row["agents"]["ReAct"]
        print(f"\nQ: {row['question']}")
        for s in r["steps"]:
            print(format_step(s))
        mark = "OK" if r["correct"] else "WRONG"
        print(f"  [gold: {row['gold']}]  -> {mark}")

    print("\n==================== ACCURACY ====================")
    for name, _ in agents:
        print(f"  {name:16s}: {scores[name]}/{n}  ({scores[name]/n*100:.0f}%)")

    # Contrast on one multi-hop question where BOTH baselines fail.
    contrast_q = next(
        (row for row in per_question
         if not row["agents"]["Reasoning-only"]["correct"]
         and not row["agents"]["Acting-only"]["correct"]
         and row["agents"]["ReAct"]["correct"]),
        per_question[0],
    )
    print(f"\n=== Why tools + reasoning are BOTH needed ===\nQ: {contrast_q['question']}  (gold {contrast_q['gold']})")
    for name in ("ReAct", "Reasoning-only", "Acting-only"):
        a = contrast_q["agents"][name]
        print(f"  {name:16s} -> {a['answer']:12s} {'✓' if a['correct'] else '✗'}")

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed:.3f}s")

    out = {
        "config": {"n_questions": n, "seconds": round(elapsed, 3)},
        "accuracy": {name: {"correct": scores[name], "total": n,
                            "rate": round(scores[name] / n, 4)} for name, _ in agents},
        "questions": per_question,
        "contrast": contrast_q,
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "react_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    react_rate = scores["ReAct"] / n
    if react_rate <= scores["Reasoning-only"] / n or react_rate <= scores["Acting-only"] / n:
        raise SystemExit("ReAct did not beat both baselines — check the setup.")
    print("\nOK: ReAct (reason + act) beats reasoning-only and acting-only.")


if __name__ == "__main__":
    main()
