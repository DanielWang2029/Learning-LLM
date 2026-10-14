"""Compare direct response, CoT, and tool-integrated reasoning on arithmetic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_data import problems
from src.reasoner import chain_of_thought, direct, tool_integrated_reasoning


def score(agent, rows):
    outputs = [agent(row) for row in rows]
    accuracy = sum(out["answer"] == row["answer"] for out, row in zip(outputs, rows)) / len(rows)
    return accuracy, outputs


def main() -> None:
    torch.manual_seed(12122)
    rows = problems()
    direct_acc, direct_outputs = score(direct, rows)
    cot_acc, cot_outputs = score(chain_of_thought, rows)
    tir_acc, tir_outputs = score(tool_integrated_reasoning, rows)
    # Select a large instance so the executor's benefit is visible in the trace.
    index = next(i for i, row in enumerate(rows) if row["scale"] == "large")
    example = rows[index]
    payload = {
        "paper": "Qwen2.5-Math", "metric": "exact final-answer match",
        "accuracy": {"direct": round(direct_acc * 100, 1), "cot": round(cot_acc * 100, 1), "tir": round(tir_acc * 100, 1)},
        "example": {
            "question": example["question"], "gold": example["answer"],
            "direct": direct_outputs[index], "cot": cot_outputs[index], "tir": tir_outputs[index],
        },
    }
    (ROOT / "data" / "tir_results.json").write_text(json.dumps(payload, indent=2))

    print("Qwen2.5-Math — tiny Tool-Integrated Reasoning reproduction")
    print(f"Direct answer accuracy:          {direct_acc * 100:5.1f}%")
    print(f"Chain-of-thought accuracy:       {cot_acc * 100:5.1f}%")
    print(f"Tool-integrated reasoning:       {tir_acc * 100:5.1f}%")
    print("\nExecuted TIR trace")
    print(f"  Question: {example['question']}")
    print(f"  CoT:      {cot_outputs[index]['trace']}")
    print(f"  Tool:     {payload['example']['tir']['tool_call']} → {payload['example']['tir']['tool_result']}")
    print(f"  Final:    {tir_outputs[index]['trace']}")
    if not (tir_acc > cot_acc > direct_acc):
        raise SystemExit("The controlled tool-use advantage was not observed.")
    print("\nOK: execution supplies exact arithmetic to the reasoning trace.")


if __name__ == "__main__":
    main()
