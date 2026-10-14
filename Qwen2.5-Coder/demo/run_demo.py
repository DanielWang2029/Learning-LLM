"""Train tiny decoders with and without FIM, then test code infilling.

Run from this paper folder: ``python demo/run_demo.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_data import examples, fim_tokens, next_tokens
from src.fim_model import TinyDecoder, greedy_middle


def encode(rows: list[list[str]], vocab: dict[str, int]) -> torch.Tensor:
    return torch.tensor([[vocab[token] for token in row] for row in rows], dtype=torch.long)


def train(sequences: torch.Tensor, vocab_size: int, seed: int) -> TinyDecoder:
    torch.manual_seed(seed)
    model = TinyDecoder(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(500):
        logits = model(sequences[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), sequences[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def evaluate(
    model: TinyDecoder, rows: list[dict[str, list[str]]], vocab: dict[str, int], inverse: list[str]
) -> tuple[float, dict[str, object]]:
    correct = 0
    worked: dict[str, object] = {}
    for i, row in enumerate(rows):
        prompt = ["<PRE>", *row["prefix"], "<SUF>", *row["suffix"], "<MID>"]
        got = [inverse[idx] for idx in greedy_middle(model, [vocab[x] for x in prompt], vocab["<MID>"], vocab["<END>"])]
        expected = row["middle"]
        correct += got == expected
        if i == 0:
            worked = {"prefix": " ".join(row["prefix"]), "suffix": " ".join(row["suffix"]),
                      "expected": " ".join(expected), "prediction": " ".join(got)}
    return correct / len(rows), worked


def main() -> None:
    torch.manual_seed(7)
    rows = examples()
    tokens = sorted({token for row in rows for token in fim_tokens(row) + next_tokens(row)})
    vocab = {token: i for i, token in enumerate(tokens)}
    inverse = tokens

    fim = train(encode([fim_tokens(row) for row in rows], vocab), len(vocab), seed=7)
    baseline = train(encode([next_tokens(row) for row in rows], vocab), len(vocab), seed=7)
    fim_acc, example = evaluate(fim, rows, vocab, inverse)
    baseline_acc, _ = evaluate(baseline, rows, vocab, inverse)

    payload = {
        "paper": "Qwen2.5-Coder", "metric": "exact middle-string match",
        "fim_accuracy": round(fim_acc * 100, 1), "baseline_accuracy": round(baseline_acc * 100, 1),
        "example": example, "format": "<PRE> prefix <SUF> suffix <MID> middle <END>",
    }
    (ROOT / "data" / "fim_results.json").write_text(json.dumps(payload, indent=2))

    print("Qwen2.5-Coder — tiny CPU FIM reproduction")
    print("Training objective: <PRE> prefix <SUF> suffix <MID> middle <END>")
    print(f"FIM exact infill accuracy:       {fim_acc * 100:5.1f}%")
    print(f"Next-token-only baseline:       {baseline_acc * 100:5.1f}%")
    print("\nHeld-out-format infill prompt")
    print(f"  prefix:   {example['prefix']}")
    print(f"  suffix:   {example['suffix']}")
    print(f"  expected: {example['expected']}")
    print(f"  FIM emits:{example['prediction']}")
    if fim_acc < 0.95 or fim_acc <= baseline_acc:
        raise SystemExit("FIM did not demonstrate an infilling advantage.")
    print("\nOK: moving suffix before the target teaches bidirectional code infilling.")


if __name__ == "__main__":
    main()
