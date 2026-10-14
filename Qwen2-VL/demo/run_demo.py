"""Train a tiny Qwen2-VL-style dynamic-resolution / M-RoPE classifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.synthesizer import make_batch
from src.model import DynamicResolutionClassifier


def train(use_mrope: bool, seed: int) -> float:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DynamicResolutionClassifier(use_mrope=use_mrope)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    resolutions = [24, 32, 40]
    model.train()
    for step in range(420):
        images, labels = make_batch(resolutions[step % 3], 24, rng)
        logits, _ = model(torch.from_numpy(images))
        loss = F.cross_entropy(logits, torch.from_numpy(labels))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for resolution in resolutions:
            images, labels = make_batch(resolution, 160, rng)
            pred = model(torch.from_numpy(images))[0].argmax(-1).numpy()
            correct += (pred == labels).sum(); total += len(labels)
    return correct / total


def main() -> None:
    torch.manual_seed(7)
    results = {"mrope_accuracy": train(True, 7), "no_position_accuracy": train(False, 7)}
    resolutions = [24, 32, 40]
    results["patch_size"] = 4
    results["token_counts"] = [{"resolution": r, "tokens": (r // 4) ** 2} for r in resolutions]
    results["chance_accuracy"] = 0.25
    output = ROOT / "data" / "demo_results.json"
    output.write_text(json.dumps(results, indent=2))
    print("Qwen2-VL tiny CPU reproduction")
    print("Naive dynamic resolution (no resize):")
    for item in results["token_counts"]:
        print(f"  {item['resolution']}×{item['resolution']} image -> {item['tokens']} visual tokens")
    print(f"M-RoPE quadrant accuracy:       {results['mrope_accuracy']:.1%}")
    print(f"No positional rotation accuracy: {results['no_position_accuracy']:.1%} (chance: 25.0%)")
    print(f"Wrote visualization data: {output}")


if __name__ == "__main__":
    main()
