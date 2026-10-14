"""Train a short-context retriever, then apply DCA without further training."""
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]
from dca import TinyRelativeRetriever  # noqa: E402
from generate import make_needles  # noqa: E402


def predict(model, tokens, dca):
    probs = model(torch.tensor(tokens), len(tokens), dca)
    return int(probs.argmax())


def accuracy(model, length, dca, count=80):
    examples = make_needles(length, count, seed=17)
    return sum(predict(model, x, dca) == target for x, target in examples) / count


def main():
    torch.manual_seed(7)
    model = TinyRelativeRetriever(train_length=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)

    # Short-only training: attention learns that the matching content is useful.
    for step in range(280):
        tokens, target = make_needles(64, 1, seed=step)[0]
        log_probs = torch.log(model(torch.tensor(tokens), 64, False) + 1e-9)
        loss = -log_probs[target]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lengths = [64, 128, 256, 512, 1024]
    vanilla = [round(accuracy(model, n, False), 3) for n in lengths]
    dca = [round(accuracy(model, n, True), 3) for n in lengths]
    example, needle = make_needles(256, 1, seed=99)[0]
    print("Qwen2.5-1M — training-free Dual Chunk Attention")
    print(f"short-context training: length=64, learned content scale={model.content_scale.item():.2f}")
    print("length | vanilla absolute-relative | DCA remapped-relative")
    for n, a, b in zip(lengths, vanilla, dca):
        print(f"{n:>6} | {a:>25.1%} | {b:>20.1%}")
    print(f"worked 256-token needle: true={needle}, vanilla={predict(model, example, False)}, "
          f"DCA={predict(model, example, True)}")

    trace = {
        "title": "DCA needle recall after short-context training",
        "train_length": 64,
        "chunk_size": 32,
        "lengths": lengths,
        "vanilla_recall": vanilla,
        "dca_recall": dca,
        "worked_example": {"length": 256, "needle": needle,
                           "vanilla_prediction": predict(model, example, False),
                           "dca_prediction": predict(model, example, True)},
    }
    (ROOT / "data" / "dca_results.json").write_text(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
