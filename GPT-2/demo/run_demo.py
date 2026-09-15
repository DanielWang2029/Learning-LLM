"""GPT-2 zero-shot multitask demo.

Trains a small decoder-only LM with *only* the next-token objective on a corpus
of lines like ``reverse: cafe = efac;``. Then, with the weights frozen, it
evaluates the model *zero-shot* on freshly sampled, unseen inputs for each task
by prompting ``<task>: <input> =`` and greedily decoding the answer. No
task-specific head is ever added — a single LM performs every task.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "data"))

from src import GPT, GPTConfig  # noqa: E402
import generate_data  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
PAD = 0


class CharTokenizer:
    def __init__(self, alphabet):
        self.stoi = {"<pad>": 0}
        self.itos = {0: "<pad>"}
        for i, ch in enumerate(alphabet, start=1):
            self.stoi[ch] = i
            self.itos[i] = ch
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) != PAD)


def get_lm_batch(data, block_size, batch_size, device, rng):
    ix = [rng.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def run_task(model, tok, task, inp, device, max_new=8):
    prompt = f"{task}: {inp} = "
    ids = torch.tensor([tok.encode(prompt)], device=device)
    sep_id = tok.stoi[generate_data.SEP]
    out = model.generate(ids, max_new_tokens=max_new, stop_token=sep_id, greedy=True)
    gen = tok.decode(out[0, ids.size(1):].tolist())
    return gen.split(generate_data.SEP)[0]


def evaluate(model, tok, device, rng, n=100):
    results = {}
    examples = {}
    for task in generate_data.TASKS:
        correct = 0
        for k in range(n):
            inp = generate_data.random_input(rng)
            expected = generate_data.TASKS[task](inp)
            got = run_task(model, tok, task, inp, device)
            if got == expected:
                correct += 1
            if k == 0:
                examples[task] = {"input": inp, "expected": expected, "got": got}
        results[task] = correct / n
    return results, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--num-examples", type=int, default=5000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the training corpus (regenerate deterministically for the demo).
    gen = random.Random(0)
    tasks = list(generate_data.TASKS)
    text = "".join(
        generate_data.make_example(gen.choice(tasks), generate_data.random_input(gen))
        for _ in range(args.num_examples)
    )
    alphabet = sorted(set(text))
    tok = CharTokenizer(alphabet)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.block_size,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {n_params:,} | vocab={tok.vocab_size}")
    print(f"Tasks embedded in text: {tasks}")
    print(f"Config: d_model={cfg.d_model} layers={cfg.num_layers} heads={cfg.num_heads}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()
    loss_curve = []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_lm_batch(data, args.block_size, args.batch_size, device, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f} | ppl {math.exp(min(loss.item(),20)):6.2f}")
            loss_curve.append({"step": step, "loss": round(loss.item(), 4)})
    print(f"\nLM training done in {time.time() - start:.1f}s (next-token objective only)\n")

    # ---- Zero-shot evaluation ---------------------------------------------
    print("== Zero-shot evaluation on unseen inputs (no fine-tuning) ==")
    eval_rng = random.Random(123)
    results, examples = evaluate(model, tok, device, eval_rng, n=60)
    for task in tasks:
        ex = examples[task]
        print(f"  {task:>8}: acc {results[task]*100:5.1f}%   e.g. '{ex['input']}' -> "
              f"'{ex['got']}' (want '{ex['expected']}')")
    mean_acc = sum(results.values()) / len(results)
    print(f"\nMean zero-shot accuracy across tasks: {mean_acc*100:.1f}%")

    # ---- Capture an attention heatmap for the viz -------------------------
    with torch.no_grad():
        sample_text = "reverse: cafe = "
        ids = torch.tensor([tok.encode(sample_text)], device=device)
        model(ids)
        attn = model.blocks[-1].attn.attn_weights[0].mean(0)
    payload = {
        "tasks": tasks,
        "accuracies": {k: round(v, 4) for k, v in results.items()},
        "examples": examples,
        "mean_accuracy": round(mean_acc, 4),
        "loss_curve": loss_curve,
        "attn_text": sample_text,
        "attn_labels": list(sample_text),
        "causal_attention": [[round(v, 4) for v in row] for row in attn.tolist()],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "gpt2_sample.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote data/gpt2_sample.json for the visualization.")

    if mean_acc < 0.8:
        raise SystemExit(f"Zero-shot accuracy too low ({mean_acc*100:.1f}% < 80%).")
    print("OK: a single next-token LM performs every task zero-shot.")


if __name__ == "__main__":
    main()
