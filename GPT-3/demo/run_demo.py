"""GPT-3 in-context few-shot learning demo.

A small decoder-only LM is trained on prompts that each contain a run of
``x>y`` demonstration pairs from one rule (a permutation) plus a query. The
rules overlap, so the model must *infer* which rule the demonstrations imply
and apply it to a new query symbol — it cannot simply copy an answer.

After training (weights frozen), we measure accuracy on held-out prompts as a
function of the number of demonstrations K (the "shots"). Accuracy climbs from
near chance at K=0 toward near-perfect as K grows — in-context learning with no
weight updates. The accuracy-vs-shots curve is written to
``data/gpt3_sample.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "data"))

from src import GPT, GPTConfig  # noqa: E402
import generate_data as gd  # noqa: E402

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


def make_batch(tasks, batch_size, kmax, tok, max_len, device, rng):
    x = torch.full((batch_size, max_len), PAD, dtype=torch.long)
    for i in range(batch_size):
        task = rng.choice(tasks)
        k = rng.randint(0, kmax)
        seq = tok.encode(gd.make_sequence(task, k, rng))[:max_len]
        x[i, : len(seq)] = torch.tensor(seq)
    x = x.to(device)
    inp, tgt = x[:, :-1], x[:, 1:].clone()
    tgt[inp == PAD] = -100  # ignore predictions made from padding
    return inp, tgt


def build_prompt(task, k, tok, rng):
    syms = list(gd.SYMBOLS)
    rng.shuffle(syms)
    demo_inputs = syms[:k]
    query = rng.choice(syms[k:]) if k < len(syms) else rng.choice(syms)
    pairs = [gd.format_pair(task, x) for x in demo_inputs]
    prompt = (gd.PAIR_SEP.join(pairs) + gd.PAIR_SEP if pairs else "") + f"{query}{gd.MAP}"
    return prompt, task[query]


@torch.no_grad()
def accuracy_at_k(model, tasks, k, tok, device, rng, trials=200):
    correct = 0
    for _ in range(trials):
        task = rng.choice(tasks)
        prompt, expected = build_prompt(task, k, tok, rng)
        ids = torch.tensor([tok.encode(prompt)], device=device)
        logits = model.next_token_logits(ids)
        pred = tok.itos[int(logits.argmax().item())]
        correct += int(pred == expected)
    return correct / trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = gd.build_tasks(args.n_tasks, seed=7)
    alphabet = sorted(set(gd.SYMBOLS + gd.MAP + gd.PAIR_SEP + gd.END))
    tok = CharTokenizer(alphabet)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {n_params:,} | vocab={tok.vocab_size}")
    print(f"Rule family: {len(tasks)} overlapping permutations over {len(gd.SYMBOLS)} symbols")
    print(f"Chance accuracy ~ {100.0 / len(gd.SYMBOLS):.1f}%\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        inp, tgt = make_batch(tasks, args.batch_size, args.kmax, tok, args.max_len, device, rng)
        _, loss = model(inp, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f}")
    print(f"\nTraining done in {time.time() - start:.1f}s\n")

    # ---- Accuracy vs number of shots K ------------------------------------
    print("== In-context few-shot accuracy (no weight updates) ==")
    eval_rng = random.Random(2024)
    curve = []
    for k in range(0, args.kmax + 1):
        acc = accuracy_at_k(model, tasks, k, tok, device, eval_rng, trials=300)
        bar = "#" * int(acc * 40)
        print(f"  K={k}  acc {acc*100:5.1f}%  |{bar}")
        curve.append({"shots": k, "accuracy": round(acc, 4)})

    acc0 = curve[0]["accuracy"]
    accmax = curve[-1]["accuracy"]
    print(f"\n0-shot {acc0*100:.1f}%  ->  {args.kmax}-shot {accmax*100:.1f}%  "
          f"(+{(accmax-acc0)*100:.1f} pts from context alone)")

    # ---- A worked few-shot example ----------------------------------------
    demo_rng = random.Random(5)
    task = demo_rng.choice(tasks)
    prompt, expected = build_prompt(task, args.kmax, tok, demo_rng)
    ids = torch.tensor([tok.encode(prompt)], device=device)
    pred = tok.itos[int(model.next_token_logits(ids).argmax().item())]
    print(f"\nExample {args.kmax}-shot prompt: {prompt!r}")
    print(f"  model predicts '{pred}'  (correct answer '{expected}')")

    payload = {
        "n_tasks": len(tasks),
        "n_symbols": len(gd.SYMBOLS),
        "chance": round(1.0 / len(gd.SYMBOLS), 4),
        "curve": curve,
        "example_prompt": prompt,
        "example_pred": pred,
        "example_expected": expected,
        "tasks": [dict(t) for t in tasks],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "gpt3_sample.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote data/gpt3_sample.json for the visualization.")

    if not (accmax >= acc0 + 0.2 and accmax >= 0.8):
        raise SystemExit("Few-shot curve did not rise enough to demonstrate in-context learning.")
    print("OK: accuracy rises with the number of in-context demonstrations.")


if __name__ == "__main__":
    main()
