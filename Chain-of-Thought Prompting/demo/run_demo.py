"""End-to-end Chain-of-Thought demo (CPU, < ~60s).

Trains two copies of the SAME tiny Transformer on multi-digit addition:

* a DIRECT model, trained to emit the answer immediately ("12+345=357"), and
* a CHAIN-OF-THOUGHT model, trained to emit a digit-by-digit scratchpad with
  explicit carries before the answer.

It then measures exact-match accuracy on a held-out set (overall and bucketed
by problem size) and prints a worked scratchpad example. Results are written to
``data/cot_results.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

torch.set_num_threads(1)  # shared 4-core box: avoid thread oversubscription

# Make the paper's `src/` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import CharVocab, TinyGPT, format_example, make_dataset, parse_answer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def encode_examples(problems, mode, vocab):
    """Return list of (token_ids, prompt_len) for each problem in a mode."""
    rows = []
    for a, b in problems:
        prompt, target = format_example(a, b, mode)
        ids = vocab.encode(prompt + target)
        rows.append((ids, len(prompt)))
    return rows


def make_batch(rows, idxs, device):
    """Pad a minibatch and build inputs/targets with a loss mask over targets."""
    seqs = [rows[i][0] for i in idxs]
    plens = [rows[i][1] for i in idxs]
    maxlen = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.zeros(B, maxlen - 1, dtype=torch.long, device=device)
    y = torch.zeros(B, maxlen - 1, dtype=torch.long, device=device)
    mask = torch.zeros(B, maxlen - 1, dtype=torch.bool, device=device)
    ar = torch.arange(maxlen - 1, device=device)
    for bi, (s, plen) in enumerate(zip(seqs, plens)):
        t = torch.tensor(s, dtype=torch.long, device=device)
        L = t.numel()
        x[bi, : L - 1] = t[:-1]
        y[bi, : L - 1] = t[1:]
        # Only score positions whose *target* falls in the completion region.
        mask[bi] = (ar + 1 >= plen) & (ar < L - 1)
    return x, y, mask


@torch.no_grad()
def exact_match(model, problems, mode, vocab, device, max_new_tokens):
    """Greedy-decode each prompt and return per-problem correctness (0/1).

    We group problems by prompt length and decode each group as its own
    (unpadded) batch. This keeps absolute positions aligned with training —
    left-padding a decoder-only model would shift every token's position.
    """
    model.eval()
    by_len: dict[int, list[int]] = {}
    prompts = [format_example(a, b, mode)[0] for a, b in problems]
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)

    results = [0] * len(problems)
    for plen, group in by_len.items():
        idx = torch.tensor(
            [vocab.encode(prompts[i]) for i in group], dtype=torch.long, device=device
        )
        out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=vocab.eos_id)
        gen = out[:, plen:]
        for row, i in enumerate(group):
            a, b = problems[i]
            text = vocab.decode(gen[row].tolist())
            results[i] = int(parse_answer(text, mode) == str(a + b))
    return results


def train_model(mode, train_problems, eval_problems, vocab, args, device, curve_steps):
    torch.manual_seed(args.seed)
    rows = encode_examples(train_problems, mode, vocab)
    block_size = max(len(r[0]) for r in rows) + 2
    model = TinyGPT(
        vocab_size=len(vocab),
        block_size=block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    n = len(rows)
    curve = []
    max_new = block_size
    g = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        model.train()
        idxs = torch.randint(0, n, (args.batch_size,), generator=g).tolist()
        x, y, mask = make_batch(rows, idxs, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        )
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in curve_steps:
            acc = sum(
                exact_match(model, eval_problems[: args.curve_eval], mode, vocab, device, max_new)
            ) / args.curve_eval
            curve.append({"step": step, "acc": round(acc, 4)})
            print(f"  [{mode:6s}] step {step:4d}/{args.steps} | loss {loss.item():.3f} "
                  f"| held-out exact-match {acc*100:5.1f}%")
    return model, curve, max_new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--curve-eval", type=int, default=200)
    p.add_argument("--min-digits", type=int, default=1)
    p.add_argument("--max-digits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cpu")
    vocab = CharVocab()

    train_problems = make_dataset(args.n_train, args.min_digits, args.max_digits, seed=1)
    eval_problems = make_dataset(args.n_eval, args.min_digits, args.max_digits, seed=999)
    # de-duplicate the eval set against training
    train_set = set(train_problems)
    eval_problems = [pb for pb in eval_problems if pb not in train_set]

    curve_steps = set(range(200, args.steps + 1, 200)) | {args.steps}

    print(f"Device: cpu | vocab={len(vocab)} | train={len(train_problems)} "
          f"eval={len(eval_problems)} | digits {args.min_digits}-{args.max_digits}\n")

    start = time.time()
    results = {}
    curves = {}
    max_new_by_mode = {}
    for mode in ("direct", "cot"):
        print(f"Training {mode.upper()} model...")
        model, curve, max_new = train_model(
            mode, train_problems, eval_problems, vocab, args, device, curve_steps
        )
        results[mode] = model
        curves[mode] = curve
        max_new_by_mode[mode] = max_new

    elapsed = time.time() - start

    # Final evaluation: overall + bucketed by max operand digit-length.
    def digits(pb):
        return max(len(str(pb[0])), len(str(pb[1])))

    overall = {}
    by_digits = {str(d): {} for d in range(args.min_digits, args.max_digits + 1)}
    res_by_mode = {}
    for mode in ("direct", "cot"):
        res = exact_match(results[mode], eval_problems, mode, vocab, device, max_new_by_mode[mode])
        res_by_mode[mode] = res
        overall[mode] = round(sum(res) / len(res), 4)
        for d in range(args.min_digits, args.max_digits + 1):
            sub = [r for r, pb in zip(res, eval_problems) if digits(pb) == d]
            if sub:
                by_digits[str(d)][mode] = round(sum(sub) / len(sub), 4)

    # A concrete worked example: prefer a hard (max-digit) held-out problem
    # that CoT solves but DIRECT gets wrong, for the cleanest contrast.
    hard_idx = [
        i for i, pb in enumerate(eval_problems) if digits(pb) == args.max_digits
    ]
    clean = [i for i in hard_idx if res_by_mode["cot"][i] and not res_by_mode["direct"][i]]
    chosen = clean[0] if clean else hard_idx[0]
    hard = eval_problems[chosen]
    ex = {}
    for mode in ("direct", "cot"):
        prompt = format_example(*hard, mode)[0]
        ids = torch.tensor([vocab.encode(prompt)], device=device)
        gen = results[mode].generate(ids, max_new_tokens=max_new_by_mode[mode], eos_id=vocab.eos_id)
        text = vocab.decode(gen[0, len(prompt):].tolist()).split("$")[0]
        ex[mode] = {"completion": text, "answer": parse_answer(text, mode)}

    print(f"\nTrained both models in {elapsed:.1f}s")
    print("\n================ EXACT-MATCH ACCURACY (held-out) ================")
    print(f"  DIRECT : {overall['direct']*100:5.1f}%")
    print(f"  CoT    : {overall['cot']*100:5.1f}%")
    print("\n  by max operand digit-length:")
    for d in by_digits:
        row = by_digits[d]
        print(f"    {d}-digit  direct {row.get('direct',0)*100:5.1f}%   "
              f"cot {row.get('cot',0)*100:5.1f}%")

    print(f"\n  Worked example  {hard[0]}+{hard[1]} = {hard[0]+hard[1]}")
    print(f"    direct -> {ex['direct']['completion']}   (answer {ex['direct']['answer']})")
    print(f"    cot    -> {ex['cot']['completion']}")
    print(f"             answer {ex['cot']['answer']}")

    out = {
        "config": {
            "steps": args.steps, "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "digits": [args.min_digits, args.max_digits],
            "n_train": len(train_problems), "n_eval": len(eval_problems),
            "seconds": round(elapsed, 1),
        },
        "overall": overall,
        "by_digits": by_digits,
        "curve": {
            "steps": [c["step"] for c in curves["direct"]],
            "direct": [c["acc"] for c in curves["direct"]],
            "cot": [c["acc"] for c in curves["cot"]],
        },
        "example": {
            "problem": f"{hard[0]}+{hard[1]}",
            "answer": str(hard[0] + hard[1]),
            "direct": ex["direct"],
            "cot": ex["cot"],
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "cot_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if overall["cot"] <= overall["direct"]:
        raise SystemExit("Chain-of-thought did not beat direct — check training.")
    print("\nOK: chain-of-thought scratchpad beats direct answering.")


if __name__ == "__main__":
    main()
