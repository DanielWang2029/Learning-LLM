"""T5 end-to-end demo: span-corruption pre-training + text-to-text multitask.

Runs on CPU in under a minute:

1. **Span-corruption pre-training.** Random spans of toy sentences are replaced
   by sentinel tokens; the single encoder–decoder model learns to reconstruct
   the dropped spans. Prints the reconstruction accuracy climbing.
2. **Text-to-text multitask.** The *same* model is then trained on three tasks
   framed identically as "input string -> output string" and distinguished only
   by a task token: ``copy``, ``reverse`` and ``sort``. Prints each task's
   exact-match accuracy (via greedy decoding) reaching a high value.

Writes ``data/t5_sample.json`` (per-task accuracy + a cross-attention heatmap)
for the visualization.

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
import torch.nn as nn

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "data"))

from src import T5, BOS, EOS, PAD  # noqa: E402
import generate_data as gd  # noqa: E402

DATA_DIR = PAPER_DIR / "data"


def pad(seqs, device):
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out.to(device)


def batch_from_pairs(pairs, device):
    """pairs: list of (encoder_input, target). Returns src, tgt_in, tgt_out."""
    src = pad([p[0] for p in pairs], device)
    tgt_in = pad([[BOS] + p[1][:-1] for p in pairs], device)
    tgt_out = pad([p[1] for p in pairs], device)
    return src, tgt_in, tgt_out


def span_batch(succ, bs, length, rng, device):
    seqs = [gd.sample_text(succ, length, rng) for _ in range(bs)]
    return batch_from_pairs([gd.corrupt_spans(s, rng) for s in seqs], device)


def task_batch(succ, bs, length, rng, device, tasks):
    pairs = []
    for _ in range(bs):
        t = rng.choice(tasks)
        seq = [rng.randrange(gd.CONTENT_START, gd.CONTENT_START + N_CONTENT) for _ in range(length)]
        pairs.append(gd.task_example(t, seq))
    return batch_from_pairs(pairs, device)


def train_loop(model, batch_fn, steps, lr, log_every, tag):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    curve = []
    for step in range(1, steps + 1):
        model.train()
        src, tgt_in, tgt_out = batch_fn()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  [{tag}] step {step:4d}/{steps} | loss {loss.item():.4f}")
            curve.append({"step": step, "loss": round(loss.item(), 4)})
    return curve


@torch.no_grad()
def span_reconstruction_acc(model, succ, rng, device, n=128, length=12):
    src, tgt_in, tgt_out = span_batch(succ, n, length, rng, device)
    logits = model(src, tgt_in)
    pred = logits.argmax(-1)
    mask = tgt_out != PAD
    return (pred[mask] == tgt_out[mask]).float().mean().item()


@torch.no_grad()
def task_accuracy(model, succ, rng, device, task, n=120, length=6):
    correct = 0
    example = None
    for k in range(n):
        seq = [rng.randrange(gd.CONTENT_START, gd.CONTENT_START + N_CONTENT) for _ in range(length)]
        enc, target = gd.task_example(task, seq)
        src = pad([enc], device)
        decoded = model.greedy_decode(src, max_len=length + 2, device=device)[0, 1:].tolist()
        # trim at EOS
        if EOS in decoded:
            decoded = decoded[: decoded.index(EOS)]
        expected = target[:-1]
        if decoded == expected:
            correct += 1
        if k == 0:
            example = {"input": seq, "expected": expected, "got": decoded}
    return correct / n, example


N_CONTENT = 20


def main() -> None:
    global N_CONTENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-steps", type=int, default=800)
    parser.add_argument("--task-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-content", type=int, default=20)
    parser.add_argument("--text-len", type=int, default=12)
    parser.add_argument("--task-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    N_CONTENT = args.n_content
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    succ = gd.build_grammar(args.n_content, args.seed)
    vocab = gd.vocab_size(args.n_content)
    tasks = list(gd.TASK_TOKENS)

    model = T5(
        vocab_size=vocab,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=max(args.text_len, args.task_len) + 4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {n_params:,} | vocab={vocab}")
    print(f"Config: d_model={args.d_model} layers={args.num_layers} heads={args.num_heads}\n")

    start = time.time()
    print("== Stage 1: span-corruption pre-training ==")
    pre_curve = train_loop(
        model, lambda: span_batch(succ, args.batch_size, args.text_len, rng, device),
        args.pretrain_steps, args.lr, args.log_every, "pretrain",
    )
    recon = span_reconstruction_acc(model, succ, random.Random(7), device)
    print(f"  span reconstruction token accuracy: {recon * 100:.1f}%\n")

    print("== Stage 2: text-to-text multitask (copy / reverse / sort) ==")
    task_curve = train_loop(
        model, lambda: task_batch(succ, args.batch_size, args.task_len, rng, device, tasks),
        args.task_steps, args.lr, args.log_every, "multitask",
    )

    print("\n== Exact-match accuracy per task (greedy decoding) ==")
    eval_rng = random.Random(2024)
    accs, examples = {}, {}
    for t in tasks:
        acc, ex = task_accuracy(model, succ, eval_rng, device, t, length=args.task_len)
        accs[t] = acc
        examples[t] = ex
        print(f"  {t:>7}: {acc*100:5.1f}%   {ex['input']} -> {ex['got']} (want {ex['expected']})")
    mean_acc = sum(accs.values()) / len(accs)
    print(f"\nMean task accuracy: {mean_acc*100:.1f}% | total time {time.time()-start:.1f}s")

    # ---- Capture a cross-attention heatmap for the viz --------------------
    seq = [rng.randrange(gd.CONTENT_START, gd.CONTENT_START + N_CONTENT) for _ in range(args.task_len)]
    enc, target = gd.task_example("reverse", seq)
    src = pad([enc], device)
    tgt_in = pad([[BOS] + target[:-1]], device)
    model.eval()
    with torch.no_grad():
        model(src, tgt_in)
        cross = model.decoder.layers[-1].cross_attn.attn_weights[0].mean(0)  # (tgt, src)
    payload = {
        "tasks": tasks,
        "accuracies": {k: round(v, 4) for k, v in accs.items()},
        "examples": examples,
        "mean_accuracy": round(mean_acc, 4),
        "span_reconstruction_acc": round(recon, 4),
        "pretrain_curve": pre_curve,
        "task_curve": task_curve,
        "cross_attention": {
            "task": "reverse",
            "encoder_input": enc,
            "decoder_input": [BOS] + target[:-1],
            "src_labels": [str(t) for t in enc],
            "tgt_labels": ["<s>"] + [str(t) for t in target[:-1]],
            "matrix": [[round(v, 4) for v in row] for row in cross.tolist()],
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "t5_sample.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote data/t5_sample.json for the visualization.")

    if not (recon >= 0.88 and mean_acc >= 0.9):
        raise SystemExit(f"T5 demo did not converge (recon {recon*100:.1f}%, tasks {mean_acc*100:.1f}%).")
    print("OK: one text-to-text model pre-trained by span corruption solves all three tasks.")


if __name__ == "__main__":
    main()
