"""End-to-end masked-language-modelling demo for the tiny BERT encoder.

What it does (all on CPU, well under a minute):

1. Builds a synthetic "successor grammar" corpus (see data/generate_data.py).
2. Trains the bidirectional encoder with the masked-LM objective: ~15% of the
   tokens in each sentence are corrupted (80% -> [MASK], 10% -> random,
   10% kept) and the model must predict their original identity.
3. Prints the masked-token accuracy climbing from chance to near-perfect.
4. Shows a few worked examples (masked input -> predicted tokens).
5. Captures a real self-attention heatmap and writes it to
   ``data/mlm_sample.json`` for the visualization.

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

from src.model import BertModel, CLS, MASK, NUM_SPECIAL, PAD  # noqa: E402
import generate_data  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
IGNORE = -100


def make_batch(succ, batch_size, seq_len, rng: random.Random, device):
    """Sample sentences, prepend [CLS], and apply BERT-style MLM corruption."""
    content_ids = list(succ.keys())
    tokens, labels = [], []
    for _ in range(batch_size):
        sent = generate_data.sample_sentence(succ, seq_len, rng)
        toks = [CLS] + sent
        lab = [IGNORE] * len(toks)
        for i in range(1, len(toks)):  # never mask [CLS]
            if rng.random() < 0.15:
                lab[i] = toks[i]  # remember the original id
                r = rng.random()
                if r < 0.8:
                    toks[i] = MASK
                elif r < 0.9:
                    toks[i] = rng.choice(content_ids)  # random replacement
                # else: keep the token unchanged
        # Guarantee at least one masked target per sentence.
        if all(x == IGNORE for x in lab):
            j = rng.randint(1, len(toks) - 1)
            lab[j] = sent[j - 1]
            toks[j] = MASK
        tokens.append(toks)
        labels.append(lab)
    return (
        torch.tensor(tokens, device=device),
        torch.tensor(labels, device=device),
    )


def masked_accuracy(logits, labels) -> float:
    mask = labels != IGNORE
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(-1)
    return (pred[mask] == labels[mask]).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--n-content", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    succ = generate_data.build_grammar(args.n_content, rng)
    vocab_size = NUM_SPECIAL + args.n_content

    model = BertModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.seq_len + 1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {n_params:,}")
    print(
        f"Config: d_model={args.d_model} layers={args.num_layers} "
        f"heads={args.num_heads} | vocab={vocab_size} seq_len={args.seq_len}"
    )
    print(f"Chance masked-token accuracy ~ {100.0 / args.n_content:.1f}%\n")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        tokens, labels = make_batch(succ, args.batch_size, args.seq_len, rng, device)
        logits = model(tokens)
        loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                t_eval, l_eval = make_batch(succ, 128, args.seq_len, rng, device)
                acc = masked_accuracy(model(t_eval), l_eval)
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f} | masked-acc {acc * 100:5.1f}%")

    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        t_eval, l_eval = make_batch(succ, 256, args.seq_len, rng, device)
        final_acc = masked_accuracy(model(t_eval), l_eval)
    print(f"\nTrained {args.steps} steps in {elapsed:.1f}s")
    print(f"Final masked-token accuracy: {final_acc * 100:.1f}%")

    # ---- A few worked examples --------------------------------------------
    print("\nExample masked-LM predictions ([M] = masked position):")
    with torch.no_grad():
        toks, labs = make_batch(succ, 3, args.seq_len, rng, device)
        preds = model(toks).argmax(-1)
        for b in range(3):
            shown, truth, guess = [], [], []
            for i in range(toks.size(1)):
                if labs[b, i].item() != IGNORE:
                    shown.append("[M]")
                    truth.append(str(labs[b, i].item()))
                    guess.append(str(preds[b, i].item()))
                else:
                    shown.append(str(toks[b, i].item()))
            print(f"  input : {' '.join(shown)}")
            print(f"  truth : {' '.join(truth)}   pred : {' '.join(guess)}")

    # ---- Capture a self-attention heatmap for the visualization -----------
    with torch.no_grad():
        sent = generate_data.sample_sentence(succ, 8, rng)
        seq = [CLS] + sent
        toks = torch.tensor([seq], device=device)
        model(toks)
        # Mean over heads of the last encoder layer's self-attention.
        attn = model.encoder.layers[-1].self_attn.attn_weights[0].mean(0)
    labels_str = ["[CLS]"] + [str(t) for t in sent]
    sample = {
        "tokens": seq,
        "labels": labels_str,
        "self_attention": [[round(v, 4) for v in row] for row in attn.tolist()],
        "final_masked_accuracy": round(final_acc, 4),
        "vocab_size": vocab_size,
        "n_content": args.n_content,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "mlm_sample.json").write_text(json.dumps(sample, indent=2))
    print("\nWrote data/mlm_sample.json (self-attention heatmap for the viz).")

    if final_acc < 0.9:
        raise SystemExit(f"MLM did not converge (accuracy {final_acc * 100:.1f}% < 90%).")
    print("OK: bidirectional encoder learned to fill in masked tokens.")


if __name__ == "__main__":
    main()
