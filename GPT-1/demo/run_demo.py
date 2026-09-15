"""GPT-1 end-to-end demo: generative pre-training then discriminative fine-tuning.

Runs on CPU in well under a minute and shows GPT-1's central claim — that
unsupervised pre-training gives features that transfer to a supervised task:

1. **Generative pre-training.** Train the decoder-only LM to predict the next
   character on an unlabeled two-topic corpus. Prints the loss / perplexity
   dropping and generates a short continuation of a prompt.
2. **Discriminative fine-tuning.** Attach a linear head to the final token and
   fine-tune on a *small* labeled topic-classification set. We do this twice —
   once starting from the pre-trained weights, once from scratch — and show the
   pre-trained model reaches much higher accuracy from the same few labels.

Writes ``data/gpt1_sample.json`` for the visualization.

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

from src import GPT, GPTConfig, GPTClassifier  # noqa: E402
import generate_data  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
PAD = 0


class CharTokenizer:
    def __init__(self, alphabet):
        self.itos = {0: "<pad>"}
        self.stoi = {"<pad>": 0}
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


def pad_batch(examples, tokenizer, device):
    ids = [tokenizer.encode(e["text"]) for e in examples]
    lengths = torch.tensor([len(i) for i in ids], device=device)
    maxlen = max(len(i) for i in ids)
    padded = torch.full((len(ids), maxlen), PAD, dtype=torch.long)
    for r, seq in enumerate(ids):
        padded[r, : len(seq)] = torch.tensor(seq)
    labels = torch.tensor([e["label"] for e in examples], device=device)
    return padded.to(device), lengths, labels


def finetune(gpt, train, test, tokenizer, device, steps, lr, batch_size, rng, tag):
    clf = GPTClassifier(gpt, num_classes=2).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    xb, lb, yb = pad_batch(test, tokenizer, device)
    for step in range(1, steps + 1):
        clf.train()
        batch = [train[rng.randrange(len(train))] for _ in range(batch_size)]
        x, lengths, y = pad_batch(batch, tokenizer, device)
        _, loss = clf(x, lengths, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        logits, _ = clf(xb, lb, None)
        acc = (logits.argmax(-1) == yb).float().mean().item()
    print(f"  [{tag}] fine-tune {steps} steps -> test accuracy {acc * 100:.1f}%")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--finetune-steps", type=int, default=60)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_path = DATA_DIR / "corpus.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text())
    else:
        # Build the corpus inline (deterministic) if it has not been generated.
        gen = random.Random(0)
        pretrain = "".join(generate_data.make_sentence(gen.randint(0, 1), gen) for _ in range(600))
        labeled = [
            {"text": generate_data.make_sentence(lbl, gen).strip(), "label": lbl}
            for lbl in (gen.randint(0, 1) for _ in range(400))
        ]
        corpus = {"alphabet": sorted(set(pretrain)), "pretrain_text": pretrain, "classification": labeled}

    tok = CharTokenizer(corpus["alphabet"])
    data = torch.tensor(tok.encode(corpus["pretrain_text"]), dtype=torch.long)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.block_size,
    )
    gpt = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in gpt.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {n_params:,} | vocab={tok.vocab_size}")
    print(f"Config: d_model={cfg.d_model} layers={cfg.num_layers} heads={cfg.num_heads}\n")

    # ---- Stage 1: generative pre-training ---------------------------------
    print("== Stage 1: generative pre-training (next-char prediction) ==")
    opt = torch.optim.AdamW(gpt.parameters(), lr=args.lr)
    start = time.time()
    loss_curve = []
    first_ppl = None
    for step in range(1, args.pretrain_steps + 1):
        gpt.train()
        x, y = get_lm_batch(data, args.block_size, args.batch_size, device, rng)
        _, loss = gpt(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            ppl = math.exp(min(loss.item(), 20))
            loss_curve.append({"step": step, "loss": round(loss.item(), 4), "ppl": round(ppl, 3)})
            if first_ppl is None:
                first_ppl = ppl
            print(f"step {step:4d}/{args.pretrain_steps} | loss {loss.item():.4f} | ppl {ppl:6.2f}")
    final_ppl = math.exp(min(loss.item(), 20))
    print(f"\nPre-training done in {time.time() - start:.1f}s | perplexity {first_ppl:.1f} -> {final_ppl:.2f}")

    # ---- Generation from a prompt -----------------------------------------
    prompt = "the "
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = gpt.generate(ids, max_new_tokens=40, greedy=True)
    generated = tok.decode(out[0].tolist())
    print(f"\nGreedy continuation of {prompt!r}:\n  {generated!r}")

    # ---- Stage 2: discriminative fine-tuning ------------------------------
    print("\n== Stage 2: discriminative fine-tuning (topic classification) ==")
    labeled = list(corpus["classification"])
    rng.shuffle(labeled)
    train = labeled[: args.n_train]
    test = labeled[args.n_train : args.n_train + args.n_test]
    print(f"labeled: {len(train)} train / {len(test)} test (task: animals vs plants)")

    # Snapshot pre-trained weights so both runs start fairly.
    pretrained_state = {k: v.clone() for k, v in gpt.state_dict().items()}

    acc_pre = finetune(gpt, train, test, tok, device, args.finetune_steps, args.finetune_lr, 16, random.Random(1), "pretrained")

    scratch = GPT(cfg).to(device)  # fresh random init
    acc_scratch = finetune(scratch, train, test, tok, device, args.finetune_steps, args.finetune_lr, 16, random.Random(1), "from-scratch")

    print(f"\nTransfer gain: pre-trained {acc_pre * 100:.1f}%  vs  scratch {acc_scratch * 100:.1f}%  "
          f"(+{(acc_pre - acc_scratch) * 100:.1f} pts)")

    # ---- Capture a causal attention heatmap for the viz -------------------
    gpt.load_state_dict(pretrained_state)
    gpt.eval()
    with torch.no_grad():
        sample_text = "the cat runs today ."
        ids = torch.tensor([tok.encode(sample_text)], device=device)
        gpt(ids)
        attn = gpt.blocks[-1].attn.attn_weights[0].mean(0)  # (t, t)
    payload = {
        "prompt": prompt,
        "generated": generated,
        "loss_curve": loss_curve,
        "first_ppl": round(first_ppl, 3),
        "final_ppl": round(final_ppl, 3),
        "acc_pretrained": round(acc_pre, 4),
        "acc_scratch": round(acc_scratch, 4),
        "attn_text": sample_text,
        "attn_labels": list(sample_text),
        "causal_attention": [[round(v, 4) for v in row] for row in attn.tolist()],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "gpt1_sample.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote data/gpt1_sample.json for the visualization.")

    if not (final_ppl < first_ppl and acc_pre >= acc_scratch + 0.1 and acc_pre >= 0.8):
        raise SystemExit("Demo did not meet success criteria (perplexity drop + transfer gain).")
    print("OK: pre-training lowered perplexity and transferred to the downstream task.")


if __name__ == "__main__":
    main()
