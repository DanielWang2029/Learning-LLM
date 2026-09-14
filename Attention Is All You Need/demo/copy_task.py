"""Train the Transformer on a toy sequence-copy task.

This is the standard sanity check for a sequence-to-sequence Transformer:
given a random sequence of tokens, the model must reproduce it exactly.
With the default settings it trains to ~100% exact-sequence accuracy in
roughly half a minute on CPU, demonstrating that the implementation in
`transformer/` — and the development environment — works end to end.

Run with:  python demo/copy_task.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Make the paper's `transformer/` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformer import Transformer
from transformer.model import subsequent_mask

PAD, BOS, EOS = 0, 1, 2
NUM_SPECIAL = 3  # reserved token ids: PAD, BOS, EOS


def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Create a batch for the copy task.

    src:      [content...]                       (length seq_len)
    tgt_in:   [BOS, content...]                  (decoder input, teacher forced)
    tgt_out:  [content..., EOS]                  (expected next tokens)
    """
    content = torch.randint(
        NUM_SPECIAL, vocab_size, (batch_size, seq_len), device=device
    )
    src = content
    bos = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    eos = torch.full((batch_size, 1), EOS, dtype=torch.long, device=device)
    tgt_in = torch.cat([bos, content], dim=1)
    tgt_out = torch.cat([content, eos], dim=1)
    return src, tgt_in, tgt_out


def make_masks(src: torch.Tensor, tgt_in: torch.Tensor):
    # No padding in this toy task, so the source mask attends everywhere.
    src_mask = torch.ones(
        src.size(0), 1, src.size(1), dtype=torch.bool, device=src.device
    )
    tgt_mask = subsequent_mask(tgt_in.size(1), device=tgt_in.device)
    return src_mask, tgt_mask


def evaluate(model, batch_size, seq_len, vocab_size, device) -> float:
    """Return exact-sequence accuracy via greedy autoregressive decoding."""
    model.eval()
    src, _, tgt_out = make_batch(batch_size, seq_len, vocab_size, device)
    src_mask = torch.ones(
        src.size(0), 1, src.size(1), dtype=torch.bool, device=device
    )
    decoded = model.greedy_decode(
        src, src_mask, max_len=seq_len + 1, start_symbol=BOS
    )
    pred = decoded[:, 1 : seq_len + 1]  # drop BOS, compare content region
    correct = (pred == tgt_out[:, :seq_len]).all(dim=1).float().mean().item()
    return correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model parameters: {num_params:,}")
    print(
        f"Config: d_model={args.d_model} layers={args.num_layers} "
        f"heads={args.num_heads} d_ff={args.d_ff} | task: copy "
        f"seq_len={args.seq_len} vocab={args.vocab_size}\n"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(
            args.batch_size, args.seq_len, args.vocab_size, device
        )
        src_mask, tgt_mask = make_masks(src, tgt_in)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, args.vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            acc = evaluate(model, 200, args.seq_len, args.vocab_size, device)
            print(
                f"step {step:4d}/{args.steps} | loss {loss.item():.4f} "
                f"| seq-acc {acc * 100:5.1f}%"
            )

    elapsed = time.time() - start
    final_acc = evaluate(model, 500, args.seq_len, args.vocab_size, device)
    print(f"\nTrained {args.steps} steps in {elapsed:.1f}s")
    print(f"Final exact-sequence accuracy: {final_acc * 100:.1f}%")

    # Show one concrete example so the copy behaviour is visible.
    model.eval()
    src, _, _ = make_batch(1, args.seq_len, args.vocab_size, device)
    src_mask = torch.ones(1, 1, args.seq_len, dtype=torch.bool, device=device)
    decoded = model.greedy_decode(
        src, src_mask, max_len=args.seq_len + 1, start_symbol=BOS
    )
    print("\nExample")
    print(f"  input : {src[0].tolist()}")
    print(f"  output: {decoded[0, 1:].tolist()}")

    if final_acc < 0.9:
        raise SystemExit(
            f"Copy task did not converge (accuracy {final_acc * 100:.1f}% < 90%)."
        )
    print("\nOK: model learned to copy sequences.")


if __name__ == "__main__":
    main()
