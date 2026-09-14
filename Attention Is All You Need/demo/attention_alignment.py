"""Train briefly on the copy task and export the attention the model learns.

The interactive visualization (``visualization/index.html``) shows two
attention heatmaps. Untrained attention is just noise, so this script trains a
small Transformer for a short while until it solves the copy task, then dumps
the *learned* attention for one fixed example to
``data/trained_attention_sample.json`` in the exact shape the visualization
expects:

* ``cross_attention``        - decoder→encoder attention (shows the diagonal
  input↔output alignment the model discovers), and
* ``encoder_self_attention`` - encoder layer-0 self-attention.

When you serve this folder over HTTP, the visualization fetches this file and
renders live numbers; otherwise it falls back to a baked-in copy.

Run with:  python demo/attention_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from transformer import Transformer  # noqa: E402
from transformer.model import subsequent_mask  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
NUM_SPECIAL = 3
DATA_DIR = PAPER_DIR / "data"


def make_batch(bs: int, seq_len: int, vocab: int):
    content = torch.randint(NUM_SPECIAL, vocab, (bs, seq_len))
    tgt_in = torch.cat([torch.full((bs, 1), BOS), content], dim=1)
    tgt_out = torch.cat([content, torch.full((bs, 1), EOS)], dim=1)
    return content, tgt_in, tgt_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--example", type=int, nargs="+", default=[5, 8, 3, 9, 7, 4])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    V, L = args.vocab_size, args.seq_len

    model = Transformer(
        V, V, d_model=64, num_layers=2, num_heads=4, d_ff=128, dropout=0.1
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()

    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(64, L, V)
        src_mask = torch.ones(64, 1, L, dtype=torch.bool)
        tgt_mask = subsequent_mask(tgt_in.size(1))
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 400 == 0 or step == 1:
            print(f"step {step:4d}/{args.steps} | loss {loss.item():.4f}")

    model.eval()
    seq = args.example[:L]
    src = torch.tensor([seq])
    src_mask = torch.ones(1, 1, L, dtype=torch.bool)
    decoded = model.greedy_decode(src, src_mask, max_len=L + 1, start_symbol=BOS)

    # Teacher-forced pass to capture attention against the true target.
    tgt_in = torch.tensor([[BOS] + seq])
    tgt_mask = subsequent_mask(tgt_in.size(1))
    memory = model.encode(src, src_mask)
    model.decode(tgt_in, memory, src_mask, tgt_mask)

    cross = model.decoder.layers[-1].cross_attn.attn_weights[0].mean(0)  # tgt x src
    enc_self = model.encoder.layers[0].self_attn.attn_weights[0].mean(0)  # src x src

    def r(mat):
        return [[round(v, 4) for v in row] for row in mat.tolist()]

    out = {
        "source": seq,
        "decoded": decoded[0, 1:].tolist(),
        "src_labels": [str(x) for x in seq],
        "tgt_labels": ["<b>"] + [str(x) for x in seq],
        "cross_attention": r(cross),
        "encoder_self_attention": r(enc_self),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "trained_attention_sample.json").write_text(json.dumps(out, indent=2))

    ok = out["decoded"] == seq
    print(f"\nsource : {seq}")
    print(f"output : {out['decoded']}  {'(exact match)' if ok else '(not yet exact)'}")
    print("Wrote data/trained_attention_sample.json")


if __name__ == "__main__":
    main()
