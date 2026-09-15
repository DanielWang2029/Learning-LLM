"""Walk a single example through the Transformer and narrate every stage.

Where `copy_task.py` *trains* the model, this script is about *seeing* the
architecture from "Attention Is All You Need". It runs one forward pass
through a small (untrained) Transformer and prints, step by step:

* the tensor shape after each component (embeddings, positional encoding,
  every encoder / decoder layer, the final generator), annotated with the
  section of the paper it comes from, and
* the multi-head attention weight matrices that each attention sub-layer
  actually computed.

It also writes ``data/forward_trace.json`` (shapes + a real attention matrix)
which the HTML visualization can load to display live numbers.

Run with:  python demo/inspect_structure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from transformer import (  # noqa: E402
    Decoder,
    Encoder,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    Transformer,
)
from transformer.model import Embeddings, subsequent_mask  # noqa: E402

PAD, BOS, EOS = 0, 1, 2
DATA_DIR = PAPER_DIR / "data"

# How each module maps onto the paper. Keyed by module class.
SECTION = {
    Embeddings: "3.4  Input embeddings (scaled by sqrt(d_model))",
    PositionalEncoding: "3.5  Positional encoding (added to embeddings)",
    Encoder: "3.1  Encoder stack (N identical layers)",
    Decoder: "3.1  Decoder stack (N identical layers)",
    MultiHeadAttention: "3.2  Multi-head attention",
    PositionwiseFeedForward: "3.3  Position-wise feed-forward network",
}


def _shape(x) -> str:
    if isinstance(x, torch.Tensor):
        return "x".join(str(d) for d in x.shape)
    return type(x).__name__


def ascii_heatmap(matrix: torch.Tensor, labels) -> str:
    """Render a square attention matrix as a shaded ASCII grid."""
    ramp = " .:-=+*#%@"
    m = matrix.detach()
    lo, hi = float(m.min()), float(m.max())
    span = (hi - lo) or 1.0
    header = "        " + " ".join(f"{c:>3}" for c in labels)
    lines = [header]
    for i, row in enumerate(m):
        cells = []
        for val in row:
            level = int((float(val) - lo) / span * (len(ramp) - 1))
            cells.append(ramp[level] * 3)
        lines.append(f"  {labels[i]:>4}  " + " ".join(cells))
    return "\n".join(lines)


def load_example(seq_len: int, vocab_size: int):
    """Load the first reverse-task example if present, else synthesize one."""
    path = DATA_DIR / "reverse_dataset.json"
    if path.exists():
        data = json.loads(path.read_text())
        ex = data["examples"][0]
        src = torch.tensor([ex["source"]], dtype=torch.long)
        tgt_in = torch.tensor([ex["target_in"]], dtype=torch.long)
        return src, tgt_in, "data/reverse_dataset.json"
    content = torch.randint(3, vocab_size, (1, seq_len))
    tgt_in = torch.cat([torch.tensor([[BOS]]), content.flip(1)], dim=1)
    return content, tgt_in, "(synthesized; run generate_demo_data.py for files)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    src, tgt_in, source_desc = load_example(args.seq_len, args.vocab_size)
    seq_len = src.size(1)

    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=0.0,
    )
    model.eval()

    # Record the output shape of every interesting sub-module, in call order.
    trace: list[tuple[str, str, str]] = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            section = SECTION.get(type(module), "")
            trace.append((name, _shape(output), section))
        return hook

    handles = []
    handles.append(model.src_embed.register_forward_hook(make_hook("src_embed")))
    handles.append(model.pos_encoding.register_forward_hook(make_hook("pos_encoding")))
    for i, layer in enumerate(model.encoder.layers):
        handles.append(
            layer.self_attn.register_forward_hook(make_hook(f"enc{i}.self_attn"))
        )
        handles.append(
            layer.feed_forward.register_forward_hook(make_hook(f"enc{i}.feed_forward"))
        )
    handles.append(model.encoder.register_forward_hook(make_hook("encoder (memory)")))
    for i, layer in enumerate(model.decoder.layers):
        handles.append(
            layer.self_attn.register_forward_hook(make_hook(f"dec{i}.self_attn"))
        )
        handles.append(
            layer.cross_attn.register_forward_hook(make_hook(f"dec{i}.cross_attn"))
        )
    handles.append(model.generator.register_forward_hook(make_hook("generator")))

    src_mask = torch.ones(1, 1, seq_len, dtype=torch.bool)
    tgt_mask = subsequent_mask(tgt_in.size(1))
    logits = model(src, tgt_in, src_mask, tgt_mask)

    for h in handles:
        h.remove()

    print("=" * 72)
    print("Transformer structure walkthrough  -  Attention Is All You Need")
    print("=" * 72)
    print(f"source example : {src[0].tolist()}   ({source_desc})")
    print(f"decoder input  : {tgt_in[0].tolist()}")
    print(
        f"config         : d_model={args.d_model} layers={args.num_layers} "
        f"heads={args.num_heads} d_ff={args.d_ff} vocab={args.vocab_size}"
    )
    print(
        f"tensor legend  : batch x seq_len x d_model "
        f"(=1 x {seq_len} x {args.d_model})\n"
    )

    print("Data flow (shape after each stage):")
    print("-" * 72)
    for name, shape, section in trace:
        note = f"   # {section}" if section else ""
        print(f"  {name:<20} -> {shape:<12}{note}")
    print(f"  {'logits':<20} -> {_shape(logits):<12}   # -> next-token probabilities")

    # Show a real attention matrix from the first encoder self-attention head.
    enc_attn = model.encoder.layers[0].self_attn.attn_weights  # (1, h, q, k)
    head0 = enc_attn[0, 0]
    labels = [str(t) for t in src[0].tolist()]
    print("\nEncoder layer 0 - self-attention weights (head 0):")
    print("(rows = query position's token, cols = key position's token)")
    print(ascii_heatmap(head0, labels))

    # Export a trace for the HTML visualization to optionally consume.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": vars(args),
        "source": src[0].tolist(),
        "target_in": tgt_in[0].tolist(),
        "flow": [{"stage": n, "shape": s, "paper_section": sec} for n, s, sec in trace],
        "encoder_layer0_head0_attention": head0.tolist(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    (DATA_DIR / "forward_trace.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/forward_trace.json ({out['num_parameters']:,} parameters).")
    print("Open visualization/index.html to explore the architecture visually.")


if __name__ == "__main__":
    main()
