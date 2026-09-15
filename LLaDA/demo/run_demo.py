"""End-to-end LLaDA demo: train a masked-diffusion LM, then GENERATE.

What this shows (all on CPU in well under a minute):

1. We train a small bidirectional Transformer with the masked-diffusion
   objective (random masking ratio -> predict the masked tokens).
2. **Reconstruction**: mask half of a held-out sequence and fill it in one
   shot — the model recovers the original arithmetic progression.
3. **Generation**: start from an all-``[MASK]`` sequence and iteratively unmask
   over T steps (non-autoregressively). We report the fraction of generated
   sequences that are *valid* arithmetic progressions, and print a step-by-step
   unmasking trace.

A JSON summary is written to ``data/llada_run.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # keep CPU threading predictable / fast for a tiny model

# Make ``src`` importable regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import MaskPredictor, diffusion_loss, generate, reconstruct, forward_mask  # noqa: E402

SEED = 0
VOCAB = 10          # content tokens 0..9
MASK_ID = VOCAB     # [MASK] is id 10
VOCAB_SIZE = VOCAB + 1
SEQ_LEN = 12
PERIOD = 3          # sequences repeat with this period
STEPS = 12          # reverse-diffusion steps (one token committed per step)
TEMP = 0.7          # sampling temperature for generation (diversity)
TRAIN_STEPS = 1500
BATCH = 64


def is_valid_periodic(seq, period: int) -> bool:
    """True iff ``seq`` repeats with the given period (seq[i] == seq[i+P])."""
    return all(seq[i] == seq[i % period] for i in range(len(seq)))


def make_batch(batch: int, gen: torch.Generator) -> torch.Tensor:
    motif = torch.randint(0, VOCAB, (batch, PERIOD), generator=gen)
    idx = torch.arange(SEQ_LEN) % PERIOD
    return motif[:, idx]


def main() -> None:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    device = torch.device("cpu")

    model = MaskPredictor(vocab_size=VOCAB_SIZE, d_model=96, num_layers=3, num_heads=4, d_ff=192, max_len=SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Masked-diffusion LM | params={n_params:,} | vocab={VOCAB_SIZE} seq_len={SEQ_LEN} steps={STEPS}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    start = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        tokens = make_batch(BATCH, gen).to(device)
        loss = diffusion_loss(model, tokens, MASK_ID, generator=gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:4d}/{TRAIN_STEPS} | diffusion loss {loss.item():.4f}")
    print(f"trained in {time.time() - start:.1f}s\n")

    model.eval()

    # ---- 2. Reconstruction: mask half of held-out sequences, fill in one shot.
    eval_tokens = make_batch(256, torch.Generator().manual_seed(SEED + 1)).to(device)
    _, mask, _ = forward_mask(
        eval_tokens, MASK_ID, t=torch.full((256, 1), 0.5), generator=torch.Generator().manual_seed(SEED + 2)
    )
    filled = reconstruct(model, eval_tokens, mask, MASK_ID)
    token_acc = (filled[mask] == eval_tokens[mask]).float().mean().item()
    seq_acc = (filled == eval_tokens).all(dim=1).float().mean().item()
    print("Reconstruction (50% masked, one-shot fill):")
    print(f"  masked-token accuracy : {token_acc * 100:5.1f}%")
    print(f"  exact-sequence accuracy: {seq_acc * 100:5.1f}%\n")

    # ---- 3. Generation: all-mask -> full sequence via iterative unmasking.
    gen_batch = 256
    samples = generate(
        model, SEQ_LEN, MASK_ID, steps=STEPS, batch_size=gen_batch,
        device=device, temperature=TEMP,
    )
    valid = [is_valid_periodic(s.tolist(), PERIOD) for s in samples]
    valid_rate = sum(valid) / len(valid)
    print(f"Generation from all-[MASK] ({gen_batch} samples, {STEPS} steps):")
    print(f"  valid periodic-sequence rate: {valid_rate * 100:5.1f}%")
    uniq = len({tuple(s.tolist()) for s in samples})
    print(f"  distinct sequences generated      : {uniq}/{gen_batch}\n")

    # ---- One traced generation for the visualization + console trace.
    _, trace = generate(
        model, SEQ_LEN, MASK_ID, steps=STEPS, batch_size=1, device=device,
        temperature=TEMP, record_trace=True,
    )
    print("Step-by-step unmasking trace (one sample; '_' = still masked):")
    for snap in trace:
        cells = [
            "_" if m else str(tok)
            for tok, m in zip(snap["tokens"], snap["is_mask"])
        ]
        print(f"  step {snap['step']}/{STEPS} [{snap['kept_masked']:2d} masked]: " + " ".join(f"{c:>2}" for c in cells))
    final = trace[-1]["tokens"]
    print(f"\n  final sequence: {final}  -> valid periodic: {is_valid_periodic(final, PERIOD)}")

    # A reconstruction example for the viz.
    recon_example = {
        "original": eval_tokens[0].tolist(),
        "masked": [MASK_ID if m else t for t, m in zip(eval_tokens[0].tolist(), mask[0].tolist())],
        "filled": filled[0].tolist(),
    }

    out = {
        "meta": {
            "vocab": VOCAB, "mask_id": MASK_ID, "seq_len": SEQ_LEN,
            "period": PERIOD, "steps": STEPS, "params": n_params,
        },
        "metrics": {
            "recon_token_acc": token_acc,
            "recon_seq_acc": seq_acc,
            "gen_valid_rate": valid_rate,
            "distinct": uniq,
            "total": gen_batch,
        },
        "trace": trace,
        "reconstruction": recon_example,
    }
    out_path = ROOT / "data" / "llada_run.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    assert valid_rate > 0.8, f"generation validity too low: {valid_rate:.2f}"
    assert token_acc > 0.8, f"reconstruction accuracy too low: {token_acc:.2f}"
    print("OK: masked diffusion reconstructs and generates valid structured sequences.")


if __name__ == "__main__":
    main()
