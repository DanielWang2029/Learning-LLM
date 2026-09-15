"""End-to-end Flamingo demo: a frozen LM learns to see through opening gates.

Story (all on CPU, well under a minute):

  1. Pretrain a tiny decoder-only LM on the *text* of the VQA task. It learns the
     prompt "what color is the shape ?" but the answer color is unknowable from
     text alone, so its answer accuracy sits at chance (~25%).
  2. Freeze the LM. Wrap it in Flamingo: a vision encoder + Perceiver Resampler +
     gated cross-attention layers whose tanh gates start at 0. At this point the
     model is byte-for-byte the frozen LM (we verify this).
  3. Train ONLY the new visual modules. As the gates open, visual information
     flows in and answer accuracy jumps to ~100%.
  4. Capture the gate-opening curve and a real spatial attention map, and write
     data/flamingo_demo.json for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # shared CPU box: avoid thread oversubscription

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.scenes import (  # noqa: E402
    ANSWER_POS,
    COLOR_NAMES,
    SEQ_LEN,
    STOI,
    VOCAB,
    VOCAB_SIZE,
    make_dataset,
)
from src import Flamingo  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
SEED = 0
CHANCE = 1.0 / len(COLOR_NAMES)
COLOR_IDS = [STOI[c] for c in COLOR_NAMES]


def answer_accuracy(answer_logits: torch.Tensor, color_idx: torch.Tensor) -> float:
    """Fraction of predicted answer tokens that match the true color."""
    pred = answer_logits.argmax(dim=-1)
    true = torch.tensor([COLOR_IDS[c] for c in color_idx.tolist()])
    return (pred == true).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    # ---- Data ------------------------------------------------------------
    tr_imgs, tr_seqs, tr_col = make_dataset(1200, seed=SEED)
    te_imgs, te_seqs, te_col = make_dataset(400, seed=SEED + 999)
    tr_imgs, tr_seqs = torch.from_numpy(tr_imgs), torch.from_numpy(tr_seqs)
    tr_col = torch.from_numpy(tr_col)
    te_imgs, te_seqs = torch.from_numpy(te_imgs), torch.from_numpy(te_seqs)
    te_col = torch.from_numpy(te_col)

    model = Flamingo(VOCAB_SIZE, d_model=64, num_heads=4, d_ff=128,
                     num_layers=2, num_latents=8, max_len=SEQ_LEN)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Flamingo demo | vocab={VOCAB_SIZE} | chance={CHANCE*100:.0f}% | "
          f"params={n_total:,}")

    # ---- Phase 1: pretrain the LM on TEXT ONLY ---------------------------
    print("\n[1] Pretraining the language model on text only...")
    lm = model.lm
    opt_lm = torch.optim.Adam(lm.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    batch = 96
    for step in range(1, 401):
        idx = torch.randint(0, len(tr_seqs), (batch,), generator=gen)
        seq = tr_seqs[idx]
        logits = lm(seq[:, :-1])                      # predict next token
        loss = ce(logits.reshape(-1, VOCAB_SIZE), seq[:, 1:].reshape(-1))
        opt_lm.zero_grad(); loss.backward(); opt_lm.step()
    # Text-only answer accuracy = chance (color unknowable from text).
    with torch.no_grad():
        lm_logits = lm(te_seqs[:, :-1])[:, ANSWER_POS]
    lm_acc = answer_accuracy(lm_logits, te_col)
    print(f"    LM text-only answer accuracy: {lm_acc*100:.1f}%  "
          f"(chance {CHANCE*100:.0f}%) — as expected, the LM can only guess.")

    # ---- Phase 2: freeze LM, verify Flamingo == frozen LM at init --------
    model.freeze_lm()
    with torch.no_grad():
        flam_logits0 = model(te_seqs[:, :-1], te_imgs)[:, ANSWER_POS]
    max_diff = (flam_logits0 - lm_logits).abs().max().item()
    flam_acc0 = answer_accuracy(flam_logits0, te_col)
    print(f"\n[2] Gates initialized to 0 → tanh(0)=0 → model is the pure LM.")
    print(f"    max|Flamingo − LM| logits = {max_diff:.2e}  (≈0 ⇒ identical)")
    print(f"    Flamingo answer accuracy at gate=0: {flam_acc0*100:.1f}%")

    # ---- Phase 3: train ONLY the visual modules --------------------------
    print("\n[3] Training vision + resampler + gated cross-attention "
          "(LM stays frozen)...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"    trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.0f}%)")
    opt = torch.optim.Adam(trainable, lr=8e-4)

    curve = []
    start = time.time()
    steps = 500
    log_steps = {1, 5, 10, 15, 20, 30, 40}
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, len(tr_seqs), (batch,), generator=gen)
        seq, img, col = tr_seqs[idx], tr_imgs[idx], tr_col[idx]
        logits = model(seq[:, :-1], img)[:, ANSWER_POS]     # answer-position logits
        target = torch.tensor([COLOR_IDS[c] for c in col.tolist()])
        loss = ce(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step in log_steps or step % 50 == 0:
            with torch.no_grad():
                te_logits = model(te_seqs[:, :-1], te_imgs)[:, ANSWER_POS]
            acc = answer_accuracy(te_logits, te_col)
            gates = model.gate_values()
            gate_mean = float(np.mean([abs(a) for a, _ in gates]))
            curve.append({"step": step, "loss": round(loss.item(), 4),
                          "acc": round(acc, 4),
                          "gate_attn_mean": round(gate_mean, 4)})
            print(f"    step {step:3d}/{steps} | loss {loss.item():.3f} "
                  f"| acc {acc*100:5.1f}% | mean|attn-gate| {gate_mean:.3f}")

    elapsed = time.time() - start
    with torch.no_grad():
        final_logits = model(te_seqs[:, :-1], te_imgs)[:, ANSWER_POS]
    final_acc = answer_accuracy(final_logits, te_col)
    gates = model.gate_values()
    print(f"\n    Trained {steps} steps in {elapsed:.1f}s")
    print(f"    Final VQA answer accuracy: {final_acc*100:.1f}%  "
          f"(LM-only was {lm_acc*100:.1f}%, chance {CHANCE*100:.0f}%)")
    print("    Final gate values (tanh) per layer [attn, ff]:")
    for i, (ga, gf) in enumerate(gates):
        print(f"      layer {i}: attn={ga:+.3f}  ff={gf:+.3f}")

    # ---- A concrete, correct example + spatial attention map -------------
    model.eval()
    ex_i = 0
    with torch.no_grad():
        ex_logits = model(te_seqs[ex_i:ex_i+1, :-1], te_imgs[ex_i:ex_i+1])[:, ANSWER_POS]
        pred_tok = ex_logits.argmax(-1).item()
    print("\n[4] Example held-out image:")
    print(f"    prompt : {' '.join(VOCAB[t] for t in te_seqs[ex_i, :-1].tolist())}")
    print(f"    true   : {COLOR_NAMES[te_col[ex_i].item()]}")
    print(f"    model  : {VOCAB[pred_tok]}")

    # Compose gated-xattn (answer pos -> R latents) with resampler
    # (R latents -> vision grid) to get where the model looks in the image.
    grid = compute_spatial_attention(model, te_imgs[ex_i:ex_i+1])

    # ---- Persist for the visualization -----------------------------------
    payload = {
        "colors": COLOR_NAMES,
        "chance": round(CHANCE, 4),
        "lm_only_acc": round(lm_acc, 4),
        "final_acc": round(final_acc, 4),
        "init_logit_diff": max_diff,
        "num_params": n_total,
        "num_trainable": n_train,
        "training_curve": curve,
        "final_gates": [{"attn": round(a, 4), "ff": round(f, 4)} for a, f in gates],
        "grid_size": int(np.sqrt(grid.size)),
        "spatial_attention": [[round(float(v), 4) for v in row] for row in grid],
        "example": {
            "true": COLOR_NAMES[te_col[ex_i].item()],
            "pred": VOCAB[pred_tok],
            "image": np.round(te_imgs[ex_i].numpy(), 3).tolist(),  # (3,H,W)
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "flamingo_demo.json").write_text(json.dumps(payload))
    print("\nWrote data/flamingo_demo.json")

    if final_acc < 0.9 or lm_acc > 0.45 or max_diff > 1e-4:
        raise SystemExit("Demo did not meet expectations (see printed metrics).")
    print("OK: frozen LM learned to answer image questions through opened gates.")


def compute_spatial_attention(model: Flamingo, image: torch.Tensor) -> np.ndarray:
    """answer-token → visual-grid attention = gated-xattn ∘ resampler attention."""
    with torch.no_grad():
        vis_feats = model.vision(image)                 # (1, N, d)
        n_grid = vis_feats.size(1)
        _ = model.resampler(vis_feats)                  # populate resampler attn
        _ = model(torch.zeros(1, ANSWER_POS + 1, dtype=torch.long), image)

    # Resampler last-layer attention: latents(R) attend to [vision(N) ++ latents].
    res_attn = model.resampler.layers[-1]["attn"].attn_weights  # (1, h, R, N+R)
    res = res_attn.mean(1)[0, :, :n_grid]               # (R, N) over grid only
    res = res / res.sum(-1, keepdim=True).clamp(min=1e-9)

    # Gated cross-attn (last layer): answer position attends to R latents.
    gx = model.gated_layers[-1].attn.attn_weights.mean(1)[0, ANSWER_POS]  # (R,)
    gx = gx / gx.sum().clamp(min=1e-9)

    spatial = (gx.unsqueeze(-1) * res).sum(0)           # (N,)
    side = int(np.sqrt(n_grid))
    grid = spatial.reshape(side, side).numpy()
    grid = grid / (grid.max() + 1e-9)
    return grid


if __name__ == "__main__":
    main()
