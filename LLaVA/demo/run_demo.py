"""LLaVA — Visual Instruction Tuning, reproduced at tiny scale on CPU.

Liu et al. (2023) connect a *frozen* vision encoder to a language model with a
single learned **projection**, then instruction-tune so the LM can answer
questions about an image. We reproduce that recipe end-to-end:

  1. Generate synthetic images (one colored shape each) with NumPy, paired with
     an instruction ("what color?" / "what shape?") and the true answer.
  2. Freeze a tiny CNN vision encoder; learn only the projection + a tiny LM.
  3. After instruction tuning, answer questions about **held-out** images far
     above chance.
  4. ABLATE the projection (feed the LM zeros instead of visual tokens): the
     model goes blind and collapses to chance — proving the projection is the
     bridge that makes vision usable by the LM.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared box: avoid CPU oversubscription

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

import numpy as np  # noqa: E402

from src.data import (COLOR_NAMES, ITOS, SHAPES, STOI, VOCAB,  # noqa: E402
                      make_dataset)
from src.model import LlavaTiny  # noqa: E402
from src.vision import VisionEncoder  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
IMG_SIZE = 12
N_TRAIN, N_TEST = 900, 300
STEPS, BATCH, LR = 400, 64, 1e-3


def accuracy(model, images, instr, answer, ablate: bool) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(images, instr, ablate_projection=ablate)
        pred = logits.argmax(dim=-1)
    return (pred == answer).float().mean().item()


def per_type_acc(model, images, instr, answer, ablate: bool):
    model.eval()
    with torch.no_grad():
        pred = model(images, instr, ablate_projection=ablate).argmax(dim=-1)
    out = {}
    for name, tok in (("color", "ask_color"), ("shape", "ask_shape")):
        m = instr == STOI[tok]
        out[name] = (pred[m] == answer[m]).float().mean().item()
    return out


def train(model, images, instr, answer, ablate: bool, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Only the projector + LM train; the vision encoder stays frozen.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    n = images.shape[0]
    model.train()
    for _ in range(STEPS):
        idx = rng.integers(0, n, size=BATCH)
        logits = model(images[idx], instr[idx], ablate_projection=ablate)
        loss = torch.nn.functional.cross_entropy(logits, answer[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def main() -> None:
    torch.manual_seed(0)
    tr_img, tr_instr, tr_ans = make_dataset(N_TRAIN, IMG_SIZE, seed=1)
    te_img, te_instr, te_ans = make_dataset(N_TEST, IMG_SIZE, seed=999)  # held out
    tr_img, tr_instr, tr_ans = map(torch.from_numpy, (tr_img, tr_instr, tr_ans))
    te_img, te_instr, te_ans = map(torch.from_numpy, (te_img, te_instr, te_ans))

    vision = VisionEncoder(feat_dim=16, seed=0)
    n_patches = vision(tr_img[:1]).shape[1]

    print("=" * 68)
    print("LLaVA — Visual Instruction Tuning (tiny CPU reproduction)")
    print("=" * 68)
    print(f"images: {IMG_SIZE}x{IMG_SIZE} RGB | shapes={SHAPES} colors={COLOR_NAMES}")
    print(f"vision encoder: FROZEN CNN -> {n_patches} patch tokens (dim 16)")
    print(f"trainable: projection + tiny LM | vocab={len(VOCAB)}")
    print(f"chance ≈ 25% (color, 1/4) / 33% (shape, 1/3)\n")

    # --- Full model: frozen vision + learned projection + LM.
    full = LlavaTiny(vision, vocab_size=len(VOCAB), d_model=48)
    n_train_params = sum(p.numel() for p in full.parameters() if p.requires_grad)
    t0 = time.time()
    train(full, tr_img, tr_instr, tr_ans, ablate=False, seed=42)
    full_acc = accuracy(full, te_img, te_instr, te_ans, ablate=False)
    full_types = per_type_acc(full, te_img, te_instr, te_ans, ablate=False)

    # --- Post-hoc ablation: same trained model, projection output zeroed.
    ablate_acc = accuracy(full, te_img, te_instr, te_ans, ablate=True)
    ablate_types = per_type_acc(full, te_img, te_instr, te_ans, ablate=True)

    # --- Blind baseline: a model trained WITHOUT the visual bridge.
    blind = LlavaTiny(VisionEncoder(feat_dim=16, seed=0), vocab_size=len(VOCAB),
                      d_model=48)
    train(blind, tr_img, tr_instr, tr_ans, ablate=True, seed=7)
    blind_acc = accuracy(blind, te_img, te_instr, te_ans, ablate=True)
    elapsed = time.time() - t0

    print(f"trainable params: {n_train_params:,}\n")
    print(f"{'condition':<34}{'overall':>9}{'color':>9}{'shape':>9}")
    print("-" * 61)
    print(f"{'FULL (projection trained)':<34}{full_acc*100:>8.1f}%"
          f"{full_types['color']*100:>8.1f}%{full_types['shape']*100:>8.1f}%")
    print(f"{'ABLATED (projection -> zeros)':<34}{ablate_acc*100:>8.1f}%"
          f"{ablate_types['color']*100:>8.1f}%{ablate_types['shape']*100:>8.1f}%")
    print(f"{'BLIND (trained w/o vision)':<34}{blind_acc*100:>8.1f}%"
          f"{'—':>9}{'—':>9}")

    # A few concrete held-out predictions.
    print("\nHeld-out examples (question -> model answer | truth):")
    show = []
    with torch.no_grad():
        pred_full = full(te_img[:8], te_instr[:8]).argmax(-1)
        pred_abl = full(te_img[:8], te_instr[:8], ablate_projection=True).argmax(-1)
    for i in range(6):
        q = ITOS[int(te_instr[i])]
        a_full, a_abl, truth = (ITOS[int(pred_full[i])], ITOS[int(pred_abl[i])],
                                ITOS[int(te_ans[i])])
        mark = "OK " if a_full == truth else "XX "
        print(f"  {mark}[{q:>10}]  full={a_full:<9} ablated={a_abl:<9} truth={truth}")
        show.append({
            "image": te_img[i].permute(1, 2, 0).tolist(),  # HWC for the browser
            "question": q, "truth": truth,
            "pred_full": a_full, "pred_ablated": a_abl,
        })

    print(f"\nTotal train+eval time: {elapsed:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "img_size": IMG_SIZE, "n_patches": int(n_patches),
        "shapes": SHAPES, "colors": COLOR_NAMES, "vocab": VOCAB,
        "results": {
            "full": {"overall": full_acc, **full_types},
            "ablated": {"overall": ablate_acc, **ablate_types},
            "blind": {"overall": blind_acc},
        },
        "chance": {"color": 1 / len(COLOR_NAMES), "shape": 1 / len(SHAPES)},
        "samples": show,
    }
    (DATA_DIR / "llava_results.json").write_text(json.dumps(out))
    print(f"Wrote {DATA_DIR / 'llava_results.json'}")

    if not (full_acc > 0.85 and ablate_acc < 0.5):
        raise SystemExit(
            f"Demo not convincing: full={full_acc:.2f} ablated={ablate_acc:.2f}"
        )
    print("OK: the learned projection is what lets the LM see the image.")


if __name__ == "__main__":
    main()
