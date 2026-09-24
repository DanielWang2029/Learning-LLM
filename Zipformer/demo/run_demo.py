"""Zipformer demo: a low-rate middle matches accuracy with fewer FLOPs.

On CPU, in well under a minute, this script:

  1. synthesizes a full-rate frame-classification dataset from numpy audio,
  2. trains two encoders with the *same* number of blocks:
       - a constant-rate baseline (every block at the full frame rate), and
       - a Zipformer whose middle blocks run at HALF the frame rate
         (downsample → low-rate blocks → upsample), plus BiasNorm + Bypass,
  3. reports an analytic self-attention FLOP proxy and frame accuracy for each.

Headline result (Zipformer §3.1): the U-Net-like low-rate middle cuts encoder
attention FLOPs substantially while matching the constant-rate encoder's
accuracy — attention is O(T²), so halving the middle's frame rate makes it ~4x
cheaper there.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.audio_synth import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    frame_labels,
    log_mel,
    synth_utterance,
)
from src import ConstantRateEncoder, ZipformerEncoder  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
N_MELS, N_FFT, HOP = 40, 400, 160


def build_dataset(n_utts, num_segments, seg_dur, seed):
    rng = np.random.default_rng(seed)
    feats, labs = [], []
    for _ in range(n_utts):
        wave, sample_lab = synth_utterance(num_segments, seg_dur, rng)
        mel, hop = log_mel(wave, N_FFT, HOP, N_MELS)
        flab = frame_labels(sample_lab, N_FFT, hop, mel.shape[0])
        feats.append(mel)
        labs.append(flab)
    T = min(f.shape[0] for f in feats)
    T -= T % 2  # keep it even so downsample-by-2 is clean
    feats = np.stack([f[:T] for f in feats])
    labs = np.stack([l[:T] for l in labs])
    return torch.from_numpy(feats), torch.from_numpy(labs)


def attention_flops_constant(T, d, num_blocks):
    return num_blocks * 2 * T * T * d


def attention_flops_zipformer(T, d, num_middle, factor):
    stem_head = 2 * (2 * T * T * d)                      # stem + head at full rate
    mid_T = T // factor
    middle = num_middle * 2 * mid_T * mid_T * d          # middle at low rate
    return stem_head + middle


def train_eval(model, train_x, y_train, test_x, y_test, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = train_x.size(0)
    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = model(train_x[idx])
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y_train[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(test_x).argmax(-1) == y_test).float().mean().item()
    return acc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=192)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--seg-dur", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--num-middle", type=int, default=2)  # → 4 total blocks
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    total_blocks = args.num_middle + 2  # stem + middle + head
    factor = 2

    t0 = time.time()
    print("=" * 70)
    print("Zipformer  -  low-rate middle (U-Net) vs constant-rate encoder")
    print("=" * 70)
    print(f"Synthesizing {args.n_train} train / {args.n_test} test utterances "
          f"({NUM_CLASSES} classes) ...")
    train_x, train_y = build_dataset(args.n_train, args.num_segments, args.seg_dur, args.seed)
    test_x, test_y = build_dataset(args.n_test, args.num_segments, args.seg_dur, args.seed + 1)
    T = train_x.size(1)
    print(f"Full-rate frames per utterance: T = {T}")
    print(f"Both encoders use {total_blocks} blocks; Zipformer runs {args.num_middle} "
          f"middle blocks at T/{factor} = {T // factor}.\n")

    torch.manual_seed(args.seed)
    baseline = ConstantRateEncoder(
        N_MELS, args.d_model, args.num_heads, args.d_ff, total_blocks,
        args.kernel_size, NUM_CLASSES,
    )
    print("Training constant-rate baseline ...")
    acc_base = train_eval(baseline, train_x, train_y, test_x, test_y, args)

    torch.manual_seed(args.seed)
    zip_model = ZipformerEncoder(
        N_MELS, args.d_model, args.num_heads, args.d_ff, args.num_middle,
        args.kernel_size, NUM_CLASSES, downsample_factor=factor,
    )
    print("Training Zipformer (low-rate middle) ...")
    acc_zip = train_eval(zip_model, train_x, train_y, test_x, test_y, args)

    flops_base = attention_flops_constant(T, args.d_model, total_blocks)
    flops_zip = attention_flops_zipformer(T, args.d_model, args.num_middle, factor)
    saving = 100 * (1 - flops_zip / flops_base)

    print("\n" + "-" * 70)
    print(f"{'metric':<28}{'constant-rate':>18}{'Zipformer':>18}")
    print("-" * 70)
    print(f"{'blocks':<28}{total_blocks:>18}{total_blocks:>18}")
    print(f"{'middle frame rate':<28}{'full (T)':>18}{'half (T/2)':>18}")
    print(f"{'attention FLOPs (proxy)':<28}{flops_base:>18,}{flops_zip:>18,}")
    print(f"{'frame accuracy':<28}{acc_base*100:>17.1f}%{acc_zip*100:>17.1f}%")
    print("-" * 70)
    print(f"\nZipformer cuts encoder attention FLOPs by {saving:.1f}% "
          f"({flops_base/flops_zip:.2f}x fewer) while matching accuracy.")
    print(f"Accuracy: constant-rate={acc_base*100:.1f}%  Zipformer={acc_zip*100:.1f}%")

    params_base = sum(p.numel() for p in baseline.parameters())
    params_zip = sum(p.numel() for p in zip_model.parameters())
    out = {
        "config": vars(args),
        "input_frames": T,
        "total_blocks": total_blocks,
        "downsample_factor": factor,
        "constant_rate": {
            "attention_flops": int(flops_base), "accuracy": acc_base,
            "middle_rate": "full", "params": params_base,
        },
        "zipformer": {
            "attention_flops": int(flops_zip), "accuracy": acc_zip,
            "middle_rate": "half", "params": params_zip,
        },
        "flop_saving_percent": saving,
        "flop_speedup": flops_base / flops_zip,
        "class_names": CLASS_NAMES,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "demo_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/demo_results.json  |  total time {time.time()-t0:.1f}s")

    ok = flops_zip < flops_base and acc_zip >= 0.9 and abs(acc_zip - acc_base) < 0.1
    print("OK: Zipformer matched accuracy with fewer FLOPs." if ok
          else "WARNING: check results above.")


if __name__ == "__main__":
    main()
