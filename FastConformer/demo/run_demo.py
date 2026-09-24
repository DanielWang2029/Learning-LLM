"""FastConformer demo: 4x regular vs 8x depthwise-separable sub-sampling.

End to end, on CPU, in well under a minute this script:

  1. synthesizes a frame-labeling dataset from numpy audio (no downloads),
  2. trains two *identical* Conformer encoders that differ only in their
     sub-sampling front end -- baseline 4x regular conv vs FastConformer's
     8x depthwise-separable conv,
  3. reports, for each, the token count the attention/conv blocks operate
     over, an analytic multiply-add (MAC) compute proxy, and frame accuracy.

The headline result (FastConformer §2.1): 8x sub-sampling roughly *halves*
the token count vs 4x -> ~4x cheaper self-attention (quadratic in tokens) and
a much cheaper sub-sampling block -- with no loss in accuracy.

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

torch.set_num_threads(1)  # keep CPU timing honest and stable

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.audio_synth import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    frame_labels,
    log_mel,
    synth_utterance,
)
from src import FastConformerEncoder  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
N_MELS = 40
N_FFT = 400
HOP = 160


def pool_labels(frame_lab: np.ndarray, out_len: int) -> np.ndarray:
    """Majority-vote frame labels down to ``out_len`` sub-sampled tokens."""
    out = np.empty(out_len, dtype=np.int64)
    edges = np.linspace(0, len(frame_lab), out_len + 1).astype(int)
    for i in range(out_len):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[i] = np.bincount(frame_lab[lo:hi], minlength=NUM_CLASSES).argmax()
    return out


def build_dataset(n_utts, num_segments, seg_dur, seed):
    """Return log-mel features (N, T, n_mels) and frame labels (N, T)."""
    rng = np.random.default_rng(seed)
    feats, labs = [], []
    for _ in range(n_utts):
        wave, sample_lab = synth_utterance(num_segments, seg_dur, rng)
        mel, hop = log_mel(wave, N_FFT, HOP, N_MELS)
        flab = frame_labels(sample_lab, N_FFT, hop, mel.shape[0])
        feats.append(mel)
        labs.append(flab)
    T = min(f.shape[0] for f in feats)
    feats = np.stack([f[:T] for f in feats])
    labs = np.stack([l[:T] for l in labs])
    return torch.from_numpy(feats), torch.from_numpy(labs)


def subsampling_macs(subsampling: str, n_mels: int, d: int, T: int, k: int = 9):
    """Analytic multiply-adds for one utterance's sub-sampling block."""
    if subsampling == "4x":
        t1 = (T + 2 - 3) // 2 + 1
        t2 = (t1 + 2 - 3) // 2 + 1
        return t1 * d * n_mels * 3 + t2 * d * d * 3, t2
    # 8x depthwise-separable: three (depthwise + pointwise) stages.
    t1 = (T + 2 * (k // 2) - k) // 2 + 1
    t2 = (t1 + 2 * (k // 2) - k) // 2 + 1
    t3 = (t2 + 2 * (k // 2) - k) // 2 + 1
    macs = (t1 * n_mels * k + t1 * d * n_mels)      # stage 1 (dw + pw)
    macs += (t2 * d * k + t2 * d * d)               # stage 2
    macs += (t3 * d * k + t3 * d * d)               # stage 3
    return macs, t3


def attention_macs(T: int, d: int, num_blocks: int) -> int:
    """Quadratic self-attention MACs proxy: QK^T + AV over all blocks."""
    return num_blocks * 2 * T * T * d


def train_eval(subsampling, train_x, train_y, test_x, test_y, args):
    torch.manual_seed(args.seed)
    model = FastConformerEncoder(
        n_mels=N_MELS, d_model=args.d_model, num_heads=args.num_heads,
        d_ff=args.d_ff, num_blocks=args.num_blocks, num_classes=NUM_CLASSES,
        subsampling=subsampling, dropout=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    T = train_x.size(1)
    out_len = model(train_x[:1]).size(1)  # tokens after sub-sampling
    y_train = torch.stack([
        torch.from_numpy(pool_labels(train_y[i].numpy(), out_len))
        for i in range(train_y.size(0))
    ])
    y_test = torch.stack([
        torch.from_numpy(pool_labels(test_y[i].numpy(), out_len))
        for i in range(test_y.size(0))
    ])

    n = train_x.size(0)
    model.train()
    for epoch in range(args.epochs):
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
        pred = model(test_x).argmax(-1)
        acc = (pred == y_test).float().mean().item()

    params = sum(p.numel() for p in model.parameters())
    sub_macs, _ = subsampling_macs(subsampling, N_MELS, args.d_model, T)
    attn_macs = attention_macs(out_len, args.d_model, args.num_blocks)
    return {
        "subsampling": subsampling,
        "factor": model.factor,
        "tokens": out_len,
        "subsampling_macs": int(sub_macs),
        "attention_macs": int(attn_macs),
        "accuracy": acc,
        "params": params,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=192)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--seg-dur", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("FastConformer  -  4x regular vs 8x depthwise-separable sub-sampling")
    print("=" * 70)
    print(f"Synthesizing {args.n_train} train / {args.n_test} test utterances "
          f"({NUM_CLASSES} classes: {', '.join(CLASS_NAMES)}) ...")
    train_x, train_y = build_dataset(args.n_train, args.num_segments, args.seg_dur, args.seed)
    test_x, test_y = build_dataset(args.n_test, args.num_segments, args.seg_dur, args.seed + 1)
    T = train_x.size(1)
    print(f"Input log-mel frames per utterance (10 ms hop): T = {T}\n")

    results = {}
    for sub in ("4x", "8x"):
        print(f"Training encoder with {sub} sub-sampling ...")
        results[sub] = train_eval(sub, train_x, train_y, test_x, test_y, args)

    r4, r8 = results["4x"], results["8x"]
    print("\n" + "-" * 70)
    print(f"{'metric':<26}{'4x baseline':>16}{'8x FastConformer':>20}")
    print("-" * 70)
    print(f"{'sub-sampling type':<26}{'regular conv':>16}{'depthwise-sep':>20}")
    print(f"{'encoder tokens':<26}{r4['tokens']:>16}{r8['tokens']:>20}")
    print(f"{'sub-sampling MACs':<26}{r4['subsampling_macs']:>16,}{r8['subsampling_macs']:>20,}")
    print(f"{'attention MACs (proxy)':<26}{r4['attention_macs']:>16,}{r8['attention_macs']:>20,}")
    print(f"{'frame accuracy':<26}{r4['accuracy']*100:>15.1f}%{r8['accuracy']*100:>19.1f}%")
    print("-" * 70)

    token_ratio = r4["tokens"] / r8["tokens"]
    attn_ratio = r4["attention_macs"] / max(r8["attention_macs"], 1)
    sub_ratio = r4["subsampling_macs"] / max(r8["subsampling_macs"], 1)
    print(f"\n8x uses {token_ratio:.2f}x fewer tokens than 4x "
          f"-> {attn_ratio:.2f}x cheaper self-attention (quadratic in tokens).")
    print(f"8x depthwise-separable sub-sampling is {sub_ratio:.2f}x cheaper "
          f"than 4x regular-conv sub-sampling.")
    print(f"Accuracy retained: 4x={r4['accuracy']*100:.1f}%  8x={r8['accuracy']*100:.1f}%")

    out = {
        "config": vars(args),
        "input_frames": T,
        "results": results,
        "token_ratio_4x_over_8x": token_ratio,
        "attention_speedup_8x": attn_ratio,
        "subsampling_speedup_8x": sub_ratio,
        "class_names": CLASS_NAMES,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "demo_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/demo_results.json  |  total time {time.time()-t0:.1f}s")

    ok = (r8["tokens"] < r4["tokens"] and r8["accuracy"] >= 0.9
          and abs(r8["accuracy"] - r4["accuracy"]) < 0.1)
    print("OK: 8x halved tokens and retained accuracy." if ok
          else "WARNING: check results above.")


if __name__ == "__main__":
    main()
