"""Cache-aware streaming Conformer demo.

On CPU, in well under a minute, this script:

  1. trains ONE Conformer with **dynamic chunk masking** (a random chunk size
     each batch) so a single model works at any streaming latency,
  2. verifies the streaming (cached, chunk-by-chunk) output EQUALS the
     full-sequence forward pass within ~1e-6 -- the caching is exact,
  3. shows that per-chunk compute is CONSTANT (with a bounded left context the
     number of attended key frames stops growing) -- no past frame is recomputed,
  4. sweeps the chunk size to expose a latency/accuracy trade-off from that one
     model (bigger chunk -> more look-ahead -> better accuracy, higher latency).

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
from src import StreamingConformerEncoder  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
N_MELS, N_FFT, HOP = 40, 400, 160
FRAME_MS = HOP / 16000 * 1000  # 10 ms per frame (no sub-sampling in this demo)


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
    feats = np.stack([f[:T] for f in feats])
    labs = np.stack([l[:T] for l in labs])
    return torch.from_numpy(feats), torch.from_numpy(labs)


def anticipate(labels: torch.Tensor, delay: int) -> torch.Tensor:
    """Shift labels so frame t must predict the class at frame t+delay.

    This makes *look-ahead* genuinely useful: a frame near the right edge of a
    segment can only get its (future) label right if it can attend a few frames
    ahead — exactly the context a larger streaming chunk provides. Interiors of
    segments are unaffected.
    """
    if delay <= 0:
        return labels
    shifted = labels.clone()
    shifted[:, :-delay] = labels[:, delay:]
    return shifted


def evaluate(model, x, y, chunk_size, left_context):
    model.eval()
    with torch.no_grad():
        logits = model(x, chunk_size=chunk_size, left_context=left_context)
        return (logits.argmax(-1) == y).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=192)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--seg-dur", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--left-context", type=int, default=32,
                   help="bounded left context (frames) for the constant-compute demo")
    p.add_argument("--label-delay", type=int, default=4,
                   help="anticipatory label shift so look-ahead helps accuracy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("Cache-Aware Streaming Conformer  -  one model, many latencies")
    print("=" * 72)
    train_x, train_y = build_dataset(args.n_train, args.num_segments, args.seg_dur, args.seed)
    test_x, test_y = build_dataset(args.n_test, args.num_segments, args.seg_dur, args.seed + 1)
    train_y = anticipate(train_y, args.label_delay)
    test_y = anticipate(test_y, args.label_delay)
    T = train_x.size(1)
    # Dynamic chunk sizes seen in training; full-context (T) is included twice so
    # the offline mode -- the accuracy ceiling -- is trained as thoroughly.
    train_chunks = [4, 8, 16, T, T]
    print(f"Frames per utterance: T = {T} ({FRAME_MS:.0f} ms/frame). "
          f"Anticipatory label delay: {args.label_delay} frames "
          f"({args.label_delay*FRAME_MS:.0f} ms of useful look-ahead).")
    print(f"Dynamic training chunks: {train_chunks}\n")

    torch.manual_seed(args.seed)
    model = StreamingConformerEncoder(
        n_mels=N_MELS, d_model=args.d_model, num_heads=args.num_heads,
        d_ff=args.d_ff, num_blocks=args.num_blocks, kernel_size=args.kernel_size,
        num_classes=NUM_CLASSES,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("[1/4] Training with dynamic chunk masking ...")
    n = train_x.size(0)
    rng = np.random.default_rng(args.seed)
    model.train()
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            chunk = int(train_chunks[rng.integers(len(train_chunks))])  # random per batch
            logits = model(train_x[idx], chunk_size=chunk, left_context=None)
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), train_y[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---- [2/4] streaming == full-context equality --------------------------
    print("[2/4] Verifying streaming (cached) == full-context forward ...")
    model.eval()
    check_chunk = 8
    with torch.no_grad():
        full = model(test_x, chunk_size=check_chunk, left_context=None)
        stream, key_counts_unbounded = model.forward_streaming(
            test_x, chunk_size=check_chunk, left_context=None)
    max_diff = (full - stream).abs().max().item()
    print(f"      chunk={check_chunk}, max |full - streaming| = {max_diff:.2e}  "
          f"(< 1e-5 required)")

    # ---- [3/4] constant per-chunk compute (bounded left context) -----------
    print("[3/4] Per-chunk compute with bounded left context "
          f"Lc={args.left_context} ...")
    with torch.no_grad():
        _, key_counts = model.forward_streaming(
            test_x, chunk_size=check_chunk, left_context=args.left_context)
    steady = key_counts[len(key_counts) // 2:]
    print(f"      attended key frames per chunk: {key_counts}")
    print(f"      -> constant at {max(steady)} once the cache is full "
          f"(no recomputation of past frames).")

    # ---- [4/4] latency / accuracy sweep from the single model --------------
    print("[4/4] Latency/accuracy sweep (one model, varying chunk size) ...")
    sweep = []
    for chunk in [2, 4, 8, 16, T]:
        acc = evaluate(model, test_x, test_y, chunk_size=chunk, left_context=None)
        latency = chunk * FRAME_MS
        label = "full/offline" if chunk >= T else f"chunk={chunk}"
        sweep.append({"chunk": int(chunk), "latency_ms": latency, "accuracy": acc,
                      "label": label})

    print("\n" + "-" * 60)
    print(f"{'mode':<16}{'chunk':>8}{'latency':>12}{'frame acc':>14}")
    print("-" * 60)
    for s in sweep:
        print(f"{s['label']:<16}{s['chunk']:>8}{s['latency_ms']:>10.0f}ms"
              f"{s['accuracy']*100:>13.1f}%")
    print("-" * 60)
    lo, hi = sweep[0]["accuracy"], sweep[-1]["accuracy"]
    print(f"\nSame weights, streamed at different latencies: accuracy rises from "
          f"{lo*100:.1f}% (lowest latency) to {hi*100:.1f}% (offline) as look-ahead grows.")

    out = {
        "config": vars(args),
        "frames": T,
        "frame_ms": FRAME_MS,
        "equality": {"chunk": check_chunk, "max_abs_diff": max_diff,
                     "key_counts_unbounded": key_counts_unbounded},
        "constant_compute": {"left_context": args.left_context,
                             "key_counts": key_counts,
                             "steady_state_keys": int(max(steady))},
        "latency_accuracy_sweep": sweep,
        "class_names": CLASS_NAMES,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "demo_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/demo_results.json  |  total time {time.time()-t0:.1f}s")

    ok = (max_diff < 1e-5 and len(set(steady)) == 1 and hi >= lo)
    print("OK: exact caching, constant compute, and a real latency/accuracy trade-off."
          if ok else "WARNING: check results above.")


if __name__ == "__main__":
    main()
