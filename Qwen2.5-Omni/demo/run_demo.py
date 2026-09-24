"""Block-wise streaming audio encoder demo (CPU, < 60 s).

Trains a tiny audio encoder with FULL attention on a synthetic per-frame
recognition task, then shows that switching to Qwen2.5-Omni's BLOCK-WISE
(2-second) attention:

1. closely approximates full-attention perception (small output difference,
   near-identical frame accuracy),
2. is mathematically equivalent to encoding each block independently — i.e. it
   enables streaming/chunked prefill,
3. turns the O(T²) attention cost into O(T · block), scaling linearly, and
4. yields bounded, constant per-block prefill latency.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.synth_audio import make_example  # noqa: E402
from src import AudioTagger, frames_per_seconds, log_mel_spectrogram  # noqa: E402
from src.encoder import FRAME_SECONDS, block_diagonal_mask  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

SEQ_LEN = 20
VOCAB_SIZE = 10
FRAMES_PER_TOKEN = 8       # 8 encoder frames (~0.32 s) per token
D_MODEL = 96


def build_dataset(n: int, rng: np.random.Generator):
    mels, labels = [], []
    for _ in range(n):
        _, audio, lab = make_example(SEQ_LEN, VOCAB_SIZE, FRAMES_PER_TOKEN, rng)
        mels.append(log_mel_spectrogram(audio))
        labels.append(lab)
    return torch.tensor(np.stack(mels)), torch.tensor(np.stack(labels)).long()


def frame_accuracy(model, mel, labels, block_frames) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(mel, block_frames=block_frames)
        pred = logits.argmax(-1)
        n = min(pred.size(1), labels.size(1))
        return (pred[:, :n] == labels[:, :n]).float().mean().item()


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("=" * 70)
    print("Qwen2.5-Omni demo — block-wise streaming audio encoder")
    print("=" * 70)

    block2s = frames_per_seconds(2.0)
    print(f"Encoder frame = {FRAME_SECONDS*1000:.0f} ms  ->  a 2 s block = {block2s} frames")

    print("Synthesizing audio + log-Mel spectrograms...")
    train_mel, train_lab = build_dataset(256, rng)
    eval_mel, eval_lab = build_dataset(64, rng)
    n_frames = train_mel.size(2)
    enc_frames = SEQ_LEN * FRAMES_PER_TOKEN
    print(f"  {train_mel.size(0)} train / {eval_mel.size(0)} eval clips | "
          f"{enc_frames} encoder frames ({enc_frames*FRAME_SECONDS:.1f}s) each\n")

    model = AudioTagger(VOCAB_SIZE, n_mels=128, d_model=D_MODEL, num_layers=3,
                        num_heads=4, d_ff=192)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  encoder parameters: {n_params:,}")

    # --- Train with FULL attention ---
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    crit = torch.nn.CrossEntropyLoss()
    batch, steps = 32, 180
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, train_mel.size(0), (batch,))
        logits = model(train_mel[idx])            # full attention
        loss = crit(logits.reshape(-1, VOCAB_SIZE), train_lab[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0 or step == 1:
            acc = frame_accuracy(model, eval_mel, eval_lab, None)
            print(f"  step {step:3d}/{steps} | loss {loss.item():.4f} | full-attn frame-acc {acc*100:5.1f}%")
    print(f"  trained in {time.time()-start:.1f}s\n")

    # --- 1. Block-wise closely approximates full attention ---
    print("-" * 70)
    print("Approximation: full vs block-wise attention (same weights)")
    print("-" * 70)
    acc_full = frame_accuracy(model, eval_mel, eval_lab, None)
    approx_rows = []
    for secs in (0.4, 0.8, 2.0, 4.0):
        bf = frames_per_seconds(secs)
        acc_b = frame_accuracy(model, eval_mel, eval_lab, bf)
        with torch.no_grad():
            full_emb = model.encoder(eval_mel[:8])
            blk_emb = model.encoder(eval_mel[:8], block_frames=bf)
        diff = (full_emb - blk_emb)
        rel = (diff.norm() / full_emb.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            full_emb.reshape(-1, D_MODEL), blk_emb.reshape(-1, D_MODEL), dim=-1
        ).mean().item()
        approx_rows.append({"block_s": secs, "block_frames": bf,
                            "frame_acc": round(acc_b, 4), "rel_l2": round(rel, 4),
                            "cosine": round(cos, 4)})
        print(f"  block={secs:>3}s ({bf:>3} fr) | frame-acc {acc_b*100:5.1f}% | "
              f"rel-L2 diff {rel*100:5.1f}% | cosine {cos:.3f}")
    print(f"  full attention          | frame-acc {acc_full*100:5.1f}% (reference)")

    # --- 2. Streaming equivalence: block-diagonal pass == per-block encode ---
    with torch.no_grad():
        masked = model.encoder(eval_mel[:1], block_frames=block2s)
        streamed, counts = model.encoder.streaming_encode(eval_mel[:1], block2s)
    max_dev = (masked - streamed).abs().max().item()
    print(f"\nStreaming equivalence: max|block-diagonal - per-block| = {max_dev:.2e} "
          f"({'IDENTICAL' if max_dev < 1e-4 else 'differs'})")
    print(f"  streamed as blocks of sizes: {counts}")

    # --- 3. Cost: full O(T^2) vs block-wise O(T*block) ---
    print("\n" + "-" * 70)
    print("Attention cost: full O(T^2) vs block-wise O(T*block)")
    print("-" * 70)
    print("  (block-wise runs blocks on the batch axis, as the paper describes)")
    cost_rows = []
    layer = model.encoder.transformer
    for T in (50, 100, 200, 400, 800, 1600):
        x = torch.randn(1, T, D_MODEL)
        t0 = time.time()
        for _ in range(3):
            layer(x)                                 # full attention over T
        t_full = (time.time() - t0) / 3
        # Partition frames into blocks stacked on the batch axis.
        pad = (-T) % block2s
        xb = torch.cat([x, torch.zeros(1, pad, D_MODEL)], dim=1) if pad else x
        xb = xb.reshape(xb.size(1) // block2s, block2s, D_MODEL)
        t0 = time.time()
        for _ in range(3):
            layer(xb)                                # block-local attention only
        t_block = (time.time() - t0) / 3
        pairs_full = T * T
        pairs_block = T * min(T, block2s)
        cost_rows.append({"T": T, "pairs_full": pairs_full, "pairs_block": pairs_block,
                          "ms_full": round(t_full*1000, 2), "ms_block": round(t_block*1000, 2)})
        print(f"  T={T:>4} | attn-pairs full {pairs_full:>9,} vs block {pairs_block:>7,} "
              f"({pairs_full/pairs_block:5.1f}x) | time {t_full*1000:6.1f}ms vs {t_block*1000:6.1f}ms")

    # --- 4. Streaming prefill: bounded per-block latency ---
    print("\n" + "-" * 70)
    print("Streaming prefill: per-block latency over a long audio")
    print("-" * 70)
    long_tokens = SEQ_LEN * 6
    _, long_audio, _ = make_example(long_tokens, VOCAB_SIZE, FRAMES_PER_TOKEN, rng)
    long_mel = torch.tensor(log_mel_spectrogram(long_audio)).unsqueeze(0)
    frames_total = long_mel.size(2)
    block_times = []
    x = model.encoder._frames(long_mel)
    with torch.no_grad():
        for i, s in enumerate(range(0, x.size(1), block2s)):
            chunk = x[:, s : s + block2s]
            t0 = time.time()
            model.encoder.transformer(chunk)
            block_times.append(round((time.time() - t0) * 1000, 2))
    print(f"  audio: {x.size(1)} encoder frames ({x.size(1)*FRAME_SECONDS:.1f}s) -> "
          f"{len(block_times)} blocks")
    print(f"  per-block latency (ms): {block_times}")
    print(f"  mean {np.mean(block_times):.2f} ms, max {np.max(block_times):.2f} ms — bounded")

    # --- Bake JSON for the visualization ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # a small block-diagonal mask sample for the diagram
    sample_frames = 12
    sample_block = 4
    mask_sample = (~block_diagonal_mask(sample_frames, sample_block)).int().tolist()
    payload = {
        "config": {"seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE, "d_model": D_MODEL,
                   "frame_ms": FRAME_SECONDS * 1000, "block_2s_frames": block2s,
                   "params": n_params},
        "acc_full": round(acc_full, 4),
        "approx": approx_rows,
        "streaming_max_dev": max_dev,
        "streaming_block_counts": counts,
        "cost": cost_rows,
        "prefill_block_ms": block_times,
        "prefill_frames": int(x.size(1)),
        "mask_sample": {"frames": sample_frames, "block": sample_block, "allowed": mask_sample},
    }
    out = DATA_DIR / "demo_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out} (open visualization/index.html to explore).")

    best = approx_rows[2]  # the 2 s block
    if best["frame_acc"] < acc_full - 0.05:
        raise SystemExit("Block-wise (2s) diverged too far from full attention.")
    if max_dev > 1e-4:
        raise SystemExit("Streaming (per-block) not equivalent to block-diagonal pass.")
    print("OK: 2 s block-wise attention approximates full attention, is streaming-"
          "equivalent, and scales linearly.")


if __name__ == "__main__":
    main()
