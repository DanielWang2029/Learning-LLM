"""Uni-ASR demo (CPU, < 60 s): unified non-streaming + streaming with fallback.

Trains ONE tiny model (causal Conformer encoder + adapter + LLM decoder) jointly
for non-streaming and streaming, using interleaved / truncated-prefix examples
with loss masks (the SS + context-aware paradigm, §2.2–§2.3). Then it decodes the
same audio three ways and reports token accuracy:

* non-streaming (full audio),
* streaming, naive chunked decoding, and
* streaming, with the latest-token FALLBACK re-decode (§2.3),

showing the fallback recovers the boundary errors that naive decoding makes — at
equal latency — across a range of chunk sizes.

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

from data.synth_audio import FRAME_MS, FRAMES_PER_TOKEN, make_example  # noqa: E402
from src import IGNORE, UniASR, log_mel_spectrogram  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

N_TOKENS = 8
VOCAB = 8
W = FRAMES_PER_TOKEN          # 4 encoder frames per token
T_FRAMES = N_TOKENS * W       # 32 encoder frames
N_MELS = 80


def build_dataset(n, rng):
    mels, toks = [], []
    for _ in range(n):
        t, audio = make_example(N_TOKENS, VOCAB, rng)
        mels.append(log_mel_spectrogram(audio))
        toks.append(t)
    return torch.tensor(np.stack(mels)), torch.tensor(toks).long()


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    print("=" * 70)
    print("Uni-ASR demo — unified non-streaming + streaming ASR with fallback")
    print("=" * 70)
    print(f"token = {W} encoder frames ({W*FRAME_MS:.0f} ms): 2 onset (shared) + 2 nucleus (identity)")

    train_mel, train_tok = build_dataset(256, rng)
    eval_mel, eval_tok = build_dataset(48, rng)
    print(f"  {train_mel.size(0)} train / {eval_mel.size(0)} eval utterances | "
          f"{N_TOKENS} tokens ({T_FRAMES} frames, {T_FRAMES*FRAME_MS/1000:.2f}s) each\n")

    model = UniASR(VOCAB, W, n_mels=N_MELS, d_model=96, d_llm=96,
                   enc_layers=3, dec_layers=2, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    crit = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    batch, steps = 32, 300

    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, train_mel.size(0), (batch,))
        frames = model.encode(train_mel[idx])            # (B, T', d), causal
        if step % 2 == 0:
            seen = T_FRAMES                               # non-streaming
        else:
            seen = int(torch.randint(2, T_FRAMES + 1, (1,)))  # streaming truncation
        seq, targets = model.training_batch(frames, train_tok[idx], seen, True)
        logits = model.decoder(seq)
        loss = crit(logits.reshape(-1, VOCAB), targets.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 60 == 0 or step == 1:
            print(f"  step {step:3d}/{steps} | loss {loss.item():.4f} (seen={seen})")
    print(f"  trained in {time.time()-start:.1f}s\n")

    # --- Evaluation ---
    def token_acc(pred_list, gold):
        c = t = 0
        for p, g in zip(pred_list, gold.tolist()):
            for a, b in zip(p, g):
                c += int(a == b); t += 1
        return c / t

    model.eval()
    ns_preds = [model.decode_full(eval_mel[i:i+1], N_TOKENS) for i in range(eval_mel.size(0))]
    ns_acc = token_acc(ns_preds, eval_tok)
    print("-" * 70)
    print(f"Non-streaming accuracy (full audio): {ns_acc*100:.1f}%")
    print("-" * 70)

    print("\nStreaming: naive vs latest-token fallback, by chunk size (equal latency)")
    print(f"  {'chunk':>8} | {'naive acc':>10} | {'fallback acc':>12} | {'naive err':>9} | {'fallback err':>12}")
    rows = []
    chunk_frames_list = [2, 3, 4, 5, 6, 8]
    for C in chunk_frames_list:
        naive = [model.decode_stream(eval_mel[i:i+1], N_TOKENS, C, fallback=False)
                 for i in range(eval_mel.size(0))]
        fb = [model.decode_stream(eval_mel[i:i+1], N_TOKENS, C, fallback=True)
              for i in range(eval_mel.size(0))]
        na, fa = token_acc(naive, eval_tok), token_acc(fb, eval_tok)
        rows.append({"chunk_frames": C, "chunk_ms": C * FRAME_MS,
                     "naive_acc": round(na, 4), "fallback_acc": round(fa, 4),
                     "naive_err": round(1 - na, 4), "fallback_err": round(1 - fa, 4)})
        print(f"  {C*FRAME_MS:5.0f}ms | {na*100:9.1f}% | {fa*100:11.1f}% | "
              f"{(1-na)*100:8.1f}% | {(1-fa)*100:11.1f}%")

    mean_naive = float(np.mean([r["naive_err"] for r in rows]))
    mean_fb = float(np.mean([r["fallback_err"] for r in rows]))
    print(f"\n  mean streaming error — naive {mean_naive*100:.1f}%  vs  fallback {mean_fb*100:.1f}%")
    reduction = (mean_naive - mean_fb) / mean_naive * 100 if mean_naive > 0 else 0.0
    print(f"  fallback cuts streaming token error by {reduction:.0f}% (relative)")

    # concrete example at a hard (misaligned) chunk size
    C = 3
    i = 0
    ex = {
        "tokens": eval_tok[0].tolist(),
        "chunk_frames": C,
        "naive": model.decode_stream(eval_mel[i:i+1], N_TOKENS, C, fallback=False),
        "fallback": model.decode_stream(eval_mel[i:i+1], N_TOKENS, C, fallback=True),
        "nonstreaming": ns_preds[0],
    }
    print(f"\nExample (chunk={C*FRAME_MS:.0f}ms)")
    print(f"  gold        : {ex['tokens']}")
    print(f"  non-stream  : {ex['nonstreaming']}")
    print(f"  naive       : {ex['naive']}")
    print(f"  fallback    : {ex['fallback']}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"n_tokens": N_TOKENS, "vocab": VOCAB, "frames_per_token": W,
                   "frame_ms": FRAME_MS, "params": n_params},
        "nonstreaming_acc": round(ns_acc, 4),
        "by_chunk": rows,
        "mean_naive_err": round(mean_naive, 4),
        "mean_fallback_err": round(mean_fb, 4),
        "error_reduction_pct": round(reduction, 1),
        "example": ex,
    }
    out = DATA_DIR / "demo_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out} (open visualization/index.html to explore).")

    if mean_fb >= mean_naive or ns_acc < 0.9:
        raise SystemExit("Demo did not demonstrate the fallback benefit clearly.")
    print("OK: one model does non-streaming + streaming; fallback cuts streaming errors "
          "at equal latency.")


if __name__ == "__main__":
    main()
