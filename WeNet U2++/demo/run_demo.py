"""U2++ demo: one shared model, streaming CTC + attention rescoring.

On CPU, in well under a minute, this script:

  1. synthesizes an audio->token task where the tokens A and B are acoustically
     identical and only *label context* (C->A, D->B) tells them apart,
  2. trains ONE U2++ model (shared Conformer encoder + CTC head + L2R & R2L
     attention decoders) with DYNAMIC CHUNK MASKING,
  3. shows the same weights run full-context and streaming (a chunk-size sweep),
  4. shows the second-pass ATTENTION RESCORING fixes the streaming CTC
     hypotheses that the frame-independent first pass gets wrong.

Because CTC is conditionally independent per frame, it guesses A vs B roughly at
chance; the autoregressive attention decoders resolve them from context, so
rescoring sharply lowers the token error rate.

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

from data.audio_synth import TOKENS, log_mel, synth_utterance  # noqa: E402
from src import U2PP, token_error_rate  # noqa: E402

DATA_DIR = PAPER_DIR / "data"
N_MELS, N_FFT, HOP = 40, 400, 160


def build_dataset(n_utts, num_pairs, seg_dur, seed):
    rng = np.random.default_rng(seed)
    feats, toks = [], []
    for _ in range(n_utts):
        wave, tokens = synth_utterance(num_pairs, seg_dur, rng)
        mel, _ = log_mel(wave, N_FFT, HOP, N_MELS)
        feats.append(mel)
        toks.append(tokens)
    T = min(f.shape[0] for f in feats)
    feats = np.stack([f[:T] for f in feats])
    toks = np.stack(toks)  # fixed length 2*num_pairs
    return torch.from_numpy(feats), torch.from_numpy(toks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--num-pairs", type=int, default=4)   # -> 8 tokens per utt
    p.add_argument("--seg-dur", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=55)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--enc-blocks", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--stream-chunk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("WeNet U2++  -  streaming CTC first pass + attention rescoring")
    print("=" * 72)
    train_x, train_tok = build_dataset(args.n_train, args.num_pairs, args.seg_dur, args.seed)
    test_x, test_tok = build_dataset(args.n_test, args.num_pairs, args.seg_dur, args.seed + 1)
    T, L = train_x.size(1), train_tok.size(1)
    refs = [test_tok[i].tolist() for i in range(test_tok.size(0))]
    # Dynamic chunk sizes spanning streaming to full context. Full context (T)
    # is over-sampled so the offline mode -- the accuracy ceiling -- is trained
    # at least as thoroughly as the streaming ones.
    train_chunks = [4, 8, 16, T, T, T]
    print(f"Frames/utt T={T}, tokens/utt L={L}. Tokens: {TOKENS} "
          f"(A,B differ only by a weak noisy cue; grammar C->A, D->B).")
    print(f"Dynamic training chunks: {train_chunks}\n")

    torch.manual_seed(args.seed)
    model = U2PP(n_mels=N_MELS, d_model=args.d_model, enc_blocks=args.enc_blocks,
                 dec_layers=args.dec_layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tok_len = torch.full((args.batch_size,), L, dtype=torch.long)

    print("[1/3] Training U2++ (CTC + L2R + R2L) with dynamic chunk masking ...")
    n = train_x.size(0)
    rng = np.random.default_rng(args.seed)
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        ep_ctc = ep_att = 0.0
        nb = 0
        for i in range(0, n - args.batch_size + 1, args.batch_size):
            idx = perm[i : i + args.batch_size]
            chunk = int(train_chunks[rng.integers(len(train_chunks))])
            loss, ctc, att = model.loss(train_x[idx], train_tok[idx], tok_len,
                                        chunk_size=chunk, ctc_weight=args.ctc_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_ctc += ctc; ep_att += att; nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      epoch {epoch+1:3d}/{args.epochs} | ctc {ep_ctc/nb:.3f} "
                  f"| att {ep_att/nb:.3f}")

    model.eval()

    # ---- [2/3] one model, offline vs streaming (CTC greedy) ----------------
    print("\n[2/3] Same weights, full-context vs streaming (CTC greedy) ...")
    sweep = []
    for chunk in [4, 8, 16, T]:
        hyps = model.ctc_greedy(test_x, chunk_size=chunk)
        ter, acc = token_error_rate(hyps, refs)
        sweep.append({"chunk": int(chunk), "ter": ter, "seq_acc": acc,
                      "label": "full/offline" if chunk >= T else f"chunk={chunk}"})
    print(f"      {'mode':<16}{'token err':>12}{'seq acc':>10}")
    for s in sweep:
        print(f"      {s['label']:<16}{s['ter']*100:>11.1f}%{s['seq_acc']*100:>9.1f}%")

    # ---- [3/3] streaming CTC vs attention rescoring ------------------------
    print(f"\n[3/3] Attention rescoring at streaming chunk={args.stream_chunk} ...")
    greedy_hyps, rescored_hyps = model.rescore(
        test_x, chunk_size=args.stream_chunk, beam_size=8, n_best=4,
        ctc_weight=args.ctc_weight, att_weight=1.0)
    ter_g, acc_g = token_error_rate(greedy_hyps, refs)
    ter_r, acc_r = token_error_rate(rescored_hyps, refs)
    rel = 100 * (ter_g - ter_r) / max(ter_g, 1e-9)

    print("\n" + "-" * 64)
    print(f"{'decode mode':<34}{'token err':>14}{'seq acc':>12}")
    print("-" * 64)
    print(f"{'CTC greedy (streaming)':<34}{ter_g*100:>13.1f}%{acc_g*100:>11.1f}%")
    print(f"{'+ attention rescoring (U2++)':<34}{ter_r*100:>13.1f}%{acc_r*100:>11.1f}%")
    print("-" * 64)
    print(f"\nRescoring cuts streaming token error by {rel:.0f}% relative "
          f"({ter_g*100:.1f}% -> {ter_r*100:.1f}%).")

    # a concrete example where rescoring fixed the hypothesis
    example = None
    for g, r, ref in zip(greedy_hyps, rescored_hyps, refs):
        if g != ref and r == ref:
            example = {"reference": ref, "ctc_greedy": g, "rescored": r,
                       "ref_tokens": [TOKENS[t] for t in ref],
                       "greedy_tokens": [TOKENS[t] for t in g],
                       "rescored_tokens": [TOKENS[t] for t in r]}
            break
    if example:
        print("\nExample fixed by rescoring:")
        print(f"  reference : {' '.join(example['ref_tokens'])}")
        print(f"  CTC greedy: {' '.join(example['greedy_tokens'])}")
        print(f"  rescored  : {' '.join(example['rescored_tokens'])}")

    out = {
        "config": vars(args), "frames": T, "tokens_per_utt": L,
        "tokens": TOKENS,
        "chunk_sweep_ctc": sweep,
        "rescoring": {
            "stream_chunk": args.stream_chunk,
            "ctc_greedy": {"ter": ter_g, "seq_acc": acc_g},
            "rescored": {"ter": ter_r, "seq_acc": acc_r},
            "relative_ter_reduction": rel,
        },
        "example_fixed": example,
    }
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "demo_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/demo_results.json  |  total time {time.time()-t0:.1f}s")

    ok = ter_r < ter_g and acc_r >= acc_g
    print("OK: one model streams and offline-decodes; rescoring improves CTC."
          if ok else "WARNING: check results above.")


if __name__ == "__main__":
    main()
