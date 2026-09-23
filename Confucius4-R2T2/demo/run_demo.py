"""End-to-end CPU demo of the Confucius4-R2T2 streaming decoding paradigm.

It reproduces the model card's central claim — *append-only, true-streaming*
transcription via a Longest Stable Prefix (LSP) — and contrasts it with a naive
streaming decoder that revises earlier text (the flicker R2T2 removes).

Run with:  python demo/run_demo.py

The script:
  1. synthesizes a single-speaker "utterance" of formant phones (numpy),
  2. streams it chunk-by-chunk through a transparent matched-filter recognizer,
  3. commits text with LSP (append-only) and, separately, naively,
  4. reports revision counts, accuracy (1 - token error rate), and latency,
  5. sweeps chunk sizes from 80 ms to 2 s, and
  6. writes data/streaming_run.json for the visualization.
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

from src import (  # noqa: E402
    FRAMES_PER_PHONE,
    HOP,
    NUM_PHONES,
    SAMPLE_RATE,
    LSPDecoder,
    MatchedFilterRecognizer,
    NaiveDecoder,
    synth_utterance,
    tokens_to_text,
)

DATA_DIR = PAPER_DIR / "data"
FRAME_MS = HOP * 1000 // SAMPLE_RATE          # 10 ms / frame
PHONE_MS = FRAMES_PER_PHONE * FRAME_MS        # 200 ms / phone
CHUNK_SIZES_MS = [80, 160, 320, 560, 1120, 2000]


def edit_distance(a, b) -> int:
    """Levenshtein distance between two token sequences."""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def token_accuracy(hyp, ref) -> float:
    if not ref:
        return 1.0
    return 1.0 - edit_distance(hyp, ref) / len(ref)


def stream(wav, ref, chunk_ms, recognizer, decoder):
    """Feed `wav` to `decoder` in `chunk_ms` chunks; return per-chunk trace."""
    chunk_frames = max(1, chunk_ms // FRAME_MS)
    chunk_samples = chunk_frames * HOP
    total_samples = len(wav)
    trace = []
    chunk_index = 0
    pos = 0
    while pos < total_samples:
        chunk_index += 1
        pos = min(total_samples, pos + chunk_samples)
        hyp = recognizer.hypothesize(wav[:pos])
        newly = decoder.step(hyp, chunk_index, FRAMES_PER_PHONE)
        trace.append(
            {
                "chunk": chunk_index,
                "audio_ms": chunk_index * chunk_ms,
                "hypothesis": list(hyp),
                "committed": list(decoder.committed) if hasattr(decoder, "committed") else list(decoder.displayed),
                "newly_committed": newly if newly else [],
            }
        )
    # Flush trailing tokens once audio has ended.
    final_hyp = recognizer.hypothesize(wav)
    decoder.finalize(final_hyp, chunk_index, FRAMES_PER_PHONE)
    committed = decoder.committed if hasattr(decoder, "committed") else decoder.displayed
    return trace, list(committed)


def mean_latency_ms(emissions, chunk_ms) -> float:
    lats = []
    for e in emissions:
        commit_ms = e.chunk_index * chunk_ms
        audio_end_ms = e.audio_end_frame * FRAME_MS
        lats.append(max(0, commit_ms - audio_end_ms))
    return float(np.mean(lats)) if lats else 0.0


def main() -> None:
    rng = np.random.default_rng(7)
    n_phones = 12
    ref = [int(x) for x in rng.integers(0, NUM_PHONES, size=n_phones)]
    wav, _ = synth_utterance(ref, rng)
    recognizer = MatchedFilterRecognizer()

    print("=" * 74)
    print("Confucius4-R2T2  -  Longest Stable Prefix (LSP), append-only streaming")
    print("=" * 74)
    print(f"Synthesized utterance : {n_phones} phones, {len(wav)/SAMPLE_RATE:.1f}s "
          f"({PHONE_MS} ms/phone, {FRAME_MS} ms/frame)")
    print(f"Ground truth          : {tokens_to_text(ref)}\n")

    # ---- Detailed side-by-side run at 160 ms (the model card's headline chunk).
    demo_chunk = 160
    t0 = time.time()
    lsp = LSPDecoder(stability_window=2, unfixed_token_num=1)
    naive = NaiveDecoder()
    lsp_trace, lsp_final = stream(wav, ref, demo_chunk, recognizer, lsp)
    naive_trace, naive_final = stream(wav, ref, demo_chunk, recognizer, naive)

    print(f"Streaming at chunk = {demo_chunk} ms  (append-only LSP)")
    print("-" * 74)
    print(f"{'t (ms)':>7} | {'raw hypothesis (trailing edge unstable)':<40} | committed prefix")
    for row in lsp_trace:
        hyp = tokens_to_text(row["hypothesis"])
        com = tokens_to_text(row["committed"])
        mark = "  <- +" + tokens_to_text(row["newly_committed"]) if row["newly_committed"] else ""
        print(f"{row['audio_ms']:>7} | {hyp:<40} | {com}{mark}")

    print("\nAppend-only check: does any committed token ever change?")
    # Reconstruct committed history and verify monotonic (never revised).
    prev = []
    revised = False
    for row in lsp_trace:
        com = row["committed"]
        for i in range(min(len(prev), len(com))):
            if prev[i] != com[i]:
                revised = True
        prev = com
    print(f"  LSP revisions to already-emitted text : {lsp.revisions + (1 if revised else 0)}")
    print(f"  Naive decoder revisions (flicker)     : {naive.revisions}")

    print("\nFinal transcripts")
    print(f"  reference : {tokens_to_text(ref)}")
    print(f"  LSP       : {tokens_to_text(lsp_final)}   (acc {token_accuracy(lsp_final, ref)*100:.1f}%)")
    print(f"  naive     : {tokens_to_text(naive_final)}   (acc {token_accuracy(naive_final, ref)*100:.1f}%)")

    # ---- Chunk-size sweep: latency / accuracy / revisions trade-off.
    print("\nChunk-size sweep (one recognizer, no retraining)")
    print("-" * 74)
    print(f"{'chunk (ms)':>10} | {'LSP acc':>8} | {'LSP rev':>7} | {'LSP lat (ms)':>12} | "
          f"{'naive rev':>9} | {'naive lat':>9}")
    sweep = []
    for cms in CHUNK_SIZES_MS:
        lsp_d = LSPDecoder(stability_window=2, unfixed_token_num=1)
        naive_d = NaiveDecoder()
        _, lf = stream(wav, ref, cms, recognizer, lsp_d)
        _, nf = stream(wav, ref, cms, recognizer, naive_d)
        row = {
            "chunk_ms": cms,
            "lsp_acc": round(token_accuracy(lf, ref), 4),
            "lsp_revisions": lsp_d.revisions,
            "lsp_latency_ms": round(mean_latency_ms(lsp_d.emissions, cms), 1),
            "naive_acc": round(token_accuracy(nf, ref), 4),
            "naive_revisions": naive_d.revisions,
            "naive_latency_ms": round(mean_latency_ms(naive_d.emissions, cms), 1),
        }
        sweep.append(row)
        print(f"{cms:>10} | {row['lsp_acc']*100:>7.1f}% | {row['lsp_revisions']:>7} | "
              f"{row['lsp_latency_ms']:>12} | {row['naive_revisions']:>9} | {row['naive_latency_ms']:>9}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s on CPU.")

    total_naive_rev = sum(r["naive_revisions"] for r in sweep)
    total_lsp_rev = sum(r["lsp_revisions"] for r in sweep)
    assert total_lsp_rev == 0, "LSP must be append-only (0 revisions)"
    assert total_naive_rev > 0, "naive decoder should revise text"
    print("OK: LSP committed text is append-only (0 revisions) while staying accurate.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "reference": ref,
        "phone_names": [tokens_to_text([i]) for i in range(NUM_PHONES)],
        "frame_ms": FRAME_MS,
        "phone_ms": PHONE_MS,
        "demo_chunk_ms": demo_chunk,
        "lsp_trace": lsp_trace,
        "naive_trace": naive_trace,
        "lsp_final": lsp_final,
        "naive_final": naive_final,
        "lsp_revisions": total_lsp_rev,
        "naive_revisions_demo": naive.revisions,
        "sweep": sweep,
    }
    (DATA_DIR / "streaming_run.json").write_text(json.dumps(out, indent=2))
    print("Wrote data/streaming_run.json  ->  open visualization/index.html")


if __name__ == "__main__":
    main()
