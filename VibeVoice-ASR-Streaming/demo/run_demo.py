"""End-to-end CPU demo of VibeVoice-ASR-Streaming.

Reproduces the paper's core idea (arXiv:2609.02812, §3.1): LLM-based *streaming
speaker-attributed ASR* that interleaves fixed-size audio chunks, a small
lookahead, and previously generated text, emitting "who said what" incrementally
with a chunk-end token — no separate diarization stage.

Run with:  python demo/run_demo.py

Pipeline:
  1. synthesize two-speaker conversations (two distinct synthetic timbres),
  2. train a tiny interleaved speech-text Transformer (teacher forced),
  3. stream a held-out conversation chunk-by-chunk and print "who said what",
  4. report transcription + speaker-attribution accuracy over a test set,
  5. ablate the lookahead to show it helps at chunk boundaries,
  6. write data/streaming_run.json for the visualization.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from src import (  # noqa: E402
    FRAME_MS,
    N_MELS,
    VOCAB,
    StreamingSAASR,
    features,
    gen_conversation,
    make_dataset,
    score,
    stream_decode,
    synth_conversation,
    token_str,
)
from src.audio import PHONE_NAMES  # noqa: E402
from src.data_gen import PAD  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

N_CHUNKS = 6
CHUNK_PHONES = 3
LOOKAHEAD = 1          # future latent frames (~ the paper's 4-frame / 0.5 s look-ahead)
SEED = 0


def collate(batch):
    """Pad a list of (audio, text, is_audio, target, tmask) to a batch."""
    S = max(a.shape[0] for a, *_ in batch)
    B = len(batch)
    audio = np.zeros((B, S, N_MELS), dtype=np.float32)
    text = np.zeros((B, S), dtype=np.int64)
    isaud = np.zeros((B, S), dtype=bool)
    target = np.full((B, S), PAD, dtype=np.int64)
    tmask = np.zeros((B, S), dtype=bool)
    pad = np.ones((B, S), dtype=bool)
    for i, (a, t, m, tg, tm) in enumerate(batch):
        s = a.shape[0]
        audio[i, :s] = a; text[i, :s] = t; isaud[i, :s] = m
        target[i, :s] = tg; tmask[i, :s] = tm; pad[i, :s] = False
    return (torch.tensor(audio), torch.tensor(text), torch.tensor(isaud),
            torch.tensor(target), torch.tensor(tmask), torch.tensor(pad))


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("=" * 76)
    print("VibeVoice-ASR-Streaming  -  LLM-based streaming speaker-attributed ASR")
    print("=" * 76)
    print(f"config: {N_CHUNKS} chunks x {CHUNK_PHONES} phones, lookahead={LOOKAHEAD} frame(s), "
          f"frame={FRAME_MS} ms, vocab={VOCAB}")

    # ---- 1-2. data + training
    train = make_dataset(280, rng, N_CHUNKS, CHUNK_PHONES, LOOKAHEAD, N_MELS)
    model = StreamingSAASR(N_MELS, VOCAB, d_model=96, n_layers=2, n_heads=4, d_ff=192)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    print(f"model parameters: {n_params:,}\n")

    steps, bs = 500, 32
    print("training (teacher-forced interleaved speech-text)")
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train), size=bs)
        audio, text, isaud, target, tmask, pad = collate([train[i] for i in idx])
        logits = model(audio, text, isaud, pad_mask=pad)
        loss = crit(logits[tmask], target[tmask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:3d}/{steps} | loss {loss.item():.4f}")

    # ---- 3. detailed streaming run on a fresh conversation
    ex_rng = np.random.default_rng(123)
    phones, raw, ords = gen_conversation(ex_rng, N_CHUNKS, CHUNK_PHONES)
    wav, nf = synth_conversation(phones, raw, ex_rng)
    feats = features(wav, nf)
    emissions, pred_phones, trace = stream_decode(
        model, feats, phones, ords, CHUNK_PHONES, LOOKAHEAD, with_lookahead=True)

    print("\nStreaming 'who said what' (emitted incrementally, chunk by chunk)")
    print("-" * 76)
    running = ""
    for row in trace:
        a0, a1 = row["audio_ms"]
        txt = row["text"] if row["text"] else "(no speech)"
        print(f"  t=[{a0:>5},{a1:>5}] ms  chunk {row['chunk']}  ->  {txt!r}")
    print("\nReconstructed predicted transcript:")
    pred_str, last = [], -1
    for o, p in pred_phones:
        if o != last:
            pred_str.append(f"\n  Speaker{o}:")
            last = o
        pred_str.append(" " + PHONE_NAMES[p])
    print("".join(pred_str))

    sc = score(pred_phones, phones, ords)
    print(f"\n  example transcription acc = {sc['transcription_acc']*100:.1f}%  "
          f"speaker-attribution acc = {sc['speaker_acc']*100:.1f}%")

    # ---- 4. test-set metrics
    test_rng = np.random.default_rng(999)
    def eval_set(with_la):
        ts, ss, n = 0.0, 0.0, 0
        for _ in range(40):
            ph, rw, od = gen_conversation(test_rng, N_CHUNKS, CHUNK_PHONES)
            w, f = synth_conversation(ph, rw, test_rng)
            ft = features(w, f)
            _, pp, _ = stream_decode(model, ft, ph, od, CHUNK_PHONES, LOOKAHEAD, with_lookahead=with_la)
            s = score(pp, ph, od)
            ts += s["transcription_acc"]; ss += s["speaker_acc"]; n += 1
        return ts / n, ss / n

    tr_la, sp_la = eval_set(True)
    tr_no, sp_no = eval_set(False)
    print("\nTest set (40 conversations)")
    print("-" * 76)
    print(f"  with lookahead    : transcription {tr_la*100:5.1f}%   speaker {sp_la*100:5.1f}%")
    print(f"  without lookahead : transcription {tr_no*100:5.1f}%   speaker {sp_no*100:5.1f}%")
    print(f"  lookahead gain (transcription): +{(tr_la - tr_no)*100:.1f} points")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s on CPU.")
    assert tr_la > 0.9, "streaming transcription should be accurate with lookahead"
    assert sp_la > 0.9, "streaming speaker attribution should be accurate"
    assert tr_la >= tr_no, "lookahead should not hurt transcription"
    print("OK: streaming speaker-attributed ASR works; lookahead helps at boundaries.")

    # ---- 5. write viz data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # reference per chunk for the timeline
    ref_chunks = []
    last = -1
    for k in range(N_CHUNKS):
        lo = k * CHUNK_PHONES
        toks = []
        for i in range(lo, lo + CHUNK_PHONES):
            if ords[i] != last:
                toks.append(f"Speaker{ords[i]}")
                last = ords[i]
            toks.append(PHONE_NAMES[phones[i]])
        ref_chunks.append(toks)

    out = {
        "config": {"n_chunks": N_CHUNKS, "chunk_phones": CHUNK_PHONES,
                   "lookahead": LOOKAHEAD, "frame_ms": FRAME_MS,
                   "frames_per_chunk": 2 * CHUNK_PHONES - 1},
        "example": {
            "phones": phones, "ords": ords,
            "ref_chunks": ref_chunks,
            "trace": trace,
            "mel": np.round(feats, 3).tolist(),
            "transcription_acc": round(sc["transcription_acc"], 4),
            "speaker_acc": round(sc["speaker_acc"], 4),
        },
        "metrics": {
            "with_lookahead": {"transcription": round(tr_la, 4), "speaker": round(sp_la, 4)},
            "without_lookahead": {"transcription": round(tr_no, 4), "speaker": round(sp_no, 4)},
        },
    }
    (DATA_DIR / "streaming_run.json").write_text(json.dumps(out))
    print("Wrote data/streaming_run.json  ->  open visualization/index.html")


if __name__ == "__main__":
    main()
