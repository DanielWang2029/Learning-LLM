"""Synthetic audio generator for the Uni-ASR streaming demo.

No audio is downloaded — everything is numpy. The task is audio -> token
transcription, but each token is deliberately built so its identity only becomes
clear in the *second half* of its span:

    token = [ onset | onset | nucleus | nucleus ]   (4 encoder frames, ~160 ms)

* the ONSET frames are a fixed tone, identical for every token (no information);
* the NUCLEUS frames carry the token's own frequency (all the information).

This mirrors the real problem Uni-ASR's fallback decoding targets: near a chunk
boundary a token's acoustic evidence has not fully arrived, so decoding it early
is a guess. Once the next chunk delivers the nucleus, re-decoding fixes it.

Run standalone to write a small preview:

    python data/synth_audio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 40.0                                  # one encoder frame (conv /4 of 10 ms hop)
SAMPLES_PER_FRAME = int(round(FRAME_MS / 1000.0 * SAMPLE_RATE))  # 640
FRAMES_PER_TOKEN = 4                             # onset(2) + nucleus(2)
ONSET_FRAMES = 2
ONSET_FREQ = 150.0                               # shared, information-free onset


def token_frequency(token_id: int) -> float:
    """Map a token id to its (well-separated) nucleus frequency."""
    return 260.0 * (2.0 ** (token_id / 3.0))


def _tone(freq: float, n_samples: int, phase0: float, rng) -> tuple[np.ndarray, float]:
    t = np.arange(n_samples) / SAMPLE_RATE
    wave = np.zeros(n_samples)
    for h, amp in enumerate((1.0, 0.5, 0.3), start=1):
        wave += amp * np.sin(2 * np.pi * freq * h * t + phase0 * h)
    wave += 0.01 * rng.standard_normal(n_samples)
    phase1 = (phase0 + 2 * np.pi * freq * n_samples / SAMPLE_RATE) % (2 * np.pi)
    peak = np.max(np.abs(wave)) or 1.0
    return (wave / peak).astype(np.float32), phase1


def token_waveform(token_id: int, rng) -> np.ndarray:
    """Render one token: fixed onset, then its identifying nucleus."""
    onset, ph = _tone(ONSET_FREQ, ONSET_FRAMES * SAMPLES_PER_FRAME, 0.0, rng)
    nucleus, _ = _tone(
        token_frequency(token_id),
        (FRAMES_PER_TOKEN - ONSET_FRAMES) * SAMPLES_PER_FRAME, ph, rng,
    )
    return np.concatenate([onset, nucleus])


def tokens_to_audio(tokens: list[int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    segs = [token_waveform(t, rng) for t in tokens]
    return np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)


def make_example(seq_len: int, vocab_size: int, rng: np.random.Generator):
    tokens = [int(rng.integers(0, vocab_size)) for _ in range(seq_len)]
    seed = int(rng.integers(0, 2**31 - 1))
    return tokens, tokens_to_audio(tokens, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(__file__).resolve().parent
    examples = []
    for _ in range(args.num_examples):
        tokens, audio = make_example(args.seq_len, args.vocab_size, rng)
        examples.append(
            {"tokens": tokens, "duration_s": round(audio.shape[0] / SAMPLE_RATE, 2),
             "encoder_frames": args.seq_len * FRAMES_PER_TOKEN}
        )
    payload = {
        "sample_rate": SAMPLE_RATE, "frame_ms": FRAME_MS,
        "frames_per_token": FRAMES_PER_TOKEN, "onset_frames": ONSET_FRAMES,
        "vocab_size": args.vocab_size, "examples": examples,
        "note": "token identity lives in the nucleus (second half) -> boundary ambiguity.",
    }
    (out_dir / "audio_preview.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} preview examples -> {out_dir / 'audio_preview.json'}")


if __name__ == "__main__":
    main()
