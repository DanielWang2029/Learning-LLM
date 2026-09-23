"""Synthetic audio generator for the Voxtral reproduction.

No audio is ever downloaded. Everything here is generated from scratch with
numpy: pure tones, linear chirps and *formant stacks* (a fundamental plus a
few harmonics), which give a mel-spectrogram with clear, distinguishable
bands — perfect for a tiny audio->token transcription task.

The demo turns a sequence of integer *tokens* into a waveform: each token id
is rendered as a short formant-stack "syllable" whose fundamental frequency
encodes the id. Transcribing the audio therefore means reading back the token
sequence, which is exactly the job of Voxtral's encoder + adapter + decoder.

Run standalone to write a small human-readable preview to ``data/``:

    python data/synth_audio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # Whisper front-end operates at 16 kHz.

# Task vocabulary. Ids 0..2 are reserved control tokens; content tokens >= 3
# each map to a distinct fundamental frequency.
PAD, BOS, EOS = 0, 1, 2
NUM_SPECIAL = 3


def token_frequency(token_id: int) -> float:
    """Map a content token id to a fundamental frequency (Hz).

    Frequencies are spread geometrically across the speech range so adjacent
    ids stay well separated on the (log-scaled) mel axis.
    """
    k = token_id - NUM_SPECIAL
    return 200.0 * (2.0 ** (k / 3.0))  # third-octave steps: well separated on mel


def formant_stack(freq: float, duration: float, rng: np.random.Generator) -> np.ndarray:
    """A short voiced "syllable": a fundamental plus decaying harmonics."""
    n = int(round(duration * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    wave = np.zeros(n, dtype=np.float64)
    for h, amp in enumerate((1.0, 0.5, 0.33, 0.22), start=1):
        wave += amp * np.sin(2 * np.pi * freq * h * t)
    # Raised-cosine envelope avoids clicks at segment boundaries.
    env = np.hanning(n) if n > 1 else np.ones(n)
    wave *= env
    wave += 0.01 * rng.standard_normal(n)  # a touch of breathiness
    peak = np.max(np.abs(wave)) or 1.0
    return (wave / peak).astype(np.float32)


def chirp(f0: float, f1: float, duration: float) -> np.ndarray:
    """A linear frequency sweep — handy for spectrogram sanity checks."""
    n = int(round(duration * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    inst = f0 + (f1 - f0) * t / max(duration, 1e-9)
    phase = 2 * np.pi * np.cumsum(inst) / SAMPLE_RATE
    wave = np.sin(phase) * (np.hanning(n) if n > 1 else 1.0)
    return wave.astype(np.float32)


def tokens_to_audio(
    tokens: list[int],
    seg_duration: float = 0.20,
    seed: int = 0,
) -> np.ndarray:
    """Render a list of content tokens into one continuous waveform."""
    rng = np.random.default_rng(seed)
    segments = [
        formant_stack(token_frequency(tok), seg_duration, rng) for tok in tokens
    ]
    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


def make_example(
    seq_len: int,
    vocab_size: int,
    rng: np.random.Generator,
    seg_duration: float = 0.20,
):
    """Sample a random token sequence and its rendered waveform."""
    tokens = [
        int(rng.integers(NUM_SPECIAL, vocab_size)) for _ in range(seq_len)
    ]
    seed = int(rng.integers(0, 2**31 - 1))
    audio = tokens_to_audio(tokens, seg_duration=seg_duration, seed=seed)
    return tokens, audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for _ in range(args.num_examples):
        tokens, audio = make_example(args.seq_len, args.vocab_size, rng)
        examples.append(
            {
                "tokens": tokens,
                "num_samples": int(audio.shape[0]),
                "duration_s": round(audio.shape[0] / SAMPLE_RATE, 3),
                "token_freqs_hz": [round(token_frequency(t), 1) for t in tokens],
            }
        )

    payload = {
        "sample_rate": SAMPLE_RATE,
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS},
        "vocab_size": args.vocab_size,
        "note": "Audio is synthesized on the fly from these token sequences.",
        "examples": examples,
    }
    (out_dir / "audio_preview.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} preview examples -> {out_dir / 'audio_preview.json'}")


if __name__ == "__main__":
    main()
