"""Synthetic audio generator for the Qwen2.5-Omni streaming-encoder demo.

No audio is downloaded — everything is generated with numpy: pure tones,
chirps and formant stacks. The streaming demo turns a sequence of integer
tokens into one long waveform, where each token is a short formant-stack
"syllable" whose fundamental frequency encodes its id. The block-wise encoder
then has to recognise each frame's token, in independent ~2-second blocks.

Run standalone to write a small preview:

    python data/synth_audio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

# Encoder frames are ~40 ms (10 ms hop, conv /4). We render each token as an
# integer number of encoder frames so per-frame labels are exact.
FRAME_MS = 40.0


def token_frequency(token_id: int) -> float:
    """Map a token id to a well-separated fundamental frequency (Hz)."""
    return 200.0 * (2.0 ** (token_id / 3.0))  # third-octave steps


def formant_stack(freq: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """A voiced syllable: fundamental + decaying harmonics + a little noise."""
    t = np.arange(n_samples) / SAMPLE_RATE
    wave = np.zeros(n_samples, dtype=np.float64)
    for h, amp in enumerate((1.0, 0.5, 0.33, 0.22), start=1):
        wave += amp * np.sin(2 * np.pi * freq * h * t)
    wave += 0.01 * rng.standard_normal(n_samples)
    peak = np.max(np.abs(wave)) or 1.0
    return (wave / peak).astype(np.float32)


def chirp(f0: float, f1: float, duration: float) -> np.ndarray:
    """A linear frequency sweep — for spectrogram sanity checks."""
    n = int(round(duration * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    inst = f0 + (f1 - f0) * t / max(duration, 1e-9)
    phase = 2 * np.pi * np.cumsum(inst) / SAMPLE_RATE
    return (np.sin(phase) * (np.hanning(n) if n > 1 else 1.0)).astype(np.float32)


def tokens_to_audio(
    tokens: list[int],
    frames_per_token: int,
    seed: int = 0,
) -> np.ndarray:
    """Render tokens into one continuous waveform (constant amplitude, no gaps).

    Each token occupies exactly ``frames_per_token`` encoder frames so that a
    per-frame label array lines up cleanly with the encoder output.
    """
    rng = np.random.default_rng(seed)
    samples_per_frame = int(round(FRAME_MS / 1000.0 * SAMPLE_RATE))  # 640
    n = frames_per_token * samples_per_frame
    segs = [formant_stack(token_frequency(tok), n, rng) for tok in tokens]
    return np.concatenate(segs) if segs else np.zeros(0, dtype=np.float32)


def make_example(
    seq_len: int,
    vocab_size: int,
    frames_per_token: int,
    rng: np.random.Generator,
):
    """Sample a token sequence, its waveform, and per-encoder-frame labels."""
    tokens = [int(rng.integers(0, vocab_size)) for _ in range(seq_len)]
    seed = int(rng.integers(0, 2**31 - 1))
    audio = tokens_to_audio(tokens, frames_per_token, seed=seed)
    labels = np.repeat(tokens, frames_per_token)  # one label per encoder frame
    return tokens, audio, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=10)
    parser.add_argument("--frames-per-token", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(__file__).resolve().parent

    examples = []
    for _ in range(args.num_examples):
        tokens, audio, _ = make_example(
            args.seq_len, args.vocab_size, args.frames_per_token, rng
        )
        examples.append(
            {
                "tokens": tokens,
                "duration_s": round(audio.shape[0] / SAMPLE_RATE, 2),
                "encoder_frames": args.seq_len * args.frames_per_token,
            }
        )
    payload = {
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "vocab_size": args.vocab_size,
        "examples": examples,
    }
    (out_dir / "audio_preview.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(examples)} preview examples -> {out_dir / 'audio_preview.json'}")


if __name__ == "__main__":
    main()
