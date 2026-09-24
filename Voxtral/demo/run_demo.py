"""End-to-end Voxtral demo (CPU, < 60 s).

Trains the tiny Whisper-encoder + 4x-adapter + LLM-decoder pipeline on a
synthetic audio -> token transcription task, shows that it learns, and then
QUANTIFIES the adapter's central contribution: a 4x reduction in the number of
audio tokens the decoder sees (50 Hz -> 12.5 Hz) and what that buys in context
length (§2.2).

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)  # keep CPU timing stable and fast

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

from data.synth_audio import SAMPLE_RATE, make_example  # noqa: E402
from src import Voxtral, log_mel_spectrogram  # noqa: E402
from src.model import BOS, EOS  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

SEQ_LEN = 6
VOCAB_SIZE = 12
SEG_DURATION = 0.20
N_MELS = 128
ADAPTER_STRIDE = 4


def build_dataset(n: int, rng: np.random.Generator):
    mels, text_in, text_out = [], [], []
    for _ in range(n):
        tokens, audio = make_example(SEQ_LEN, VOCAB_SIZE, rng, SEG_DURATION)
        mels.append(log_mel_spectrogram(audio))
        text_in.append([BOS] + tokens)
        text_out.append(tokens + [EOS])
    mel = torch.tensor(np.stack(mels))                 # (n, n_mels, frames)
    return mel, torch.tensor(text_in), torch.tensor(text_out)


def evaluate(model, mel, text_out) -> tuple[float, float]:
    """Return (exact-sequence accuracy, per-token accuracy) via greedy decode."""
    model.eval()
    exact, tok_correct, tok_total = 0, 0, 0
    for i in range(mel.size(0)):
        pred = model.transcribe(mel[i : i + 1], max_len=SEQ_LEN + 2)
        target = text_out[i, :SEQ_LEN].tolist()
        if pred == target:
            exact += 1
        for a, b in zip(pred, target):
            tok_correct += int(a == b)
        tok_total += len(target)
    return exact / mel.size(0), tok_correct / tok_total


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    print("=" * 70)
    print("Voxtral demo — Whisper encoder + 4x adapter + LLM decoder")
    print("=" * 70)

    print("Synthesizing audio and computing log-Mel spectrograms...")
    train_mel, train_in, train_out = build_dataset(384, rng)
    eval_mel, eval_in, eval_out = build_dataset(64, rng)
    mel_frames = train_mel.size(2)
    print(
        f"  {train_mel.size(0)} train / {eval_mel.size(0)} eval clips | "
        f"{SEG_DURATION*SEQ_LEN:.1f}s each | mel: {N_MELS} bins x {mel_frames} frames"
    )

    model = Voxtral(
        vocab_size=VOCAB_SIZE,
        n_mels=N_MELS,
        d_encoder=64,
        d_llm=64,
        enc_layers=2,
        dec_layers=2,
        num_heads=4,
        d_ff=128,
        adapter_stride=ADAPTER_STRIDE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Report the frame rates at each stage on a real clip.
    with torch.no_grad():
        enc = model.encoder(eval_mel[:1])          # 50 Hz
        adapted = model.adapter(enc)               # 12.5 Hz
    t50, t125 = enc.size(1), adapted.size(1)
    print(f"  model parameters: {n_params:,}")
    print(
        f"  frames/clip: encoder(50Hz)={t50}  ->  adapter(12.5Hz)={t125}  "
        f"(x{t50/t125:.2f} fewer tokens into the decoder)\n"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()
    batch = 32
    steps = 800
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, train_mel.size(0), (batch,))
        logits = model(train_mel[idx], train_in[idx])
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), train_out[idx].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == 1:
            seq_acc, tok_acc = evaluate(model, eval_mel, eval_out)
            curve.append({"step": step, "loss": round(loss.item(), 4),
                          "seq_acc": round(seq_acc, 4), "tok_acc": round(tok_acc, 4)})
            print(f"  step {step:3d}/{steps} | loss {loss.item():.4f} | "
                  f"token-acc {tok_acc*100:5.1f}% | seq-acc {seq_acc*100:5.1f}%")
    elapsed = time.time() - start

    seq_acc, tok_acc = evaluate(model, eval_mel, eval_out)
    print(f"\nTrained {steps} steps in {elapsed:.1f}s")
    print(f"Final token accuracy    : {tok_acc*100:.1f}%")
    print(f"Final sequence accuracy : {seq_acc*100:.1f}%")

    # One concrete transcription example.
    pred = model.transcribe(eval_mel[:1], max_len=SEQ_LEN + 2)
    print("\nExample transcription")
    print(f"  target tokens : {eval_out[0, :SEQ_LEN].tolist()}")
    print(f"  decoded tokens: {pred}")

    # --- Quantify the 4x token reduction and its effect on context length ---
    print("\n" + "-" * 70)
    print("Adapter effect: audio tokens per minute and max audio in a 32k window")
    print("-" * 70)
    rate_50, rate_125 = 50.0, 50.0 / ADAPTER_STRIDE  # Hz
    context = 32_000
    print(f"  {'duration':>10} | {'50 Hz tokens':>13} | {'12.5 Hz tokens':>15}")
    duration_rows = []
    for minutes in (1, 10, 30, 40):
        secs = minutes * 60
        n50, n125 = int(secs * rate_50), int(secs * rate_125)
        duration_rows.append({"minutes": minutes, "tokens_50hz": n50, "tokens_125hz": n125})
        print(f"  {minutes:>7} min | {n50:>13,} | {n125:>15,}")
    max_min_50 = context / rate_50 / 60
    max_min_125 = context / rate_125 / 60
    print(f"\n  A 32k context holds ~{max_min_50:.1f} min at 50 Hz, "
          f"but ~{max_min_125:.1f} min after the 4x adapter.")

    # --- Bake a real spectrogram + everything the viz needs into JSON ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disp_mel = eval_mel[0].numpy()                       # (128, frames)
    # Downsample mel bins for a compact display grid (average pool to 32 bins).
    disp_bins = 32
    pooled = disp_mel.reshape(disp_bins, N_MELS // disp_bins, -1).mean(axis=1)
    payload = {
        "task": "synthetic audio -> token transcription",
        "config": {
            "seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE, "n_mels": N_MELS,
            "adapter_stride": ADAPTER_STRIDE, "seg_duration_s": SEG_DURATION,
            "params": n_params,
        },
        "frame_rates": {
            "encoder_hz": 50, "adapter_hz": 12.5,
            "encoder_frames": int(t50), "adapter_frames": int(t125),
            "reduction": round(t50 / t125, 3),
        },
        "final": {"token_acc": round(tok_acc, 4), "seq_acc": round(seq_acc, 4),
                  "train_seconds": round(elapsed, 1)},
        "training_curve": curve,
        "context_table": duration_rows,
        "context_window": context,
        "max_minutes_50hz": round(max_min_50, 2),
        "max_minutes_125hz": round(max_min_125, 2),
        "example": {"target": eval_out[0, :SEQ_LEN].tolist(), "decoded": pred},
        "spectrogram": {
            "n_bins": pooled.shape[0], "n_frames": pooled.shape[1],
            "values": np.round(pooled, 3).tolist(),
        },
    }
    out = DATA_DIR / "demo_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out} (open visualization/index.html to explore).")

    if tok_acc < 0.9:
        raise SystemExit(f"Model did not converge (token acc {tok_acc*100:.1f}% < 90%).")
    print("OK: encoder+adapter+decoder learned to transcribe, 4x token reduction confirmed.")


if __name__ == "__main__":
    main()
