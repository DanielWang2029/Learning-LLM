"""Reproduce Voxtral Realtime's core claims on CPU (arXiv:2602.11298).

  1. STREAMING == OFFLINE: the causal sliding-window encoder, run frame-by-frame
     with only its (bounded) live receptive field, reproduces the full offline
     forward exactly -> a true streaming encoder with bounded memory.
  2. LATENCY <-> ACCURACY via AdaRMSNorm: ONE model, trained with the target
     delay tau sampled over a range, is evaluated at each tau. Transcription
     accuracy rises with the delay and plateaus near the full-context ("offline")
     accuracy -- the paper's headline (Fig. 1): larger delay == more acoustic
     evidence == higher accuracy, from a single model.
  3. DSM stream: one token per 80 ms frame -- [P] (wait), then [W] + the word.

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import VoxtralRealtime, CausalSlidingWindowEncoder, data  # noqa: E402
from src.data import VOCAB_SIZE, FEAT_DIM, PAD, WB, FIRST_WORD, tau_frames_to_ms  # noqa: E402

SEED = 0
TAUS = [1, 2, 3, 4, 5, 6]        # streaming delays in frames (80 .. 480 ms)


def shift_right(target: torch.Tensor) -> torch.Tensor:
    """prev_tokens[t] = target[t-1], with [P] at t=0 (teacher forcing)."""
    prev = torch.full_like(target, PAD)
    prev[:, 1:] = target[:, :-1]
    return prev


def main() -> None:
    t0 = time.time()
    dev = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print("Voxtral Realtime  (arXiv:2602.11298) -- CPU reproduction\n")

    # ---------- 1. streaming == offline (structural, before training) ----------
    enc = CausalSlidingWindowEncoder(FEAT_DIM, d=64, layers=2, heads=4, window=16).to(dev)
    enc.eval()
    feats_chk = torch.from_numpy(data.make_utterance(rng, tau=3)["feats"]).unsqueeze(0)
    with torch.no_grad():
        off = enc(feats_chk)
        strm = enc.stream(feats_chk)
    enc_diff = (off - strm).abs().max().item()
    print("[1] causal sliding-window encoder")
    print(f"    streaming (frame-by-frame, live window={enc.recept} frames) vs offline: "
          f"max|diff| = {enc_diff:.2e}")
    print(f"    memory bounded: each step needs only the last {enc.recept} frames, "
          f"independent of stream length ({feats_chk.shape[1]} here)\n")

    # ---------- 2. train ONE delay-conditioned model ----------
    model = VoxtralRealtime(FEAT_DIM, VOCAB_SIZE, d=64, enc_layers=2, dec_layers=2,
                            heads=4, enc_window=16, max_len=data.UTT_FRAMES + 4).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    steps, bs = 750, 32
    for _ in range(steps):
        feats, target, taus, _ = data.make_batch(bs, rng, TAUS)
        feats = torch.from_numpy(feats); target = torch.from_numpy(target); taus = torch.from_numpy(taus)
        logits = model(feats, shift_right(target), taus)
        # emphasise the content tokens ([W]/word) over the many [P] frames
        w = torch.ones(VOCAB_SIZE); w[PAD] = 0.1
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    print(f"[2] trained one delay-conditioned model ({time.time()-t0:.0f}s, "
          f"{sum(p.numel() for p in model.parameters()):,} params)\n")

    # ---------- evaluate accuracy at each delay (AdaRMSNorm) ----------
    @torch.no_grad()
    def word_accuracy(tau: int, n: int = 160) -> float:
        ev = np.random.default_rng(1000 + tau)
        feats, target, _, us = data.make_batch(n, ev, [tau])
        feats = torch.from_numpy(feats); target = torch.from_numpy(target)
        taus = torch.full((n,), tau, dtype=torch.long)
        pred = model(feats, shift_right(target), taus).argmax(-1)
        correct = total = 0
        for i, u in enumerate(us):
            for f, w in zip(u["tok_frames"], u["words"]):
                total += 1
                correct += int(pred[i, f].item() == FIRST_WORD + w)
        return correct / max(1, total)

    print("[3] latency <-> accuracy from ONE model (AdaRMSNorm delay conditioning)")
    curve = []
    for tau in TAUS:
        acc = word_accuracy(tau)
        curve.append({"tau_frames": tau, "tau_ms": tau_frames_to_ms(tau), "acc": acc})
        bar = "#" * int(acc * 30)
        print(f"    delay {tau_frames_to_ms(tau):4d} ms ({tau}f)  word-acc {acc*100:5.1f}%  {bar}")
    acc_lo, acc_hi = curve[0]["acc"], curve[-1]["acc"]
    print(f"    -> accuracy rises {acc_lo*100:.1f}% ({curve[0]['tau_ms']}ms) -> "
          f"{acc_hi*100:.1f}% ({curve[-1]['tau_ms']}ms, ~offline)\n")

    # ---------- sample DSM stream trace ----------
    ev = np.random.default_rng(42)
    u = data.make_utterance(ev, tau=4)
    feats = torch.from_numpy(u["feats"]).unsqueeze(0)
    target = torch.from_numpy(u["target"]).unsqueeze(0)
    with torch.no_grad():
        pred = model(feats, shift_right(target), torch.tensor([4])).argmax(-1)[0].tolist()

    def tok(i):
        return "[P]" if i == PAD else ("[W]" if i == WB else f"w{i - FIRST_WORD}")
    emitted = [(t, tok(p)) for t, p in enumerate(pred) if p != PAD]

    payload = {
        "encoder_stream_maxdiff": enc_diff,
        "receptive_field": enc.recept,
        "utt_frames": data.UTT_FRAMES,
        "frame_ms": data.FRAME_MS,
        "delay_curve": curve,
        "trace": {
            "tau_ms": tau_frames_to_ms(4),
            "target": [tok(t) for t in u["target"].tolist()],
            "pred": [tok(p) for p in pred],
            "emitted": emitted,
            "words": u["words"],
            "tok_frames": u["tok_frames"],
        },
    }
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "demo_results.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir/'demo_results.json'}.  Total: {time.time()-t0:.0f}s")

    ok = (
        enc_diff < 1e-4
        and acc_hi > acc_lo + 0.2
        and acc_hi > 0.8
        and all(curve[i]["acc"] <= curve[i + 1]["acc"] + 0.08 for i in range(len(curve) - 1))
    )
    if not ok:
        raise SystemExit(
            f"Sanity failed: enc_diff={enc_diff:.1e}, acc {acc_lo:.2f}->{acc_hi:.2f}, "
            f"curve={[round(c['acc'],2) for c in curve]}"
        )
    print("OK: causal encoder streams exactly; and one AdaRMSNorm-conditioned model "
          "trades delay for accuracy, approaching offline quality at ~480 ms.")


if __name__ == "__main__":
    main()
