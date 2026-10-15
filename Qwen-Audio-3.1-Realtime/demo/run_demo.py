"""Reproduce Qwen-Audio-3.1-Realtime's foundation recipe on CPU (arXiv:2609.25176).

The paper's foundation training (§3) turns a text-pretrained LLM into a reliable
*audio* policy with:
  Core-Cocktail SFT -> Multimodality OPD -> (domain GRPO experts) -> Multi-Teacher OPD,
collectively "M2-OPD" -- on-policy distillation that scores the student on its
OWN generated trajectories, using a Text Teacher for text-representable skills
and a frozen Audio Reference for audio-only behaviour (emotion/acoustic scene).

This demo reproduces the two headline mechanisms on a synthetic audio task:
  [A] ON-POLICY beats OFF-POLICY. On-policy OPD supervises the student on the
      states it actually drifts into, so it stays reliable under its own
      trajectory; off-policy SFT (gold states only) degrades (exposure bias).
  [B] The AUDIO REFERENCE preserves information beyond the transcript. The Text
      Teacher cannot supervise the audio-only STYLE; text-only OPD forgets it,
      while M2-OPD (text teacher + frozen audio reference) retains it.

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import AudioStudent, sft_step, opd_step, freeze_reference, data  # noqa: E402
from src.model import text_teacher_targets                                # noqa: E402
from src.data import V, L, N_STYLES, FEAT_DIM, STYLE_GAIN, NOISE_STD, A, S, T  # noqa: E402

SEED = 0


def to_t(batch):
    return {k: torch.from_numpy(v) for k, v in batch.items()}


@torch.no_grad()
def coverage_acc(student, k: int = 24) -> float:
    """[A] Reliability across the WHOLE state space: for every (state, content)
    pair, does the policy produce the correct next state T[state, content]?

    Off-policy SFT only ever sees the gold states, so it never learns the
    transitions for states the student drifts into at inference (exposure bias).
    On-policy OPD samples broadly and covers them. This probes exactly that gap.
    """
    rng = np.random.default_rng(555)
    correct = total = 0
    for s in range(V):
        for c in range(V):
            styles = rng.integers(0, N_STYLES, size=k)
            feat = (A[c] + STYLE_GAIN * S[styles]
                    + NOISE_STD * rng.standard_normal((k, FEAT_DIM))).astype(np.float32)
            h = student.audio_enc(torch.from_numpy(feat))        # (k, d)
            state = torch.full((k,), s, dtype=torch.long)
            logits = student.sum_head(torch.cat([h, student.state_emb(state)], dim=-1))
            correct += int((logits.argmax(-1).numpy() == T[s, c]).sum()); total += k
    return correct / total


@torch.no_grad()
def evaluate(student, n=512):
    rng = np.random.default_rng(777)
    b = to_t(data.make_batch(n, rng))
    h = student.encode(b["feats"])
    # free-running greedy over the student's OWN state trajectory (secondary view)
    states_prev, emitted = student.rollout(h, sample=False)
    tgt = text_teacher_targets(b["content"], states_prev)
    freerun_acc = (emitted == tgt).float().mean().item()
    # [B] audio-only style retention
    style_pred = student.style_logits(h).argmax(-1)
    style_acc = (style_pred == b["style_token"]).float().mean().item()
    return {
        "coverage_acc": coverage_acc(student),
        "freerun_acc": freerun_acc,
        "style_acc": style_acc,
    }


def train(make_step, steps, bs, lr, rng, init=None, **kw):
    student = AudioStudent() if init is None else init
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(steps):
        batch = to_t(data.make_batch(bs, rng))
        make_step(student, batch, opt, **kw)
    student.eval()
    return student


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print("Qwen-Audio-3.1-Realtime  (arXiv:2609.25176) -- foundation M2-OPD, CPU\n")
    print(f"task: audio-only STYLE + a {V}-state machine over {L} spoken tokens\n")

    # ---- 1. Core-Cocktail SFT (off-policy): the initial audio policy ----
    def _sft(student, batch, opt):
        return sft_step(student, batch, opt)
    sft = train(_sft, steps=500, bs=64, lr=3e-3, rng=rng)
    sft_m = evaluate(sft)
    print("[1] Core-Cocktail SFT (off-policy, gold states)")
    print(f"    coverage {sft_m['coverage_acc']*100:5.1f}%   style {sft_m['style_acc']*100:5.1f}%\n")

    # freeze the Audio Reference from the SFT checkpoint
    reference = freeze_reference(sft)

    # ---- 2. Multimodality OPD (on-policy, Text Teacher only) ----
    import copy
    def _opd_text(student, batch, opt):
        return opd_step(student, reference, batch, opt, use_audio_reference=False, temp=2.6)
    opd_text = train(_opd_text, steps=1500, bs=64, lr=2e-3, rng=rng, init=copy.deepcopy(sft))
    opd_m = evaluate(opd_text)
    print("[2] Multimodality OPD (on-policy, Text Teacher only)")
    print(f"    coverage {opd_m['coverage_acc']*100:5.1f}%   style {opd_m['style_acc']*100:5.1f}%"
          f"   (style not text-supervised -> forgets it)\n")

    # ---- 3. M2-OPD (on-policy, Text Teacher + frozen Audio Reference) ----
    def _opd_m2(student, batch, opt):
        return opd_step(student, reference, batch, opt, use_audio_reference=True, temp=2.6)
    m2 = train(_opd_m2, steps=1500, bs=64, lr=2e-3, rng=rng, init=copy.deepcopy(sft))
    m2_m = evaluate(m2)
    print("[3] M2-OPD (on-policy, Text Teacher + frozen Audio Reference)")
    print(f"    coverage {m2_m['coverage_acc']*100:5.1f}%   style {m2_m['style_acc']*100:5.1f}%"
          f"   (audio reference retains the audio-only style)\n")

    payload = {
        "config": {"V": V, "L": L, "params": sum(p.numel() for p in sft.parameters())},
        "sft": sft_m, "opd_text": opd_m, "m2_opd": m2_m,
        "seconds": round(time.time() - t0, 1),
    }
    out = ROOT / "data"; out.mkdir(exist_ok=True)
    (out / "demo_results.json").write_text(json.dumps(payload, indent=2))

    dT = (opd_m["coverage_acc"] - sft_m["coverage_acc"]) * 100
    print(f"[A] on-policy reliability: SFT {sft_m['coverage_acc']*100:.1f}% -> "
          f"OPD {opd_m['coverage_acc']*100:.1f}% / M2 {m2_m['coverage_acc']*100:.1f}%  "
          f"(+{dT:.1f} pts coverage)")
    print(f"[B] audio-only style:      text-only OPD {opd_m['style_acc']*100:.1f}%  vs  "
          f"M2-OPD {m2_m['style_acc']*100:.1f}%")
    print(f"\nWrote {out/'demo_results.json'}.  Total: {payload['seconds']}s")

    ok = (
        opd_m["coverage_acc"] > sft_m["coverage_acc"] + 0.08
        and m2_m["coverage_acc"] > sft_m["coverage_acc"] + 0.08
        and m2_m["style_acc"] > opd_m["style_acc"] + 0.2
        and m2_m["style_acc"] > 0.85
    )
    if not ok:
        raise SystemExit(
            f"Sanity failed: coverage sft={sft_m['coverage_acc']:.2f} "
            f"opd={opd_m['coverage_acc']:.2f} m2={m2_m['coverage_acc']:.2f}; "
            f"style opd={opd_m['style_acc']:.2f} m2={m2_m['style_acc']:.2f}"
        )
    print("OK: on-policy M2-OPD is more reliable than off-policy SFT, and the "
          "frozen Audio Reference preserves the audio-only attribute the Text "
          "Teacher cannot supervise.")


if __name__ == "__main__":
    main()
