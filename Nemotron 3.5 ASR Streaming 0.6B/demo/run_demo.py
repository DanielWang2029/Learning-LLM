"""End-to-end CPU demo of Nemotron 3.5 ASR (cache-aware FastConformer-RNNT).

Reproduces the model card's three headline properties from ONE trained model:
  (a) cache-aware streaming output matches a full-context pass (within tolerance),
  (b) a chunk-size sweep gives a latency/accuracy Pareto with NO retraining,
  (c) a language-ID prompt token switches the transcription on identical audio.

Run with:  python demo/run_demo.py
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
    ENC_FRAME_MS,
    ENC_FRAMES_PER_PHONE,
    LANG_NAMES,
    N_MELS,
    NUM_LANGS,
    VOCAB,
    FastConformerEncoder,
    LangFusion,
    RNNT,
    greedy_decode,
    latency_decode,
    make_dataset,
    streaming_encode,
    token_str,
    transducer_loss,
    wer,
)
from src.data_gen import BLANK, gen_utterance, token_id  # noqa: E402

DATA_DIR = PAPER_DIR / "data"

N_PHONES = 3                       # phones per utterance
NOISE = 2.0                        # heavy feature noise -> right context matters
R_SET = [0, 1, 3, 6, 13]          # right context (80ms frames) -> chunk sizes
SEED = 0
STEPS = 1800
BS = 32
S = ENC_FRAMES_PER_PHONE          # encoder frames per phone


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FastConformerEncoder(N_MELS, d_model=64, n_layers=2, n_heads=4, d_ff=128)
        self.fusion = LangFusion(64, NUM_LANGS, k=16)
        self.rnnt = RNNT(VOCAB, d_model=64, pred_dim=64, joint_dim=64)

    def enc_cond(self, mel, lang, R):
        return self.fusion(self.encoder(mel, R), lang)

    def loss(self, mel, lang, targets, R):
        cond = self.enc_cond(mel, lang, R)
        B, U = targets.shape
        start = torch.full((B, 1), BLANK, dtype=torch.long)
        labels = torch.cat([start, targets], dim=1)
        pred = self.rnnt.predict(labels)
        logp = torch.log_softmax(self.rnnt.joint(cond, pred), dim=-1)
        return transducer_loss(logp, targets, blank=BLANK)


def collate(batch):
    mel = torch.tensor(np.stack([b[0] for b in batch]))
    lang = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targets = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return mel, lang, targets


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("=" * 78)
    print("Nemotron 3.5 ASR  -  cache-aware FastConformer-RNNT streaming (small)")
    print("=" * 78)
    print(f"config: {N_PHONES} phones/utt, 8x subsampling, enc frame = {ENC_FRAME_MS} ms, "
          f"{S} enc frames/phone, vocab={VOCAB}, langs={NUM_LANGS}")

    train = make_dataset(320, rng, N_PHONES, noise=NOISE)
    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    print(f"model parameters: {n_params:,}\n")

    print("training with RANDOM right-context per step (one model for every chunk size)")
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(train), size=BS)
        mel, lang, targets = collate([train[i] for i in idx])
        R = int(rng.choice(R_SET))
        loss = model.loss(mel, lang, targets, R)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0 or step == 1:
            print(f"  step {step:4d}/{STEPS} | loss {loss.item():.4f} | R={R}")

    model.eval()

    # ---- (a) cache-aware streaming == full context
    print("\n(a) cache-aware streaming vs full-context encoder output")
    print("-" * 78)
    ex_rng = np.random.default_rng(321)
    feats, lang, phones, target = gen_utterance(ex_rng, N_PHONES, lang=0, noise=NOISE)
    mel1 = torch.tensor(feats[None])
    diffs = {}
    for R in [0, 3, 13]:
        _, _, md = streaming_encode(model.encoder, mel1, R, chunk_enc=2)
        diffs[R] = md
        print(f"  R={R:>2}  max |streaming - full| = {md:.2e}   (identical up to fp precision)")

    # ---- (b) chunk-size sweep (one model, no retraining)
    print("\n(b) chunk-size sweep: latency vs accuracy (one model, no retraining)")
    print("-" * 78)
    test = make_dataset(80, np.random.default_rng(777), N_PHONES, noise=NOISE)
    # offline (full-context) accuracy: how good the model is with unbounded latency
    off_wer = np.mean([wer(greedy_decode(model.rnnt,
                                         model.enc_cond(torch.tensor(x[0][None]),
                                                        torch.tensor([x[1]]), 13),
                                         blank=BLANK, max_labels=N_PHONES), x[3]) for x in test])
    print(f"  offline full-context accuracy (unbounded latency): {(1 - off_wer) * 100:.1f}%\n")
    print(f"{'R (80ms frames)':>15} | {'chunk (ms)':>10} | {'latency (ms)':>12} | {'WER':>7} | {'acc':>7}")
    sweep = []
    for R in R_SET:
        werr = 0.0
        for feats_t, lang_t, phones_t, target_t in test:
            cond = model.enc_cond(torch.tensor(feats_t[None]), torch.tensor([lang_t]), R)
            hyp = latency_decode(model.rnnt, cond, S, R, N_PHONES, blank=BLANK)
            werr += wer(hyp, target_t)
        w = werr / len(test)
        latency = (R + 1) * ENC_FRAME_MS
        sweep.append({"R": R, "chunk_ms": (R + 1) * ENC_FRAME_MS, "latency_ms": latency,
                      "wer": round(w, 4), "acc": round(1 - w, 4)})
        print(f"{R:>15} | {(R+1)*ENC_FRAME_MS:>10} | {latency:>12} | {w*100:>6.1f}% | {(1-w)*100:>6.1f}%")

    # ---- (c) language-ID prompt switches transcription on identical audio
    print("\n(c) language-ID prompt on IDENTICAL audio")
    print("-" * 78)
    lp_rng = np.random.default_rng(55)
    feats_c, _, phones_c, _ = gen_utterance(lp_rng, N_PHONES, lang=0, noise=NOISE)
    mel_c = torch.tensor(feats_c[None])
    decodes = {}
    for L in range(NUM_LANGS):
        cond = model.fusion(model.encoder(mel_c, 13), torch.tensor([L]))
        hyp = greedy_decode(model.rnnt, cond, blank=BLANK, max_labels=N_PHONES)
        ref = [token_id(L, p) for p in phones_c]
        acc = 1 - wer(hyp, ref)
        decodes[L] = {"tokens": [token_str(t) for t in hyp], "acc": round(acc, 3)}
        print(f"  prompt = {LANG_NAMES[L]:>7}  ->  {' '.join(token_str(t) for t in hyp)}"
              f"    (acc {acc*100:.0f}%)")
    print(f"  same audio, phones = {phones_c}  ->  prompt selects the token set (A* vs B*)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s on CPU.")
    assert max(diffs.values()) < 1e-4, "streaming must match full context"
    assert sweep[-1]["acc"] - sweep[0]["acc"] > 0.3, "larger chunk should improve accuracy"
    assert sweep[-1]["acc"] > 0.85, "full-context streaming should be accurate"
    assert decodes[0]["acc"] > 0.8 and decodes[1]["acc"] > 0.8, "prompt conditioning should work"
    print("OK: streaming==full; bigger chunks trade latency for accuracy; prompt switches language.")

    # ---- viz data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"n_phones": N_PHONES, "enc_frame_ms": ENC_FRAME_MS, "subsample": 8,
                   "enc_frames_per_phone": S, "vocab": VOCAB, "n_langs": NUM_LANGS,
                   "noise": NOISE, "offline_acc": round(1 - float(off_wer), 4)},
        "exact_match": {str(R): d for R, d in diffs.items()},
        "sweep": sweep,
        "lang_prompt": {
            "phones": phones_c,
            "decodes": {LANG_NAMES[L]: decodes[L] for L in range(NUM_LANGS)},
        },
        "mel": np.round(feats_c, 2).tolist(),
    }
    (DATA_DIR / "streaming_run.json").write_text(json.dumps(out))
    print("Wrote data/streaming_run.json  ->  open visualization/index.html")


if __name__ == "__main__":
    main()
