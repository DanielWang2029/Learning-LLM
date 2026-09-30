"""End-to-end Mimi codec demo (CPU, < 60 s).

Trains a *tiny* Mimi-style codec on synthetic "speech" and demonstrates the
properties that define Mimi in the Moshi tech report (§3.3):

  (a) LOW FRAME RATE   — a 24 kHz-style waveform is compressed to 12.5 Hz tokens;
                          we report the compression ratio and effective bitrate.
  (b) SCALABLE BITRATE — the split residual-VQ reconstructs the encoder latent
                          monotonically better as more codebooks are decoded
                          (the residual-VQ property that lets one model serve
                          many bitrates).
  (c) STREAMING = OFFLINE — running the causal encoder/decoder chunk-by-chunk
                          (carrying conv state) matches the full-sequence pass to
                          numerical precision, i.e. the codec is truly streaming.
  (d) SEMANTIC DISTILLATION — codebook #1 is trained with a cosine distillation
                          loss toward a *semantic* target (the paper distills the
                          WavLM self-supervised representation). We show the
                          alignment of codebook #1 to that target rises far above
                          an untrained baseline — the split-RVQ mechanism that
                          fuses semantic + acoustic tokens in one codec.

Honesty notes (see README): this is a tiny CPU reproduction of Mimi's *mechanism*,
not the trained 24 kHz codec. WavLM has no CPU stand-in, so codebook #1 is
distilled toward a transparent proxy target (a fixed embedding of the ground-truth
"vowel" class); the distillation loss (cosine into codebook #1) is identical to
the paper's. Faithful reconstruction of a full-band waveform needs the paper's
model scale, so property (b) is reported as the residual-VQ *latent*
reconstruction (the quantizer's actual job), which is what the scalable-bitrate
claim is about.

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)  # keep CPU timing predictable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import MimiCodec, multi_scale_spectral_loss, stft_mag
from data.generate_audio import (
    make_batch,
    SAMPLE_RATE,
    TOTAL_STRIDE,
    FRAME_RATE,
    NUM_CLASSES,
    CLIP_SAMPLES,
)

SEED = 0
CODEBOOK_SIZE = 256
NUM_ACOUSTIC = 7
TARGET_DIM = 16
STEPS = 150
BATCH = 8


def make_target_table(seed: int = 123) -> torch.Tensor:
    """Fixed random unit embedding per class — the WavLM proxy (see README)."""
    rng = np.random.default_rng(seed)
    tbl = rng.standard_normal((NUM_CLASSES, TARGET_DIM)).astype(np.float32)
    tbl /= np.linalg.norm(tbl, axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(tbl)


def frame_targets(labels: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table[labels].permute(0, 2, 1).contiguous()  # (B, TARGET_DIM, T)


def build_model() -> MimiCodec:
    torch.manual_seed(SEED)
    return MimiCodec(
        num_acoustic=NUM_ACOUSTIC,
        codebook_size=CODEBOOK_SIZE,
        semantic_target_dim=TARGET_DIM,
    )


def train(model: MimiCodec, table: torch.Tensor, steps: int = STEPS) -> None:
    rng = np.random.default_rng(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        w, y = make_batch(BATCH, rng)
        wav = torch.from_numpy(w).unsqueeze(1)
        labels = torch.from_numpy(y)
        # Quantizer dropout: train every bitrate so decoding is scalable (§3.3.1).
        n_ac = int(torch.randint(0, NUM_ACOUSTIC + 1, (1,)).item())
        out = model(wav, n_acoustic=n_ac)
        loss = F.l1_loss(out["recon"], wav) + multi_scale_spectral_loss(out["recon"], wav)
        loss = loss + 0.1 * out["commit"]
        tgt = frame_targets(labels, table)
        loss = loss + 5.0 * MimiCodec.distillation_loss(out["semantic_pred"], tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def snr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    noise = ((x - y) ** 2).mean().item()
    signal = (x ** 2).mean().item()
    return 10.0 * math.log10(signal / (noise + 1e-12))


@torch.no_grad()
def latent_rvq_curve(model: MimiCodec, wav: torch.Tensor):
    """SCALABLE BITRATE: the acoustic residual-VQ reconstructs the encoder latent
    monotonically better as more residual codebooks are used — the property that
    lets one codec serve many bitrates. We measure the acoustic RVQ's own
    reconstruction of the latent ``z`` (codebook #1 is the *parallel* distilled
    semantic codebook, reported separately in step (d)).
    """
    q = model.quantizer
    z = q.project_in(model.encoder(wav))          # projected latent the RVQ sees
    pts = []
    for k in range(1, NUM_ACOUSTIC + 1):
        q_ac, _, _ = q.acoustic(z, n_active=k)
        pts.append({
            "codebooks": k,
            "snr_db": snr_db(z, q_ac),
            "bitrate_bps": k * math.log2(model.codebook_size) * FRAME_RATE,
        })
    return pts


@torch.no_grad()
def streaming_maxdiff(model: MimiCodec, wav: torch.Tensor):
    """Chunked causal encode+decode vs. the full-sequence pass (max abs diff)."""
    latent_off = model.encoder(wav)
    model.encoder.reset_stream(wav.shape[0], wav.device)
    enc, ch = [], TOTAL_STRIDE * 2
    for i in range(0, wav.shape[-1], ch):
        enc.append(model.encoder.step(wav[..., i:i + ch]))
    enc_diff = (latent_off - torch.cat(enc, dim=-1)).abs().max().item()

    dec_off = model.decoder(latent_off)
    model.decoder.reset_stream(latent_off.shape[0], latent_off.device)
    dec = []
    for i in range(0, latent_off.shape[-1], 2):
        dec.append(model.decoder.step(latent_off[..., i:i + 2]))
    dec_diff = (dec_off - torch.cat(dec, dim=-1)).abs().max().item()
    return enc_diff, dec_diff


@torch.no_grad()
def distill_alignment(model: MimiCodec, table: torch.Tensor, wav, labels) -> float:
    """Mean cosine similarity of codebook #1's semantic projection to the target."""
    out = model(wav)
    pred = F.normalize(out["semantic_pred"], dim=1)
    tgt = F.normalize(frame_targets(labels, table), dim=1)
    return float((pred * tgt).sum(1).mean().item())


def main() -> None:
    t0 = time.time()
    table = make_target_table()

    print("Tiny Mimi neural audio codec  (Moshi tech report §3.3, CPU reproduction)")
    print(f"synthetic speech: {SAMPLE_RATE} Hz, {NUM_CLASSES} vowel classes\n")

    model = build_model()
    untrained = build_model()  # baseline for the distillation-alignment contrast
    train(model, table)
    print(f"trained tiny codec ({STEPS} steps) in {time.time()-t0:.1f}s")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    rng = np.random.default_rng(7)
    w_eval, y_eval = make_batch(24, rng)
    wav_eval = torch.from_numpy(w_eval).unsqueeze(1)
    labels_eval = torch.from_numpy(y_eval)

    # ---------- (a) low frame rate / bitrate ----------
    total_codebooks = 1 + NUM_ACOUSTIC
    bitrate = total_codebooks * math.log2(CODEBOOK_SIZE) * FRAME_RATE
    print("(a) LOW FRAME RATE  (paper: 24 kHz -> 12.5 Hz, 1.1 kbps)")
    print(f"    {SAMPLE_RATE} Hz waveform -> {FRAME_RATE:.1f} Hz tokens "
          f"(x{TOTAL_STRIDE} downsampling; {CLIP_SAMPLES} samples -> {CLIP_SAMPLES//TOTAL_STRIDE} tokens)")
    print(f"    {total_codebooks} codebooks x {int(math.log2(CODEBOOK_SIZE))} bits x "
          f"{FRAME_RATE:.1f} Hz = {bitrate:.0f} bps ({bitrate/1000:.2f} kbps)\n")

    # ---------- (b) scalable bitrate (RVQ latent reconstruction) ----------
    curve = latent_rvq_curve(model, wav_eval)
    print("(b) SCALABLE BITRATE  (split-RVQ latent reconstruction vs #codebooks)")
    for p in curve:
        bar = "#" * int(max(0, p["snr_db"]))
        print(f"    {p['codebooks']} codebooks ({p['bitrate_bps']:5.0f} bps)  "
              f"SNR {p['snr_db']:5.2f} dB  {bar}")
    snrs = [p["snr_db"] for p in curve]
    monotone = all(snrs[i] <= snrs[i + 1] + 0.25 for i in range(len(snrs) - 1))
    print(f"    -> monotonically improving: {monotone}\n")

    # ---------- (c) streaming == offline ----------
    enc_diff, dec_diff = streaming_maxdiff(model, wav_eval)
    print("(c) STREAMING == OFFLINE  (causal conv state, chunk-by-chunk)")
    print(f"    encoder max|stream-offline| = {enc_diff:.2e}")
    print(f"    decoder max|stream-offline| = {dec_diff:.2e}\n")

    # ---------- (d) semantic distillation mechanism ----------
    cos_trained = distill_alignment(model, table, wav_eval, labels_eval)
    cos_untrained = distill_alignment(untrained, table, wav_eval, labels_eval)
    print("(d) SEMANTIC DISTILLATION  (codebook #1 <- cosine distill to semantic target)")
    print(f"    codebook #1 alignment to semantic target: untrained {cos_untrained:+.3f} "
          f"-> distilled {cos_trained:+.3f}")
    print(f"    -> distillation routes semantic info into codebook #1: "
          f"{cos_trained:.2f} >> {cos_untrained:.2f}")
    print("    (full phonetic separability needs the paper's WavLM teacher + scale; "
          "see README)\n")

    # ---------- capture a sample for the visualization ----------
    rng = np.random.default_rng(11)
    w1, y1 = make_batch(1, rng)
    wav1 = torch.from_numpy(w1).unsqueeze(1)
    with torch.no_grad():
        out1 = model(wav1)
    inp1 = wav1[0, 0].cpu().numpy()
    recon1 = out1["recon"][0, 0].cpu().numpy()
    spec_in = np.log(stft_mag(wav1.squeeze(1), 256, 64)[0].cpu().numpy() + 1e-4)
    spec_re = np.log(stft_mag(out1["recon"].squeeze(1), 256, 64)[0].cpu().numpy() + 1e-4)

    def snippet(x, n=1200):
        return np.round(x[:n], 4).tolist()

    payload = {
        "sample_rate": SAMPLE_RATE,
        "token_rate": FRAME_RATE,
        "total_stride": TOTAL_STRIDE,
        "num_classes": NUM_CLASSES,
        "codebook_size": CODEBOOK_SIZE,
        "num_acoustic": NUM_ACOUSTIC,
        "total_codebooks": total_codebooks,
        "bitrate_bps": bitrate,
        "clip_samples": CLIP_SAMPLES,
        "clip_tokens": CLIP_SAMPLES // TOTAL_STRIDE,
        "bitrate_curve": [
            {"codebooks": p["codebooks"], "snr_db": round(p["snr_db"], 3),
             "bitrate_bps": round(p["bitrate_bps"], 1)} for p in curve
        ],
        "streaming": {"encoder_maxdiff": enc_diff, "decoder_maxdiff": dec_diff},
        "distill": {"trained": cos_trained, "untrained": cos_untrained},
        "sample": {
            "input_wave": snippet(inp1),
            "recon_wave": snippet(recon1),
            "spec_in": np.round(spec_in, 3).tolist(),
            "spec_recon": np.round(spec_re, 3).tolist(),
            "frame_labels": y1[0].tolist(),
            "semantic_codes": out1["semantic_codes"][0].tolist(),
            "acoustic_codes_l1": out1["acoustic_codes"][0][0].tolist(),
        },
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "demo_sample.json", "w") as f:
        json.dump(payload, f)
    print(f"Wrote {data_dir / 'demo_sample.json'} for the visualization.")
    print(f"Total demo time: {time.time()-t0:.1f}s")

    # ---------- sanity checks ----------
    ok = (
        monotone
        and snrs[-1] > snrs[0] + 1.0
        and enc_diff < 1e-3 and dec_diff < 1e-3
        and cos_trained > 0.35
        and cos_trained > cos_untrained + 0.2
    )
    if not ok:
        raise SystemExit(
            f"Sanity check failed: monotone={monotone} snr {snrs[0]:.2f}->{snrs[-1]:.2f} "
            f"enc={enc_diff:.1e} dec={dec_diff:.1e} cos {cos_untrained:.2f}->{cos_trained:.2f}"
        )
    print("OK: 12.5 Hz frame rate + scalable RVQ + streaming==offline + semantic "
          "distillation all hold.")


if __name__ == "__main__":
    main()
