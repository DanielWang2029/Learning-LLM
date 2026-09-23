"""End-to-end wav2vec 2.0 self-supervised pretraining demo (CPU, < 60 s).

Trains the full model — CNN feature encoder, product-quantized targets, masked
Transformer context network, and the contrastive InfoNCE objective — on
synthetic structured speech. We show:

  * contrastive accuracy on masked steps rising far above chance, and
  * the Gumbel-softmax codebook becoming diverse (perplexity ↑).

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

from src import Wav2Vec2, compute_mask_indices
from data.generate_audio import make_batch

SEED = 0
NUM_NEG = 20
MASK_PROB = 0.5
MASK_LEN = 4


def make_masks(T, B, rng):
    return [compute_mask_indices(T, MASK_PROB, MASK_LEN, rng) for _ in range(B)]


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    model = Wav2Vec2(dim=128, num_groups=2, num_vars=32)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_codewords = model.quantizer.num_groups * model.quantizer.num_vars
    chance = 1.0 / (NUM_NEG + 1)

    print("wav2vec 2.0 self-supervised pretraining on synthetic speech (CPU)")
    print(f"contrastive task: identify the true quantized latent among {NUM_NEG} distractors")
    print(f"chance accuracy = 1/{NUM_NEG + 1} = {chance*100:.1f}%   codebook = {num_codewords} codewords\n")

    steps, batch_size = 260, 8
    curve = []
    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        # anneal the Gumbel temperature 2.0 -> 0.5 like the paper (Section 3.1).
        model.quantizer.temp = max(0.5, 2.0 * (0.9995 ** step))
        wav_np, _ = make_batch(batch_size, rng)
        wav = torch.from_numpy(wav_np)
        with torch.no_grad():
            T = model.feature_encoder(wav).shape[1]
        masks = make_masks(T, batch_size, rng)

        loss, acc, ppl, used, _ = model(wav, masks, rng, num_negatives=NUM_NEG)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == 1:
            curve.append({"step": step, "acc": acc, "perplexity": ppl, "temp": model.quantizer.temp})
            print(f"step {step:4d}/{steps} | loss {loss.item():.3f} | "
                  f"masked-acc {acc*100:5.1f}% | codebook perplexity {ppl:5.1f}/{num_codewords} | temp {model.quantizer.temp:.2f}")

    elapsed = time.time() - start

    # ---- Final evaluation
    model.eval()
    eval_rng = np.random.default_rng(123)
    wav_np, phones = make_batch(16, eval_rng)
    wav = torch.from_numpy(wav_np)
    with torch.no_grad():
        T = model.feature_encoder(wav).shape[1]
        masks = make_masks(T, 16, eval_rng)
        loss, acc, ppl, used, mask_t = model(wav, masks, eval_rng, num_negatives=NUM_NEG)
    print(f"\nTrained {steps} steps in {elapsed:.1f}s")
    print(f"Final masked contrastive accuracy: {acc*100:.1f}%  (chance {chance*100:.1f}%)")
    print(f"Final codebook perplexity: {ppl:.1f} / {num_codewords} codewords in use")

    # ---- Capture data for the visualization
    # codeword histogram over a held-out batch
    with torch.no_grad():
        z = model.layer_norm(model.feature_encoder(wav))
        logits = model.quantizer.proj(z).view(-1, model.quantizer.num_groups, model.quantizer.num_vars)
        codes = logits.argmax(-1)  # (N, G)
    hist = []
    for g in range(model.quantizer.num_groups):
        counts = np.bincount(codes[:, g].numpy(), minlength=model.quantizer.num_vars)
        hist.append((counts / counts.sum()).round(4).tolist())

    # one example utterance: waveform (downsampled), latent frames, mask, phones
    ex_wav = wav_np[0]
    ex_wav_ds = ex_wav[::40][:400].round(4).tolist()  # thin for the browser
    ex_mask = mask_t[0].numpy().astype(int).tolist()
    ex_phones = phones[0].tolist()

    out = {
        "num_negatives": NUM_NEG,
        "chance": chance,
        "num_codewords": num_codewords,
        "num_groups": model.quantizer.num_groups,
        "num_vars": model.quantizer.num_vars,
        "final_acc": acc,
        "final_perplexity": ppl,
        "accuracy_curve": curve,
        "codebook_hist": hist,
        "example_wave": ex_wav_ds,
        "example_mask": ex_mask,
        "example_phones": ex_phones,
        "n_latent_frames": int(T),
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "demo_sample.json", "w") as f:
        json.dump(out, f)
    print(f"Wrote {data_dir / 'demo_sample.json'} for the visualization.")

    if acc < 0.6:
        raise SystemExit(f"Contrastive task did not learn (acc {acc*100:.1f}% too low).")
    print("OK: masked contrastive accuracy is well above chance and the codebook is used.")


if __name__ == "__main__":
    main()
