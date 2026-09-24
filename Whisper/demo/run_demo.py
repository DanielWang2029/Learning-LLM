"""End-to-end Whisper-style transcription demo (CPU, < 60 s).

Trains the encoder-decoder to transcribe synthetic "speech" (a sequence of tone
"words") into the correct token sequence, then reports greedy-decoding token
accuracy and shows a worked example. Demonstrates the full Whisper pipeline:
log-Mel -> conv stem (stride-2 downsample) -> Transformer encoder -> Transformer
decoder with cross-attention.

Run:  python demo/run_demo.py
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import MelFrontend, Whisper
from data.generate_audio import make_batch, NUM_TONES

SEED = 0
N_MELS = 80
# Multitask-style special tokens (a nod to Whisper's <|sot|>, <|transcribe|>...).
PAD, SOT, EOT = 0, 1, 2
NUM_SPECIAL = 3
VOCAB = NUM_TONES + NUM_SPECIAL


def encode_targets(id_lists, device):
    """content ids -> (decoder input [SOT, w...], target [w..., EOT]) padded."""
    seqs_in, seqs_out = [], []
    for ids in id_lists:
        toks = [t + NUM_SPECIAL for t in ids]
        seqs_in.append([SOT] + toks)
        seqs_out.append(toks + [EOT])
    L = max(len(s) for s in seqs_in)
    tin = torch.full((len(seqs_in), L), PAD, dtype=torch.long)
    tout = torch.full((len(seqs_in), L), PAD, dtype=torch.long)
    for i, (a, b) in enumerate(zip(seqs_in, seqs_out)):
        tin[i, : len(a)] = torch.tensor(a)
        tout[i, : len(b)] = torch.tensor(b)
    return tin.to(device), tout.to(device)


def featurize(front, wav_np):
    mel = front(torch.from_numpy(wav_np))     # (B, n_mels, T)
    return mel


@torch.no_grad()
def token_accuracy(model, front, rng, batch=64):
    model.eval()
    wav_np, id_lists = make_batch(batch, rng)
    mel = featurize(front, wav_np)
    decoded = model.transcribe(mel, SOT, EOT, max_len=16)
    correct_tok, total_tok, correct_seq = 0, 0, 0
    for pred, ids in zip(decoded, id_lists):
        pred = pred[1:]                        # drop SOT
        if EOT in pred:
            pred = pred[: pred.index(EOT)]
        gold = [t + NUM_SPECIAL for t in ids]
        for i in range(len(gold)):
            total_tok += 1
            if i < len(pred) and pred[i] == gold[i]:
                correct_tok += 1
        if pred == gold:
            correct_seq += 1
    model.train()
    return correct_tok / max(1, total_tok), correct_seq / batch


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    front = MelFrontend(n_mels=N_MELS)

    model = Whisper(n_mels=N_MELS, vocab=VOCAB, d_model=128, n_layers=2, n_heads=4, ff=256)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Whisper-style speech recognition on synthetic audio (CPU)")
    print(f"vocab = {VOCAB} tokens ({NUM_TONES} tone-words + SOT/EOT/PAD), model params = {n_params:,}\n")

    steps, batch_size = 320, 32
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        wav_np, id_lists = make_batch(batch_size, rng)
        mel = featurize(front, wav_np)
        tin, tout = encode_targets(id_lists, mel.device)
        logits = model(mel, tin)
        loss = crit(logits.reshape(-1, VOCAB), tout.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 40 == 0 or step == 1:
            tok_acc, seq_acc = token_accuracy(model, front, np.random.default_rng(999), batch=32)
            curve.append({"step": step, "token_acc": tok_acc, "seq_acc": seq_acc})
            print(f"step {step:4d}/{steps} | loss {loss.item():.3f} | "
                  f"token-acc {tok_acc*100:5.1f}% | seq-acc {seq_acc*100:5.1f}%")

    elapsed = time.time() - start
    tok_acc, seq_acc = token_accuracy(model, front, np.random.default_rng(2024), batch=128)
    print(f"\nTrained {steps} steps in {elapsed:.1f}s")
    print(f"Final token accuracy: {tok_acc*100:.1f}%   exact-sequence accuracy: {seq_acc*100:.1f}%")
    print("Note: Whisper reads a fixed 30 s chunk (non-streaming) and frames ASR as one")
    print("      seq2seq problem with multitask tokens — here a single SOT prompt token.")

    # ---- Worked example + data for the visualization.
    # Pick the first cleanly-decoded utterance so the example is easy to read.
    ex_rng = np.random.default_rng(77)
    mel, id_lists, pred, gold = None, None, None, None
    for _ in range(12):
        wav_np, ids_b = make_batch(1, ex_rng)
        m = featurize(front, wav_np)
        dec = model.transcribe(m, SOT, EOT, max_len=16)[0][1:]
        if EOT in dec:
            dec = dec[: dec.index(EOT)]
        g = [t + NUM_SPECIAL for t in ids_b[0]]
        mel, id_lists, pred, gold = m, ids_b, dec, g
        if dec == g:
            break
    print("\nExample")
    print(f"  audio words (gold ids): {id_lists[0]}")
    print(f"  decoded tokens        : {[p - NUM_SPECIAL for p in pred]}")

    mel_np = mel[0].numpy()
    out = {
        "vocab": VOCAB, "num_tones": NUM_TONES,
        "final_token_acc": tok_acc, "final_seq_acc": seq_acc,
        "accuracy_curve": curve,
        "log_mel": np.round(mel_np, 3).tolist(),
        "n_mels": int(mel_np.shape[0]), "n_frames": int(mel_np.shape[1]),
        "encoder_frames": int(mel_np.shape[1] // 2),
        "gold_ids": id_lists[0],
        "pred_ids": [p - NUM_SPECIAL for p in pred],
        "n_params": n_params,
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "demo_sample.json", "w") as f:
        json.dump(out, f)
    print(f"\nWrote {data_dir / 'demo_sample.json'} for the visualization.")

    if tok_acc < 0.85:
        raise SystemExit(f"Transcription did not converge (token acc {tok_acc*100:.1f}%).")
    print("OK: Whisper-style encoder-decoder learned to transcribe the audio.")


if __name__ == "__main__":
    main()
