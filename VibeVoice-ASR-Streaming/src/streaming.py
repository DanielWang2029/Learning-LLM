"""Streaming inference driver + speaker-attributed accuracy metrics.

Runs the trained model exactly as VibeVoice-ASR-Streaming does at inference
(§3.1): chunk by chunk, each chunk's speaker-attributed text generated
autoregressively against the accumulated speech-text history, stopping at the
<chunk_end> token before the next speech chunk is appended.
"""

from __future__ import annotations

import numpy as np
import torch

from .audio import FRAME_MS, FRAMES_PER_PHONE
from .data_gen import (
    CHUNK_END,
    PH_BASE,
    SPK_BASE,
    chunk_audio_window,
    token_str,
)


@torch.no_grad()
def stream_decode(model, feats, phones, ords, chunk_phones, lookahead,
                  with_lookahead=True, max_text=8):
    """Decode a conversation chunk-by-chunk; return emissions + running trace."""
    model.eval()
    n_frames = feats.shape[0]
    n_chunks = len(phones) // chunk_phones
    n_mels = feats.shape[1]

    audio_rows, text_rows, isaud_rows = [], [], []
    emissions = []          # per chunk: list of predicted token ids (no chunk_end)
    trace = []
    cur_ord = -1
    pred_phones = []        # list of (ord, phone)

    for k in range(n_chunks):
        start, audio_end, look_end = chunk_audio_window(k, chunk_phones, lookahead, n_frames)
        idxs = list(range(start, audio_end))
        if with_lookahead:
            idxs += list(range(audio_end, look_end))
        for fi in idxs:
            audio_rows.append(feats[fi]); text_rows.append(0); isaud_rows.append(True)

        chunk_emit = []
        for _ in range(max_text):
            a = torch.tensor(np.stack(audio_rows)[None], dtype=torch.float32)
            t = torch.tensor([text_rows], dtype=torch.long)
            m = torch.tensor([isaud_rows], dtype=torch.bool)
            logits = model(a, t, m)[0, -1]
            tok = int(logits.argmax().item())
            audio_rows.append(np.zeros(n_mels, dtype=np.float32))
            text_rows.append(tok); isaud_rows.append(False)
            if tok == CHUNK_END:
                break
            chunk_emit.append(tok)

        emissions.append(chunk_emit)
        # Decode emitted tokens into (speaker, phone) pairs, carrying speaker.
        chunk_pairs = []
        for tok in chunk_emit:
            if SPK_BASE <= tok < PH_BASE:
                cur_ord = tok - SPK_BASE
            elif tok >= PH_BASE:
                chunk_pairs.append((cur_ord, tok - PH_BASE))
                pred_phones.append((cur_ord, tok - PH_BASE))

        audio_ms0 = start * FRAME_MS
        audio_ms1 = look_end * FRAME_MS
        trace.append({
            "chunk": k,
            "audio_ms": [audio_ms0, audio_ms1],
            "emitted": [token_str(t) for t in chunk_emit],
            "text": _pretty(chunk_emit),
        })
    return emissions, pred_phones, trace


def _pretty(tokens):
    out = []
    for t in tokens:
        s = token_str(t)
        out.append(("\n" + s + ": ") if s.startswith("Speaker") else s + " ")
    return "".join(out).strip()


def true_pairs(phones, ords):
    return list(zip(ords, phones))


def score(pred_phones, phones, ords):
    """Position-aligned transcription and speaker-attribution accuracy."""
    truth = true_pairs(phones, ords)
    n = len(truth)
    n_cmp = min(len(pred_phones), n)
    phone_ok = sum(1 for i in range(n_cmp) if pred_phones[i][1] == truth[i][1])
    spk_ok = sum(1 for i in range(n_cmp) if pred_phones[i][0] == truth[i][0])
    # missing / extra predictions count as errors against the reference length.
    return {
        "transcription_acc": phone_ok / n if n else 1.0,
        "speaker_acc": spk_ok / n if n else 1.0,
        "n_ref": n,
        "n_pred": len(pred_phones),
    }
