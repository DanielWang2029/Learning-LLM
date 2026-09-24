"""Generate two-speaker conversations and the interleaved speech-text sequence.

The paper (§3.1) organizes streaming as an interleaved sequence
    [X1, Y1, X2, Y2, ...]
where Xk is the k-th speech chunk (plus a fixed L-frame lookahead) and Yk is its
speaker-attributed text, ending in a chunk-end token. Speakers are labelled by
*order of first appearance*, and a label introduced early is reused later — so
the target text depends on history carried across chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import FRAMES_PER_PHONE, NUM_PHONES, NUM_SPEAKERS, PHONE_NAMES, features, synth_conversation

# ---- Text vocabulary ------------------------------------------------------
PAD = 0
CHUNK_END = 1
SPK_BASE = 2                       # SPK0 = 2, SPK1 = 3
PH_BASE = SPK_BASE + NUM_SPEAKERS  # PH0 = 4, ...
VOCAB = PH_BASE + NUM_PHONES

SPK_TOK = [SPK_BASE + s for s in range(NUM_SPEAKERS)]
PH_TOK = [PH_BASE + p for p in range(NUM_PHONES)]


def token_str(tok):
    if tok == PAD:
        return "<pad>"
    if tok == CHUNK_END:
        return "<chunk_end>"
    if SPK_BASE <= tok < PH_BASE:
        return f"Speaker{tok - SPK_BASE}"
    return PHONE_NAMES[tok - PH_BASE]


def gen_conversation(rng, n_chunks=6, chunk_phones=3, p_switch=0.35):
    """Return (phone_ids, raw_speaker_ids, ordinal_speaker_ids)."""
    n = n_chunks * chunk_phones
    phones = [int(x) for x in rng.integers(0, NUM_PHONES, size=n)]
    raw = [int(rng.integers(0, NUM_SPEAKERS))]
    for _ in range(n - 1):
        nxt = raw[-1]
        if rng.random() < p_switch:
            nxt = 1 - nxt
        raw.append(nxt)
    # Ordinal labels by order of first appearance.
    order, mapping = [], {}
    for s in raw:
        if s not in mapping:
            mapping[s] = len(mapping)
        order.append(mapping[s])
    return phones, raw, order


def chunk_text(chunk_ords, chunk_phones_ids, last_ord):
    """Build Yk: emit a speaker label only when the speaker changes."""
    toks = []
    for ord_, pid in zip(chunk_ords, chunk_phones_ids):
        if ord_ != last_ord:
            toks.append(SPK_TOK[ord_])
            last_ord = ord_
        toks.append(PH_TOK[pid])
    toks.append(CHUNK_END)
    return toks, last_ord


def chunk_audio_window(k, chunk_phones, lookahead, n_frames_total):
    """Frames for chunk k: its audio up to the last phone's ONSET, then the
    lookahead frames that include the last phone's PEAK (future evidence)."""
    last_phone = (k + 1) * chunk_phones - 1
    start = 2 * (k * chunk_phones)
    audio_end = 2 * last_phone + 1            # includes onset of last phone
    look_end = min(n_frames_total, audio_end + lookahead)
    return start, audio_end, look_end


@dataclass
class Plan:
    """An interleaved plan: ordered entries of audio frames and text tokens."""

    entries: list = field(default_factory=list)   # (is_audio: bool, payload)
    # payload = feature vector (audio) or token id (text)


def build_plan(feats, phones, ords, chunk_phones, lookahead, with_lookahead=True):
    """Construct the full teacher-forced interleaved plan for one conversation."""
    n_frames = feats.shape[0]
    n_chunks = len(phones) // chunk_phones
    plan = Plan()
    last_ord = -1
    chunk_meta = []
    for k in range(n_chunks):
        start, audio_end, look_end = chunk_audio_window(k, chunk_phones, lookahead, n_frames)
        idxs = list(range(start, audio_end))
        if with_lookahead:
            idxs += list(range(audio_end, look_end))
        for fi in idxs:
            plan.entries.append((True, feats[fi]))
        lo = k * chunk_phones
        chunk_ords = ords[lo:lo + chunk_phones]
        chunk_pids = phones[lo:lo + chunk_phones]
        toks, last_ord = chunk_text(chunk_ords, chunk_pids, last_ord)
        for t in toks:
            plan.entries.append((False, t))
        chunk_meta.append({"chunk": k, "audio_frames": idxs, "text": toks,
                           "phones": chunk_pids, "ords": chunk_ords})
    return plan, chunk_meta


def plan_to_tensors(plan, n_mels):
    """Convert a plan to (audio_feats, text_ids, is_audio, target, target_mask)."""
    S = len(plan.entries)
    audio = np.zeros((S, n_mels), dtype=np.float32)
    text = np.zeros(S, dtype=np.int64)
    is_audio = np.zeros(S, dtype=bool)
    for i, (a, payload) in enumerate(plan.entries):
        if a:
            audio[i] = payload
            is_audio[i] = True
        else:
            text[i] = payload
    target = np.full(S, PAD, dtype=np.int64)
    tmask = np.zeros(S, dtype=bool)
    for i in range(S - 1):
        na, npay = plan.entries[i + 1]
        if not na:                      # next entry is a text token we predict
            target[i] = npay
            tmask[i] = True
    return audio, text, is_audio, target, tmask


def make_dataset(n, rng, n_chunks, chunk_phones, lookahead, n_mels):
    """Create `n` conversations as tensor tuples for training."""
    samples = []
    for _ in range(n):
        phones, raw, ords = gen_conversation(rng, n_chunks, chunk_phones)
        wav, nf = synth_conversation(phones, raw, rng)
        feats = features(wav, nf)
        plan, _ = build_plan(feats, phones, ords, chunk_phones, lookahead)
        samples.append(plan_to_tensors(plan, n_mels))
    return samples
