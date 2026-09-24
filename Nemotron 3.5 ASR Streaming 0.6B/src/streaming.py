"""Cache-aware streaming: full-context equivalence, latency-budget decode, WER."""

from __future__ import annotations

import torch

from .audio import SUBSAMPLE


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def wer(hyp, ref):
    return edit_distance(hyp, ref) / max(1, len(ref))


@torch.no_grad()
def streaming_encode(encoder, mel, R, chunk_enc=2):
    """Emulate cache-aware streaming: finalize each encoder frame from the first
    prefix in which it is stable (its full right context has arrived).

    Returns (assembled, full, max_abs_diff). Attention right-context is R *per
    layer* and convs are causal, so a frame's receptive field extends
    `n_layers * R` encoder frames to the right. Once that many future frames
    exist in the streamed prefix the frame is final, so `assembled` matches a
    single full-context pass to numerical precision — the "no redundant
    overlapping computation" property of cache-aware streaming.
    """
    encoder.eval()
    n_layers = len(encoder.blocks)
    full = encoder(mel, R)                       # (1, T, d)
    T = full.size(1)
    assembled = torch.zeros_like(full)
    done = torch.zeros(T, dtype=torch.bool)
    mel_len = mel.size(1)

    enc_step_mel = chunk_enc * SUBSAMPLE
    cur = enc_step_mel
    while not done.all():
        cur = min(cur, mel_len)
        out = encoder(mel[:, :cur, :], R)        # encode the prefix
        t_k = out.size(1)
        # A frame t is final once its right context (t + n_layers*R) is present.
        stable = min(T, t_k - n_layers * R) if cur < mel_len else T
        for t in range(stable):
            if not done[t]:
                assembled[:, t] = out[:, t]
                done[t] = True
        if cur >= mel_len:
            break
        cur += enc_step_mel
    max_diff = (assembled - full).abs().max().item()
    return assembled, full, max_diff


@torch.no_grad()
def latency_decode(rnnt, cond, enc_frames_per_phone, R, n_phones, blank=0):
    """Streaming RNN-T decode under an emission-latency budget of R encoder frames.

    A real cache-aware model must emit the token for a phone within its latency
    budget: it may consume encoder frames only up to `phone_start + R` before
    committing (append-only, no revisions). Because each phone spans several
    frames and features are noisy, a small budget forces a decision before the
    phone is integrated (errors), while a larger budget lets the decoder wait for
    a cleaner, better-integrated frame (higher accuracy, higher latency).
    """
    T = cond.size(1)
    labels = [blank]
    emitted = []
    for i in range(n_phones):
        f = min(i * enc_frames_per_phone + R, (i + 1) * enc_frames_per_phone - 1, T - 1)
        pred = rnnt.predict(torch.tensor([labels], device=cond.device))[:, -1:, :]
        logits = rnnt.joint(cond[:, f:f + 1], pred)[0, 0, 0].clone()
        logits[blank] = float("-inf")
        tok = int(logits.argmax().item())
        emitted.append(tok)
        labels.append(tok)
    return emitted
