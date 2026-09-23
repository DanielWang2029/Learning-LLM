"""Synthetic audio + hand-rolled log-mel features (numpy only).

No real audio is downloaded. Waveforms are synthesized from speech-like
primitives — sine tones, chirps, formant stacks, noise — and converted to
log-mel spectrograms with a from-scratch STFT (numpy + stdlib, CPU only).

For U2++ we build an audio -> token task designed to make the second pass
(attention rescoring) demonstrably useful:

  * Four content tokens. C and D are acoustically **distinct** anchors (clean
    low / high tones). A and B share a strong common carrier and differ only by
    a **weak, noisy** side-tone, so a frame-synchronous CTC decoder confuses
    them fairly often.
  * The label sequence has a deterministic grammar: each pair is [anchor,
    ambiguous] with the rule  C -> A  and  D -> B.

CTC is conditionally independent across output positions, so it cannot exploit
that grammar and errs on the noisy A/B tokens. The autoregressive attention
decoders learn the grammar and, during rescoring, correct those errors — the
classic reason attention rescoring improves CTC hypotheses (U2++, §3.3). Because
the A/B cue is *local* to each token, offline and streaming CTC perform
comparably, so the same model serves both regimes.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

# Content tokens (0-indexed): A, B (confusable pair), C, D (distinct anchors).
TOKENS = ["A", "B", "C", "D"]
A, B, C, D = 0, 1, 2, 3
NUM_TOKENS = len(TOKENS)


def _tone(freq, dur, rng, noise, amp=1.0, side=None, side_amp=0.0, sr=SAMPLE_RATE):
    t = np.arange(int(dur * sr)) / sr
    wave = amp * np.sin(2 * np.pi * freq * t)
    if side is not None:
        wave = wave + side_amp * np.sin(2 * np.pi * side * t)
    wave = wave * rng.uniform(0.85, 1.0) + noise * rng.standard_normal(len(t))
    return wave.astype(np.float32)


def _render_token(tok, dur, rng):
    if tok == C:                      # distinct, clean low tone
        return _tone(rng.uniform(280, 340), dur, rng, noise=0.05)
    if tok == D:                      # distinct, clean high tone
        return _tone(rng.uniform(1750, 1850), dur, rng, noise=0.05)
    # A and B: a per-segment RANDOM carrier (so segments don't look alike) plus
    # a WEAK, NOISY side-tone that is the only A/B cue (A -> low, B -> high).
    # Heavy noise makes the per-frame decision unreliable, so CTC confuses them.
    carrier = rng.uniform(700, 1100)
    side = rng.uniform(400, 470) if tok == A else rng.uniform(1400, 1500)
    return _tone(carrier, dur, rng, noise=0.5, amp=1.0, side=side, side_amp=0.4)


def synth_utterance(num_pairs: int, seg_dur: float, rng):
    """Sequence of [anchor, ambiguous] pairs with the grammar C->A, D->B.

    The anchor is a clean C or D; the ambiguous token that follows is A (after
    C) or B (after D). A/B carry only a weak noisy acoustic cue, so CTC errs on
    them while the attention decoders learn the grammar and fix them.

    Returns (waveform, token_ids) with content ids in 0..3.
    """
    tokens, waves = [], []
    for _ in range(num_pairs):
        anchor = C if rng.random() < 0.5 else D
        amb = A if anchor == C else B        # grammar: C->A, D->B
        for tok in (anchor, amb):
            tokens.append(tok)
            waves.append(_render_token(tok, seg_dur, rng))
    return np.concatenate(waves), np.array(tokens, dtype=np.int64)


# --------------------------- from-scratch log-mel ---------------------------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_power(wave, n_fft, hop):
    window = _hann(n_fft)
    n_frames = 1 + max(0, (len(wave) - n_fft) // hop)
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = wave[i * hop : i * hop + n_fft] * window
        spec = np.fft.rfft(seg)
        out[i] = spec.real ** 2 + spec.imag ** 2
    return out


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sr=SAMPLE_RATE):
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


def log_mel(wave, n_fft=400, hop=160, n_mels=40):
    power = stft_power(wave, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft)
    return np.log(power @ fb.T + 1e-6).astype(np.float32), hop
