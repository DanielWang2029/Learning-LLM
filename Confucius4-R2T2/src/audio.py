"""Numpy audio synthesizer + a from-scratch STFT / log-mel front-end.

Confucius4-R2T2 is an audio ASR model. To reproduce its *streaming decoding
mechanism* offline, on CPU, with no downloads, we synthesize speech-like audio
with numpy and turn it into features with our own STFT and mel filterbank — no
torchaudio, no librosa, no real recordings.

The synthesizer produces a sequence of "phones". Each phone is a short vowel-
like sound built from a harmonic stack (a pitch comb set by the speaker's f0)
shaped by a formant envelope (the resonances that give the phone its identity).
Crucially, the identifying formant is only fully expressed in the *second half*
of each phone; the onset is deliberately generic. This mirrors real speech,
where a sound is often ambiguous until it has (nearly) finished — and it is
exactly what makes trailing tokens unstable in a streaming decoder.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400          # 25 ms analysis window
HOP = 160            # 10 ms hop  -> one feature frame every 10 ms
N_MELS = 32
FRAMES_PER_PHONE = 20   # 200 ms per phone

# One speaker for this single-stream ASR reproduction.
SPEAKER_F0 = 120.0

# Each phone is defined by two formant centre frequencies (Hz). A distinct
# formant pair gives each phone a distinct spectral envelope.
PHONE_FORMANTS = [
    (700, 1200),
    (400, 2200),
    (600, 1700),
    (350, 900),
    (500, 2600),
    (800, 1500),
    (450, 1900),
    (650, 1050),
]
NUM_PHONES = len(PHONE_FORMANTS)
PHONE_NAMES = ["aa", "iy", "eh", "uw", "ih", "ao", "ae", "ow"]


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=40.0, fmax=None):
    """Triangular mel filterbank, implemented from scratch (shape: n_mels x bins)."""
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ctr):
            if ctr > lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb


_MEL_FB = mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def stft_magnitude(signal: np.ndarray, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    """Short-time Fourier transform magnitude, one row per frame."""
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    n_frames = 1 + (len(signal) - n_fft) // hop
    frames = np.stack(
        [signal[i * hop : i * hop + n_fft] * _WINDOW for i in range(n_frames)]
    )
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return np.abs(spec).astype(np.float32)


def log_mel(signal: np.ndarray) -> np.ndarray:
    """Waveform -> log-mel spectrogram (frames x N_MELS)."""
    mag = stft_magnitude(signal)
    mel = mag @ _MEL_FB.T
    return np.log(mel + 1e-6).astype(np.float32)


def _synth_phone(phone_id: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Render one phone as a harmonic stack shaped by its formant envelope.

    The formant envelope is ramped in so the phone's identity is only fully
    present in its second half; the onset is nearly formant-free.
    """
    n_samples = n_frames * HOP + (N_FFT - HOP)
    t = np.arange(n_samples) / SAMPLE_RATE
    f1, f2 = PHONE_FORMANTS[phone_id]

    sig = np.zeros(n_samples, dtype=np.float32)
    k = 1
    while k * SPEAKER_F0 < SAMPLE_RATE / 2:
        freq = k * SPEAKER_F0
        env = np.exp(-((freq - f1) ** 2) / (2 * 220.0 ** 2)) \
            + 0.7 * np.exp(-((freq - f2) ** 2) / (2 * 260.0 ** 2))
        amp = (0.2 + env) / k
        sig += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        k += 1

    # Formant ramp: onset ~ generic, second half ~ fully identifying.
    ramp = np.clip(np.linspace(-0.3, 1.0, n_samples), 0.05, 1.0)
    baseline = 0.15  # generic energy always present so the onset is audible
    sig = sig * (baseline + (1 - baseline) * ramp)

    sig += 0.01 * rng.standard_normal(n_samples).astype(np.float32)
    peak = np.max(np.abs(sig)) + 1e-8
    return (sig / peak * 0.9).astype(np.float32)


def synth_utterance(phone_ids, rng: np.random.Generator):
    """Concatenate phones into one waveform + return per-phone frame boundaries."""
    parts, bounds, cursor = [], [], 0
    for pid in phone_ids:
        wav = _synth_phone(pid, FRAMES_PER_PHONE, rng)
        parts.append(wav)
        bounds.append((cursor, cursor + FRAMES_PER_PHONE))
        cursor += FRAMES_PER_PHONE
    # Overlap-free concatenation on the frame grid.
    full = np.concatenate([p[: FRAMES_PER_PHONE * HOP] for p in parts])
    return full.astype(np.float32), bounds


def phone_templates() -> np.ndarray:
    """Clean full-phone log-mel template per phone (for the matched-filter ASR)."""
    rng = np.random.default_rng(0)
    tmpl = []
    for pid in range(NUM_PHONES):
        wav = _synth_phone(pid, FRAMES_PER_PHONE, rng)
        feats = log_mel(wav)[:FRAMES_PER_PHONE]
        tmpl.append(feats.mean(axis=0))
    return np.stack(tmpl)
