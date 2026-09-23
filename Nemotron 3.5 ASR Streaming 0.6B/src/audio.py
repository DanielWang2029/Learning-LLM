"""Numpy audio synthesizer + from-scratch STFT / log-mel front-end.

Nemotron 3.5 ASR is a cache-aware FastConformer-RNNT streaming model. To
reproduce its mechanisms offline on CPU with no downloads, we synthesize
speech-like audio with numpy and compute features with our own STFT + mel
filterbank — no torchaudio, no librosa, no real recordings.

Audio is a sequence of formant "phones" over a single timbre; each phone spans
several encoder frames. Two synthetic "languages" reuse the *same* acoustic
templates but map them to different token sets, so a language-ID prompt genuinely
changes the correct transcription (see `data_gen.py`). Feature-level noise is
added downstream so that integrating more frames of a phone denoises it: under a
streaming emission-latency budget, a larger chunk lets the decoder wait for more
of the phone before committing, which is the source of the accuracy/latency
trade-off in the sweep.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
HOP = 160                 # 10 ms / mel frame
N_FFT = 400               # 25 ms window
N_MELS = 32
FRAMES_PER_PHONE = 48     # mel frames per phone -> 480 ms (6 encoder frames)
SUBSAMPLE = 8             # FastConformer 8x subsampling -> 80 ms encoder frame
ENC_FRAME_MS = HOP * SUBSAMPLE * 1000 // SAMPLE_RATE   # 80 ms / encoder frame
ENC_FRAMES_PER_PHONE = FRAMES_PER_PHONE // SUBSAMPLE   # 6 encoder frames / phone

F0 = 130.0
PHONE_FORMANTS = [
    (500, 1500),
    (350, 2300),
    (700, 1100),
    (600, 1900),
]
NUM_PHONES = len(PHONE_FORMANTS)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=40.0, fmax=None):
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2)
    bin_pts = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / sr).astype(int)
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


def stft_magnitude(signal):
    n_frames = 1 + (len(signal) - N_FFT) // HOP
    frames = np.stack([signal[i * HOP:i * HOP + N_FFT] * _WINDOW for i in range(n_frames)])
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32)


def log_mel(signal):
    mel = stft_magnitude(signal) @ _MEL_FB.T
    return np.log(mel + 1e-6).astype(np.float32)


def _phone_wave(pid, n_samples, rng):
    """A harmonic stack shaped by phone pid's formant envelope."""
    t = np.arange(n_samples) / SAMPLE_RATE
    f1, f2 = PHONE_FORMANTS[pid]
    sig = np.zeros(n_samples, dtype=np.float32)
    k = 1
    while k * F0 < SAMPLE_RATE / 2:
        freq = k * F0
        w = np.exp(-((freq - f1) ** 2) / (2 * 180.0 ** 2)) \
            + 0.7 * np.exp(-((freq - f2) ** 2) / (2 * 220.0 ** 2))
        sig += ((0.1 + w) / k) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        k += 1
    return sig


def synth_utterance(phone_ids, rng):
    """Render phones into a waveform; return (wav, n_mel_frames)."""
    per = FRAMES_PER_PHONE * HOP
    segs = [_phone_wave(pid, per, rng) for pid in phone_ids]
    wav = np.concatenate(segs).astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.9
    wav = np.pad(wav, (0, N_FFT - HOP))
    return wav.astype(np.float32), len(phone_ids) * FRAMES_PER_PHONE


def features(wav, n_frames):
    return log_mel(wav)[:n_frames]
