"""Numpy two-speaker audio synthesizer + from-scratch STFT / log-mel front-end.

VibeVoice-ASR-Streaming (Tu et al., Microsoft, 2026; arXiv:2609.02812) is a
*speaker-attributed* ASR model: it must emit both the words and *who said them*.
To reproduce that offline on CPU with no downloads, we synthesize two-speaker
conversations with numpy and compute features with our own STFT + mel
filterbank — no torchaudio, no librosa, no real recordings.

Design (mirrors real speech so both tasks are learnable and separable):
  * Speaker identity  = the pitch comb (fundamental f0). Two distinct timbres,
    f0 = 110 Hz (speaker A) and 170 Hz (speaker B), present in *every* frame.
  * Phone identity     = the spectral envelope (formants). Each phone weights the
    harmonics by a distinct pair of formant resonances.
  * Each phone spans two latent frames: an ONSET frame (speaker comb only, flat
    envelope → phone not yet identifiable) and a PEAK frame (formant envelope →
    phone identifiable). This is why a small amount of *lookahead* audio helps
    near a chunk boundary: you can hear WHO is starting to speak from the onset,
    but need the peak to know WHAT was said.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
HOP = 512                 # samples/frame; n_fft == hop => clean 1 frame per hop
N_FFT = 512
N_MELS = 32
FRAMES_PER_PHONE = 2      # [onset, peak]
FRAME_MS = HOP * 1000 // SAMPLE_RATE   # 32 ms / frame

SPEAKER_F0 = [100.0, 190.0]            # two distinct synthetic timbres (pitch)
# Per-speaker spectral tilt gives each voice a distinct global "colour" on top of
# pitch: speaker 0 is dark (low-heavy), speaker 1 is bright (high-heavy).
SPEAKER_TILT = [("dark", 1500.0), ("bright", 4000.0)]
NUM_SPEAKERS = len(SPEAKER_F0)

PHONE_FORMANTS = [
    (600, 1000),
    (350, 2400),
    (750, 1350),
    (450, 1800),
    (550, 2600),
    (800, 1150),
]
NUM_PHONES = len(PHONE_FORMANTS)
PHONE_NAMES = ["aa", "iy", "eh", "uw", "ih", "ao"]


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=40.0, fmax=None):
    """Triangular mel filterbank built from scratch (n_mels x bins)."""
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


def stft_magnitude(signal):
    n_frames = 1 + (len(signal) - N_FFT) // HOP
    frames = np.stack([signal[i * HOP:i * HOP + N_FFT] * _WINDOW for i in range(n_frames)])
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32)


def log_mel(signal):
    """Waveform -> log-mel spectrogram (frames x N_MELS)."""
    mel = stft_magnitude(signal) @ _MEL_FB.T
    return np.log(mel + 1e-6).astype(np.float32)


def _speaker_tilt(sid, freq):
    """Per-speaker spectral colour (an extra timbre cue beyond pitch)."""
    kind, scale = SPEAKER_TILT[sid]
    if kind == "dark":
        return np.exp(-freq / scale)
    return 0.25 + np.minimum(freq / scale, 2.0)


def _harmonics(sid, formant, express, n_samples, rng):
    """One HOP-length frame of a harmonic stack for a speaker and phone.

    express in [0,1] scales how strongly the formant envelope shapes the
    harmonics: 0 => flat (onset, phone-agnostic), 1 => full formant (peak).
    The speaker's f0 (comb spacing) and spectral tilt are present in every frame,
    so WHO is speaking is always recoverable; WHAT is said needs the peak.
    """
    f0 = SPEAKER_F0[sid]
    t = np.arange(n_samples) / SAMPLE_RATE
    f1, f2 = formant
    sig = np.zeros(n_samples, dtype=np.float32)
    k = 1
    while k * f0 < SAMPLE_RATE / 2:
        freq = k * f0
        formant_w = np.exp(-((freq - f1) ** 2) / (2 * 200.0 ** 2)) \
            + 0.7 * np.exp(-((freq - f2) ** 2) / (2 * 240.0 ** 2))
        env = (1.0 - express) / k + express * (0.15 + formant_w) / k
        amp = env * _speaker_tilt(sid, freq)
        sig += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        k += 1
    return sig


def synth_conversation(phone_ids, speaker_ids, rng, noise=0.01):
    """Render a phone/speaker sequence into a single waveform.

    Returns (waveform, n_frames). Frame 2p is phone p's onset, frame 2p+1 its
    peak; frames map 1:1 to log_mel frames because n_fft == hop.
    """
    segs = []
    for pid, sid in zip(phone_ids, speaker_ids):
        formant = PHONE_FORMANTS[pid]
        onset = _harmonics(sid, formant, 0.0, HOP, rng)   # speaker only
        peak = _harmonics(sid, formant, 1.0, HOP, rng)    # speaker + phone
        segs.append(onset)
        segs.append(peak)
    wav = np.concatenate(segs).astype(np.float32)
    wav += noise * rng.standard_normal(len(wav)).astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.9
    # Pad so the final frame is fully covered by an STFT window.
    wav = np.pad(wav, (0, N_FFT - HOP))
    return wav.astype(np.float32), len(phone_ids) * FRAMES_PER_PHONE


def features(wav, n_frames):
    """log-mel features, trimmed to exactly n_frames."""
    return log_mel(wav)[:n_frames]
