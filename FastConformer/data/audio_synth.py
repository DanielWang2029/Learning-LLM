"""Synthetic audio + hand-rolled log-mel features (numpy only).

Nothing here downloads real audio. We synthesize short waveforms out of the
usual building blocks of speech-like signals — sine tones, linear chirps,
stacked "formant" resonances, and white noise — and turn them into log-mel
spectrograms with a from-scratch Short-Time Fourier Transform. This keeps the
whole pipeline dependency-free (numpy + stdlib) and CPU-only, exactly as a
FastConformer front-end would consume features, just tiny.

For FastConformer we build a *frame-labeling* task: each utterance is a
sequence of segments, and every segment belongs to one acoustic class. The
model must label every (subsampled) frame. This lets us compare how much the
sub-sampling factor shrinks the sequence the encoder actually has to attend
over, without needing a real transcription task.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

# The four acoustic classes the frame-labeler must distinguish. Each is a
# distinct, easily separable timbre so a tiny model can learn it in seconds.
CLASS_NAMES = ["low_tone", "high_tone", "chirp", "noise"]
NUM_CLASSES = len(CLASS_NAMES)


def sine_tone(freq: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A pure sine tone of ``freq`` Hz lasting ``dur`` seconds."""
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def chirp(f0: float, f1: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A linear frequency sweep from ``f0`` to ``f1`` Hz."""
    t = np.arange(int(dur * sr)) / sr
    k = (f1 - f0) / max(dur, 1e-6)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.sin(phase)


def formant_stack(freqs, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Sum of several sine resonances — a crude vowel-like "formant" stack."""
    t = np.arange(int(dur * sr)) / sr
    wave = np.zeros_like(t)
    for i, f in enumerate(freqs):
        wave += (0.8 ** i) * np.sin(2 * np.pi * f * t)
    return wave


def white_noise(dur: float, sr: int = SAMPLE_RATE, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.standard_normal(int(dur * sr))


def _segment_for_class(cls: int, dur: float, rng) -> np.ndarray:
    """Render one segment of a given acoustic class."""
    if cls == 0:  # low_tone: a low formant stack
        base = rng.uniform(180, 320)
        wave = formant_stack([base, 2 * base, 3 * base], dur)
    elif cls == 1:  # high_tone: a high formant stack
        base = rng.uniform(1400, 2200)
        wave = formant_stack([base, 1.5 * base], dur)
    elif cls == 2:  # chirp: sweeping tone
        lo, hi = rng.uniform(300, 600), rng.uniform(2500, 3800)
        wave = chirp(lo, hi, dur)
    else:  # noise
        wave = white_noise(dur, rng=rng)
    # A little amplitude jitter + light noise so it is not trivially clean.
    wave = wave * rng.uniform(0.7, 1.0) + 0.02 * white_noise(dur, rng=rng)
    return wave.astype(np.float32)


def synth_utterance(num_segments: int, seg_dur: float, rng):
    """Build one utterance as a random sequence of labeled segments.

    Returns the waveform and a per-sample integer label array.
    """
    waves, labels = [], []
    for _ in range(num_segments):
        cls = int(rng.integers(0, NUM_CLASSES))
        seg = _segment_for_class(cls, seg_dur, rng)
        waves.append(seg)
        labels.append(np.full(len(seg), cls, dtype=np.int64))
    return np.concatenate(waves), np.concatenate(labels)


# ----------------------------------------------------------------------------
# From-scratch log-mel front end (manual STFT + mel filterbank, numpy only).
# ----------------------------------------------------------------------------
def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_power(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude-squared STFT computed by hand. Returns (frames, n_fft/2+1)."""
    window = _hann(n_fft)
    n_frames = 1 + max(0, (len(wave) - n_fft) // hop)
    frames = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = wave[i * hop : i * hop + n_fft] * window
        spec = np.fft.rfft(seg)
        frames[i] = (spec.real ** 2 + spec.imag ** 2)
    return frames


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft/2+1)."""
    low, high = _hz_to_mel(0), _hz_to_mel(sr / 2)
    mel_points = np.linspace(low, high, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
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


def log_mel(wave: np.ndarray, n_fft: int = 400, hop: int = 160, n_mels: int = 40):
    """Waveform -> log-mel spectrogram (frames, n_mels), plus the hop used.

    n_fft=400 / hop=160 at 16 kHz gives the standard 25 ms window / 10 ms shift.
    """
    power = stft_power(wave, n_fft, hop)          # (frames, freq)
    fb = mel_filterbank(n_mels, n_fft)            # (n_mels, freq)
    mel = power @ fb.T                            # (frames, n_mels)
    return np.log(mel + 1e-6).astype(np.float32), hop


def frame_labels(sample_labels: np.ndarray, n_fft: int, hop: int, n_frames: int):
    """Reduce per-sample labels to one label per STFT frame (majority vote)."""
    out = np.empty(n_frames, dtype=np.int64)
    for i in range(n_frames):
        seg = sample_labels[i * hop : i * hop + n_fft]
        out[i] = np.bincount(seg, minlength=NUM_CLASSES).argmax()
    return out
