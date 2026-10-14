"""Synthetic audio plus NumPy implementations of STFT and log-mel features."""
import numpy as np


def synthetic_audio(seconds=1.2, sample_rate=16000):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    # Three tone regions make the streaming frames visibly non-stationary.
    freq = np.where(t < .4, 220, np.where(t < .8, 440, 660))
    return (0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.sin(2 * np.pi * 35 * t)).astype(np.float32)


def log_mel(audio, sample_rate=16000, frame=400, hop=160, bins=16):
    window = np.hanning(frame).astype(np.float32)
    spectra = []
    for start in range(0, len(audio) - frame + 1, hop):
        spectra.append(np.abs(np.fft.rfft(audio[start:start + frame] * window)) ** 2)
    power = np.stack(spectra)
    hz = np.linspace(0, sample_rate / 2, power.shape[1])
    edges = np.linspace(60, 3600, bins + 2)
    filters = np.zeros((bins, power.shape[1]), np.float32)
    for i in range(bins):
        left, center, right = edges[i:i + 3]
        filters[i] = np.maximum(0, np.minimum((hz-left)/(center-left), (right-hz)/(right-center)))
    return np.log(power @ filters.T + 1e-5).astype(np.float32)
