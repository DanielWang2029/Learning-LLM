"""Fit and extrapolate a neural scaling law: loss vs. parameter count.

The GPT-4 Technical Report states that the final loss of GPT-4 was predicted
from a family of smaller models trained with 1,000x–10,000x less compute,
using a power law of the form ``L(N) = a * N**(-alpha) + E`` (an irreducible
term ``E`` plus a reducible power-law term). We reproduce exactly that fit here
with nothing but NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalingLaw:
    a: float
    alpha: float
    E: float

    def predict(self, N: float | np.ndarray):
        return self.a * np.asarray(N, dtype=float) ** (-self.alpha) + self.E


def _fit_loglinear(N: np.ndarray, reducible: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line through (log N, log reducible) -> a, alpha, sse."""
    x = np.log(N)
    y = np.log(reducible)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    a = float(np.exp(intercept))
    pred = intercept + slope * x
    sse = float(np.sum((y - pred) ** 2))
    return a, alpha, sse


def fit_power_law(
    N: np.ndarray, L: np.ndarray, E_floor: float | None = None
) -> ScalingLaw:
    """Fit ``L = a N^-alpha + E``.

    ``E`` (irreducible loss) is found by a coarse 1-D search that minimizes the
    residual of the log-linear fit of the reducible loss ``L - E``; ``a`` and
    ``alpha`` then come from a closed-form least-squares line. This mirrors the
    report's power-law-with-offset without needing SciPy.
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)
    hi = float(L.min()) - 1e-4
    lo = 0.0 if E_floor is None else max(0.0, E_floor - 0.5)
    best = None
    for E in np.linspace(lo, hi, 200):
        reducible = L - E
        if np.any(reducible <= 0):
            continue
        a, alpha, sse = _fit_loglinear(N, reducible)
        if best is None or sse < best[0]:
            best = (sse, a, alpha, float(E))
    if best is None:  # degenerate fallback: pure power law, no offset
        a, alpha, _ = _fit_loglinear(N, L)
        return ScalingLaw(a=a, alpha=alpha, E=0.0)
    _, a, alpha, E = best
    return ScalingLaw(a=a, alpha=alpha, E=E)
