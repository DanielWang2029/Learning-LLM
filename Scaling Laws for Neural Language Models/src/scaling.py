"""Power-law fitting for the scaling laws demo.

Kaplan et al. (2020) find that the language-modeling loss falls off as a power
law in the (non-embedding) parameter count N:

    L(N) ≈ (Nc / N) ** alpha_N            (paper Eq. 1.1 / §1)

Taking logs turns this into a straight line, which is why the paper's plots are
all log-log:

    log L = alpha_N * log(Nc) - alpha_N * log N
          = b - alpha_N * log N

So a plain least-squares fit of ``log L`` against ``log N`` recovers the
exponent ``alpha_N`` (the negative slope) and ``Nc`` (from the intercept).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PowerLawFit:
    alpha: float  # scaling exponent alpha_N
    Nc: float  # critical scale N_c
    r2: float  # coefficient of determination of the log-log fit

    def predict(self, N: np.ndarray) -> np.ndarray:
        return (self.Nc / np.asarray(N, dtype=float)) ** self.alpha


def fit_power_law(N: np.ndarray, L: np.ndarray) -> PowerLawFit:
    """Fit L(N) = (Nc/N)^alpha via least squares in log-log space."""
    logN = np.log(np.asarray(N, dtype=float))
    logL = np.log(np.asarray(L, dtype=float))

    # logL = b + slope * logN, with slope = -alpha.
    slope, b = np.polyfit(logN, logL, 1)
    alpha = -slope
    # b = alpha * log(Nc)  ->  Nc = exp(b / alpha)
    Nc = float(np.exp(b / alpha)) if alpha != 0 else float("inf")

    pred = b + slope * logN
    ss_res = float(np.sum((logL - pred) ** 2))
    ss_tot = float(np.sum((logL - logL.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return PowerLawFit(alpha=float(alpha), Nc=Nc, r2=float(r2))
