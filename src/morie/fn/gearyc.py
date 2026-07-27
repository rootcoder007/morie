# morie.fn -- function file (rootcoder007/morie)
"""Geary's C with R-style verbose result."""

from collections.abc import Sequence
from typing import Union

import numpy as np


def gearyc(x: Union[Sequence, np.ndarray], W: Union[Sequence, np.ndarray]):
    """Geary's C: alternative spatial autocorrelation.

    c = 1/(2 S^2 w..) * sum_ij w_ij (Z_i - Z_j)^2  with
    S^2 = (n-1)^{-1} sum_i (Z_i - Zbar)^2, so
    c = (n-1) sum_ij w_ij (Z_i - Z_j)^2 / (2 w.. sum_i (Z_i - Zbar)^2).

    E[c] = 1 under no autocorrelation; c < 1 indicates positive spatial
    autocorrelation (neighbours similar), c > 1 negative. The
    interpretation runs opposite to Moran's I.

    References
    ----------
    Geary, R. C. (1954). The contiguity ratio and statistical mapping.
    *The Incorporated Statistician*, 5(3), 115-145.
    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC. Sec. 1.3.2.2,
    eq. (1.15), p. 22.
    """
    from ._richresult import RichResult

    a = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    n = a.size
    if W.shape != (n, n):
        raise ValueError(f"W must be {n}x{n}, got {W.shape}.")
    S0 = W.sum()
    if S0 == 0:
        raise ValueError("W sums to zero.")
    diff_sq = (a[:, None] - a[None, :]) ** 2
    num = float((W * diff_sq).sum())
    denom = float(((a - a.mean()) ** 2).sum())
    if denom == 0:
        raise ValueError("zero variance.")
    C = ((n - 1) / (2 * S0)) * num / denom
    if C < 0.5:
        pattern = "strong positive autocorrelation (clustering)"
    elif C < 0.9:
        pattern = "weak positive autocorrelation"
    elif C <= 1.1:
        pattern = "no spatial autocorrelation (~random)"
    elif C <= 1.5:
        pattern = "weak negative autocorrelation"
    else:
        pattern = "strong negative autocorrelation (dispersion)"
    return RichResult(
        title="Geary's C spatial autocorrelation",
        summary_lines=[
            ("Geary's C", C),
            ("Expected C (null)", 1.0),
            ("Pattern", pattern),
            ("n locations", n),
        ],
        interpretation=(
            f"C = {C:.4f}; null = 1, range ~[0, 2]. {pattern.capitalize()}. "
            "More sensitive to local outliers than Moran's I."
        ),
        payload={"value": C, "statistic": C, "pattern": pattern},
    )
