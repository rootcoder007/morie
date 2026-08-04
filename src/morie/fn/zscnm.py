# morie.fn -- function file (rootcoder007/morie)
"""Z-score (standardization) normalization.

MVSML (2022) sec. 2.6 p.57, "Standardization": subtract the mean and
divide by the standard deviation, X_i = (X_i - mu) / sigma.  Read from
the chapter-2 split PDF (pdf page 23 of the 35-70 split, book p.57).
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["zscore_normalization"]


def zscore_normalization(x, ddof=1):
    """Standardize a variable to mean zero and unit variance.

    p.57: "calculating its mean, mu, and standard deviation, sigma,
    for each input or output.  The standardized values are then
    calculated as X_i = (X_i - mu)/sigma", giving a variable with mean
    zero and variance one.  ``ddof=1`` is the sample standard
    deviation, matching the R ``scale()`` the book's own worked
    examples call; pass ``ddof=0`` for the population divisor.

    Parameters
    ----------
    x : array-like, the variable to standardize.
    ddof : int, delta degrees of freedom of the standard deviation.

    Returns
    -------
    RichResult with keys estimate (the standard deviation divided
    out), x_std, mean, sd, n, method.

    References
    ----------
    MVSML (2022) sec. 2.6 p.57, Standardization.
    """
    v = [float(t) for t in x]
    n = len(v)
    mu = sum(v) / n
    den = n - int(ddof)
    sd = math.sqrt(sum((t - mu) ** 2 for t in v) / den) if den > 0 else 0.0
    out = [(t - mu) / sd for t in v] if sd > 0 else [0.0] * n
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sd), "x_std": out, "mean": float(mu),
        "sd": float(sd), "n": n,
        "method": "z-score standardization (MVSML 2022 p.57)",
    }), "zscnm")


def cheatsheet():
    return "zscnm: Z-score (standardization) normalization"


# compact alias per ledger/NAMING.md
zscorenorm = zscore_normalization
