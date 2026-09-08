# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standardization (z-score): zero mean, unit variance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_standardization"]


def geron_standardization(X, ddof=0):
    """
    Standardization (z-score): zero mean, unit variance.

    Formula: x' = (x - mu) / sigma

    Column-wise, with the *population* standard deviation by default
    (``ddof=0``), matching scikit-learn's StandardScaler. A constant
    column has ``sigma = 0`` and is an error rather than a silent NaN or
    a divide-by-zero: it carries no information to scale.

    Parameters
    ----------
    X : array-like
        Data, shape (n,) or (n, d).
    ddof : int, default 0
        Delta degrees of freedom for sigma (0 = population, 1 = sample).

    Returns
    -------
    result : RichResult
        Keys: X_std, mean, scale, estimate, n, method.

    Examples
    --------
    >>> r = geron_standardization([1.0, 2.0, 3.0])
    >>> [round(float(v), 6) for v in r["X_std"].ravel()]
    [-1.224745, 0.0, 1.224745]
    >>> round(float(r["mean"][0]), 12), round(float(r["scale"][0]), 6)
    (2.0, 0.816497)
    >>> round(float(np.mean(r["X_std"])), 12), round(float(np.std(r["X_std"])), 12)
    (0.0, 1.0)

    References
    ----------
    Géron Ch 2
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_standardization: X must be a non-empty 1-D or 2-D array")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_standardization: X contains non-finite values")
    dd = int(ddof)
    if dd < 0 or dd >= A.shape[0]:
        raise ValueError(f"geron_standardization: ddof must satisfy 0 <= ddof < n rows ({A.shape[0]}), got {dd}")

    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=dd)
    bad = np.flatnonzero(sd == 0.0)
    if bad.size:
        raise ValueError(
            f"geron_standardization: column(s) {bad.tolist()} are constant (sigma = 0) and cannot be scaled"
        )
    Z = (A - mu) / sd

    return RichResult(
        title="Standardization (z-score)",
        summary_lines=[("Rows", int(A.shape[0])), ("Columns", int(A.shape[1]))],
        interpretation="Each column now has mean 0 and unit standard deviation; outliers are not bounded.",
        payload={
            "X_std": Z,
            "Z": Z,
            "mean": mu,
            "scale": sd,
            "ddof": dd,
            "estimate": float(np.max(np.abs(Z))),
            "n": int(A.shape[0]),
            "method": f"Column-wise z-score with ddof={dd}",
        },
    )


def cheatsheet():
    return "hmstz: Standardization (z-score): zero mean, unit variance"
