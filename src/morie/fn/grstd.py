# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standardization (z-score normalization) to zero mean, unit variance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_standardization"]

_METHOD = "Standardization (z-score scaling)"


def geron_standardization(X, ddof=0):
    r"""Centre and scale every column.

    .. math::
        x_{\text{scaled}} = \frac{x - \mu}{\sigma}

    Unlike min-max scaling this is not bounded, which is precisely why
    Géron prefers it when outliers are present: a single extreme value
    squashes a min-max range but only nudges a standard deviation.  A
    constant column has :math:`\sigma = 0` and cannot be standardised at
    all -- that raises rather than dividing by zero.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, n)
    ddof : int, optional
        Delta degrees of freedom for the scale (0 = population, as in
        scikit-learn's StandardScaler).

    Returns
    -------
    RichResult
        Payload keys ``scaled``, ``mean``, ``scale``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Feature Scaling section (standardization).

    Examples
    --------
    ``[1, 2, 3]`` has mean 2 and population sd ``sqrt(2/3) = 0.8165``:

    >>> r = geron_standardization([1.0, 2.0, 3.0])
    >>> [round(v, 6) for v in r["scaled"]]
    [-1.224745, 0.0, 1.224745]
    >>> round(r["scale"][0], 6)
    0.816497

    Output has mean 0 and variance 1 by construction:

    >>> import numpy as np
    >>> round(float(np.var(r["scaled"])), 12)
    1.0
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
        flat = True
    else:
        flat = False
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty 1-D or 2-D array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    ddof = int(ddof)
    if ddof < 0 or ddof >= A.shape[0]:
        raise ValueError(f"ddof must lie in [0, {A.shape[0] - 1}], got {ddof}.")

    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=ddof)
    bad = np.flatnonzero(sd == 0)
    if bad.size:
        raise ValueError(
            f"columns {bad.tolist()} are constant (sd = 0); standardization is undefined."
        )
    Z = (A - mu) / sd
    out = Z.ravel().tolist() if flat else Z.tolist()

    return RichResult(
        title="Standardization",
        summary_lines=[("Rows", int(A.shape[0])), ("Columns", int(A.shape[1]))],
        payload={
            "scaled": out,
            "mean": mu.tolist(),
            "scale": sd.tolist(),
            "estimate": out,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grstd: z = (x - mean)/sd per column, ddof=0 by default; constant columns raise"
