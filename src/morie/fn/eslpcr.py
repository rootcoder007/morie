# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Principal components regression (ESL Ch 3.5.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_pcr"]


def esl_pcr(X, y, M):
    """
    PCR: regress y on the first M principal components.

    Formula: beta^PCR = sum_{m<=M} (z_m'y / z_m'z_m) v_m, where the
    z_m are principal components of the CENTRED design and v_m their
    loadings. Because the components are orthogonal, each coefficient
    is a simple univariate regression -- no matrix inverse needed.
    PCR chooses directions by variance in X ALONE, ignoring y, which
    is exactly how it differs from PLS (eslpls) and why it can
    discard a low-variance direction that happens to predict well.
    Coefficients come back on the ORIGINAL centred scale with the
    intercept reported separately.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix WITHOUT an intercept column (it is centred).
    y : array-like, shape (n,)
        Response.
    M : int
        Components to keep, 1 <= M <= min(n - 1, p).

    Returns
    -------
    result : dict
        Keys: estimate (first coefficient), beta, intercept,
        variance_explained, singular_values, M, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.5.1 (Eq. 3.61).

    Examples
    --------
    With M = p, PCR reproduces OLS on the centred design:

    >>> import numpy as np
    >>> X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]]
    >>> y = [1.0, 2.0, 5.0, 5.0]
    >>> full = esl_pcr(X, y, 2)
    >>> Xc = np.asarray(X) - np.mean(X, axis=0)
    >>> ols = np.linalg.lstsq(Xc, np.asarray(y) - np.mean(y), rcond=None)[0]
    >>> bool(np.allclose(full["beta"], ols))
    True
    >>> one = esl_pcr(X, y, 1)
    >>> one["variance_explained"] < 1.0
    True
    >>> esl_pcr(X, y, 5)
    Traceback (most recent call last):
        ...
    ValueError: M must lie in [1, 2]; got 5.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n, p = X.shape
    M = int(M)
    if y.size != n:
        raise ValueError(f"X has {n} rows but y has {y.size} entries.")
    kmax = min(n - 1, p)
    if not 1 <= M <= kmax:
        raise ValueError(f"M must lie in [1, {kmax}]; got {M}.")
    xbar = X.mean(axis=0)
    ybar = float(y.mean())
    Xc = X - xbar
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    beta = np.zeros(p)
    for m in range(M):
        z = Xc @ V[:, m]
        zz = float(z @ z)
        if zz == 0:
            continue
        beta += (float(z @ (y - ybar)) / zz) * V[:, m]
    total = float(np.sum(S ** 2))
    return RichResult(payload={
        "estimate": float(beta[0]), "beta": [float(v) for v in beta],
        "intercept": ybar - float(xbar @ beta),
        "variance_explained": float(np.sum(S[:M] ** 2) / total) if total > 0 else float("nan"),
        "singular_values": [float(v) for v in S],
        "M": M, "n": int(n), "p": int(p),
        "method": "PCR on centred X; directions chosen by X-variance only"})


def cheatsheet():
    return "eslpcr: sum (z_m'y/z_m'z_m) v_m; ignores y when choosing directions"
