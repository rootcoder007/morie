# morie.fn -- function file (rootcoder007/morie)
"""Mahalanobis data depth."""

from __future__ import annotations

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["mahalanobis_depth"]


def mahalanobis_depth(x, mu=None, Sigma=None):
    """Mahalanobis depth of each row of ``x``.

    Formula: ``MD(z) = 1 / (1 + (z - mu)' Sigma^{-1} (z - mu))``.

    Depth is 1 exactly at ``mu`` and decreases monotonically outwards,
    so a depth ordering is a centre-outward ordering; the map is affine
    invariant, because ``Sigma^{-1}`` absorbs any non-singular linear
    transform of the data.  ``mu`` and ``Sigma`` default to the sample
    mean and the sample covariance with divisor ``n - 1``, which makes
    the depth non-robust -- one far outlier inflates ``Sigma`` and
    flattens every depth; supply robust estimates if that matters.

    Parameters
    ----------
    x : array-like
        ``n x p`` matrix of points, one per row.
    mu : array-like, optional
        Centre; the column means if omitted.
    Sigma : array-like, optional
        Scatter matrix; the sample covariance if omitted.

    Returns
    -------
    RichResult
        ``depth`` (one per row), ``estimate`` (the maximum depth),
        ``deepest`` (its 0-based row index), ``d2`` (the squared
        Mahalanobis distances), ``n``, ``p``, ``method``.

    References
    ----------
    Liu (1990), On a notion of data depth based on random simplices,
    Annals of Statistics 18:405-414, introduces data depth as a
    centre-outward ordering; the Mahalanobis form ``1/(1 + d^2)`` is the
    one given by Liu and Singh (1993), A quality index based on data
    depth and multivariate rank tests, JASA 88:252-260, and used by
    Mahalanobis (1936).  The Project Euclid PDF for Liu (1990) could not
    be retrieved from this host (the fetch returned a 1.2 kB error page,
    not the article), so this is the standard published form; it is
    anchored in the test harness on the fact that the depth at ``mu`` is
    exactly 1 and that the depths are unchanged by an affine
    transformation of the data, neither of which depends on this code.
    """
    X = T.mat(x)
    n = len(X)
    p = len(X[0])
    if any(len(r) != p for r in X):
        raise ValueError("x must be rectangular")
    if mu is None:
        mu = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    else:
        mu = T.vec(mu)
    if len(mu) != p:
        raise ValueError("mu must have one entry per column of x")
    if Sigma is None:
        if n < 2:
            raise ValueError("need at least 2 rows to estimate Sigma")
        S = [[sum((X[i][a] - mu[a]) * (X[i][b] - mu[b]) for i in range(n)) / (n - 1.0)
              for b in range(p)] for a in range(p)]
    else:
        S = T.mat(Sigma)
    if len(S) != p or any(len(r) != p for r in S):
        raise ValueError("Sigma must be p x p")
    ident = [[1.0 if a == b else 0.0 for b in range(p)] for a in range(p)]
    Sinv = _solve(S, ident)
    d2 = []
    for i in range(n):
        z = [X[i][j] - mu[j] for j in range(p)]
        q = 0.0
        for a in range(p):
            za = sum(Sinv[a][b] * z[b] for b in range(p))
            q += z[a] * za
        d2.append(q)
    dep = [1.0 / (1.0 + q) for q in d2]
    best = max(range(n), key=lambda i: (dep[i], -i))
    return RichResult(
        payload={
            "depth": dep,
            "estimate": float(dep[best]),
            "deepest": int(best),
            "d2": d2,
            "n": int(n),
            "p": int(p),
            "method": "Mahalanobis depth",
        }
    )


def _solve(A, B):
    """Solve ``A Z = B`` by Gauss-Jordan with partial pivoting."""
    k = len(A)
    aug = [A[i][:] + B[i][:] for i in range(k)]
    m = len(aug[0])
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-300:
            raise ValueError("singular scatter matrix")
        aug[c], aug[piv] = aug[piv], aug[c]
        d = aug[c][c]
        aug[c] = [v / d for v in aug[c]]
        for r in range(k):
            if r == c:
                continue
            f = aug[r][c]
            if f != 0.0:
                aug[r] = [aug[r][j] - f * aug[c][j] for j in range(m)]
    return [row[k:] for row in aug]


def cheatsheet():
    return "mahalanobis_depth(x, mu, Sigma): 1/(1 + squared Mahalanobis distance)."


# compact alias per ledger/NAMING.md
mahaldepth = mahalanobis_depth
