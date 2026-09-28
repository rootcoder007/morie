# morie.fn -- function file (rootcoder007/morie)
"""Eigenvector spatial filtering: Moran eigenvector maps (MEM), positive and negative sets, forward selection by
residual Moran's I, AIC, variance explained or leave-one-out prediction error, and Getis spatial filtering."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["moran_eigenvectors", "eigenvector_filtering", "getis_filter"]


def _mat(W):
    return [[float(v) for v in r] for r in np.asarray(W, dtype=float).tolist()]


def moran_eigenvectors(W) -> RichResult:
    r"""Moran eigenvector maps: eigenvectors of the doubly centred weights ``M S M``, ``S = (W + W')/2``, ``M = I - 11'/n``.

    The ``n - 1`` eigenvectors orthogonal to the constant are obtained from
    ``Q' S Q`` with ``Q`` the orthonormal Helmert basis of that complement, so
    a zero eigenvalue never mixes with the constant vector. Eigenvalues are
    sorted decreasingly; each eigenvector ``v`` (unit length, sign fixed so
    its first entry beyond ``1e-12`` in absolute value is positive) has
    Moran's I ``(n / S0) v' S v = (n / S0) lambda``, ``S0 = sum w_ij``
    (Griffith 1996; Tiefelsdorf and Boots 1995); ``positive`` and
    ``negative`` index those with ``I > -1/(n - 1)`` and ``I < -1/(n - 1)``.
    Eigenvectors of repeated eigenvalues are determined only up to rotation
    within their eigenspace.

    References
    ----------
    Griffith, D. A. (1996). Spatial autocorrelation and eigenfunctions of the
    geographic weights matrix accompanying geo-referenced data. *Canadian
    Geographer*, 40(4), 351-367.
    Dray, S., Legendre, P. and Peres-Neto, P. R. (2006). Spatial modelling:
    a comprehensive framework for principal coordinate analysis of neighbour
    matrices (PCNM). *Ecological Modelling*, 196(3-4), 483-493.

    Examples
    --------
    >>> r = moran_eigenvectors([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> [round(v, 6) + 0.0 for v in r.moran_i]
    [0.0, -1.0]
    """
    A = _mat(W)
    n = len(A)
    S = [[(A[i][j] + A[j][i]) / 2 for j in range(n)] for i in range(n)]
    s0 = ssum(ssum(r) for r in A)
    # orthonormal Helmert basis Q of the complement of the constant vector
    Q = [[0.0] * (n - 1) for _ in range(n)]
    for k in range(1, n):
        c = 1 / math.sqrt(k * (k + 1))
        for i in range(k):
            Q[i][k - 1] = c
        Q[k][k - 1] = -k * c
    SQ = [[ssum(S[i][m] * Q[m][b] for m in range(n)) for b in range(n - 1)] for i in range(n)]
    B = [[ssum(Q[m][a] * SQ[m][b] for m in range(n)) for b in range(n - 1)] for a in range(n - 1)]
    vals, vecs = np.linalg.eigh(np.asarray(B, dtype=float))
    vals = [float(v) for v in vals.tolist()]
    U = [[float(v) for v in r] for r in vecs.tolist()]
    order = sorted(range(n - 1), key=lambda k: -vals[k])
    lam, E = [], []
    for k in order:
        v = [ssum(Q[i][m] * U[m][k] for m in range(n - 1)) for i in range(n)]
        piv = next(x for x in v if abs(x) > 1e-12)
        if piv < 0:
            v = [-x for x in v]
        lam.append(vals[k])
        E.append(v)
    mi = [n / s0 * v for v in lam]
    e0 = -1 / (n - 1)
    return RichResult(
        payload={
            "eigenvalues": lam,
            "vectors": [[E[k][i] for k in range(len(E))] for i in range(n)],
            "moran_i": mi,
            "positive": [k for k, v in enumerate(mi) if v > e0],
            "negative": [k for k, v in enumerate(mi) if v < e0],
        }
    )


def _ols(X, y):
    p = len(X[0])
    XtX = [[ssum(r[a] * r[b] for r in X) for b in range(p)] for a in range(p)]
    Xty = [ssum(r[a] * v for r, v in zip(X, y)) for a in range(p)]
    beta = [float(v) for v in solve(XtX, Xty)]
    fit = [ssum(b * v for b, v in zip(beta, r)) for r in X]
    return beta, [a - b for a, b in zip(y, fit)], XtX


def _moran(e, A):
    n = len(e)
    m = ssum(e) / n
    d = [v - m for v in e]
    s0 = ssum(ssum(r) for r in A)
    return n / s0 * ssum(A[i][j] * d[i] * d[j] for i in range(n) for j in range(n)) / ssum(v * v for v in d)


def _loo(X, y):
    beta, res, XtX = _ols(X, y)
    inv = [[float(v) for v in r] for r in np.linalg.inv(np.asarray(XtX, dtype=float)).tolist()]
    press = 0.0
    for r, e in zip(X, res):
        h = ssum(r[a] * inv[a][b] * r[b] for a in range(len(r)) for b in range(len(r)))
        press += (e / (1 - h)) ** 2
    return press


def eigenvector_filtering(
    y,
    W,
    X=None,
    *,
    criterion: str = "moran",
    candidates: str = "positive",
    tol: float = 0.1,
    max_vectors: int | None = None,
) -> RichResult:
    r"""Forward selection of Moran eigenvectors as spatial filters in the linear model ``y = X b + E g + e``.

    ``X`` defaults to an intercept. Candidates are the ``positive`` (or
    ``negative``, ``all``) MEMs of :func:`moran_eigenvectors`. At each step
    the eigenvector giving the best criterion is added:

    - ``moran``: smallest ``|I|`` of the OLS residuals; stop when ``|I| <
      tol`` (Tiefelsdorf and Griffith 2007);
    - ``aic``: smallest ``AIC = n log(RSS/n) + 2 (p + 1)``; stop when AIC does
      not decrease (Griffith 2003);
    - ``press``: smallest leave-one-out prediction error ``sum (e_i/(1 -
      h_ii))^2``; stop when it does not decrease;
    - ``r2``: largest ``R^2``; stop when the gain is below ``tol``.

    At most n - p - 2 eigenvectors are added (p columns in X with
    the intercept), keeping two residual degrees of freedom. Returns the
    selected eigenvector indices, the coefficients, residual
    Moran's I, AIC, ``R^2`` and the proportion of variance explained by the
    filter ``R^2 - R^2_X``.

    References
    ----------
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering of
    spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A*, 39(5), 1193-1221.
    Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*.
    Springer.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = eigenvector_filtering([1.0, 2.0, 2.5, 4.0], W, criterion="aic")
    >>> r.selected
    [0]
    """
    yv = [float(v) for v in y]
    A = _mat(W)
    n = len(yv)
    Xb = (
        [[1.0] for _ in range(n)]
        if X is None
        else [[1.0] + [float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    )
    me = moran_eigenvectors(A)
    pool = {"positive": me["positive"], "negative": me["negative"], "all": list(range(len(me["eigenvalues"])))}[
        candidates
    ]
    E = me["vectors"]
    tss = ssum((v - ssum(yv) / n) ** 2 for v in yv)

    def fit(sel):
        X2 = [r + [E[i][k] for k in sel] for i, r in enumerate(Xb)]
        beta, res, _ = _ols(X2, yv)
        rss = ssum(e * e for e in res)
        return {
            "beta": beta,
            "res": res,
            "rss": rss,
            "aic": n * math.log(rss / n) + 2 * (len(X2[0]) + 1),
            "r2": 1 - rss / tss,
            "I": _moran(res, A),
            "X": X2,
        }

    base = fit([])
    cur, sel = base, []
    limit = len(pool) if max_vectors is None else min(max_vectors, len(pool))
    limit = min(limit, n - len(Xb[0]) - 2)  # keep at least two residual degrees of freedom
    while len(sel) < limit:
        if criterion == "moran" and abs(cur["I"]) < tol:
            break
        cands = [k for k in pool if k not in sel]
        scored = []
        for k in cands:
            f = fit(sel + [k])
            key = {"moran": abs(f["I"]), "aic": f["aic"], "press": _loo(f["X"], yv), "r2": -f["r2"]}[criterion]
            scored.append((key, k, f))
        key, k, f = min(scored, key=lambda t: (t[0], t[1]))
        if criterion == "aic" and f["aic"] >= cur["aic"]:
            break
        if criterion == "press" and key >= _loo(cur["X"], yv):
            break
        if criterion == "r2" and f["r2"] - cur["r2"] < tol:
            break
        sel.append(k)
        cur = f
    return RichResult(
        payload={
            "selected": sel,
            "coefficients": cur["beta"],
            "residual_moran": cur["I"],
            "aic": cur["aic"],
            "r2": cur["r2"],
            "filter_r2": cur["r2"] - base["r2"],
            "residuals": cur["res"],
        }
    )


def getis_filter(x, W) -> RichResult:
    r"""Getis (1995) spatial filtering of a positive variable with binary distance weights.

    ``x_i* = x_i (W_i/(n - 1)) / G_i``, ``G_i = sum_{j != i} w_ij x_j / sum_{j
    != i} x_j``, ``W_i = sum_{j != i} w_ij``; the spatial component is ``x_i -
    x_i*``. Undefined (``nan``) where ``G_i = 0``.

    References
    ----------
    Getis, A. (1995). Spatial filtering in a regression framework:
    examples using data on urban crime, regional inequality, and government
    expenditures. In L. Anselin and R. Florax (eds), *New Directions in
    Spatial Econometrics*, Springer, 172-185.

    Examples
    --------
    >>> r = getis_filter([2.0, 4.0, 6.0], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> [round(v, 6) for v in r.filtered]
    [2.5, 4.0, 4.5]
    """
    xv = [float(v) for v in x]
    A = _mat(W)
    n = len(xv)
    tot = ssum(xv)
    out, sp = [], []
    for i in range(n):
        wi = ssum(A[i][j] for j in range(n) if j != i)
        num = ssum(A[i][j] * xv[j] for j in range(n) if j != i)
        g = num / (tot - xv[i])
        f = xv[i] * (wi / (n - 1)) / g if g > 0 else float("nan")
        out.append(f)
        sp.append(xv[i] - f)
    return RichResult(payload={"filtered": out, "spatial": sp})


def cheatsheet() -> str:
    return "moran_eigenvectors / eigenvector_filtering / getis_filter -> eigenvector spatial filtering."
