# morie.fn -- function file (rootcoder007/morie)
"""Basic geographically weighted regression (Brunsdon, Fotheringham and Charlton 1996)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["gwr_basic", "gwr_kernel_weights"]

_KERNELS = ("gaussian", "exponential", "bisquare", "tricube", "boxcar")


def _k(d, b, kernel):
    if kernel == "gaussian":
        return math.exp(d * d / (-2.0 * b * b))
    if kernel == "exponential":
        return math.exp(-d / b)
    if d > b:
        return 0.0
    if kernel == "bisquare":
        return (1.0 - d * d / (b * b)) ** 2
    if kernel == "tricube":
        return (1.0 - d**3 / b**3) ** 3
    return 1.0


def gwr_kernel_weights(dists, bw: float, kernel: str = "bisquare", adaptive: bool = False):
    """Geographical weights of one regression location (``GWmodel::gw.weight``).

    ``dists`` are the distances from the location to every data point.  With
    ``adaptive`` the bandwidth is the ``bw``-th smallest distance (self
    included), or ``bw/n`` times the largest when ``bw > n``.

    Examples
    --------
    >>> [round(w, 6) for w in gwr_kernel_weights([0.0, 1.0, 2.0, 3.0], 2.5)]
    [1.0, 0.7056, 0.1296, 0.0]
    """
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}")
    d = [float(v) for v in dists]
    b = float(bw)
    if adaptive:
        n = len(d)
        b = sorted(d)[int(bw) - 1] if bw / n <= 1 else bw / n * max(d)
    return [_k(v, b, kernel) for v in d]


def gwr_basic(y, X, coords, bw: float, *, kernel: str = "bisquare", adaptive: bool = False) -> RichResult:
    r"""Geographically weighted regression with its diagnostics.

    At each data location ``i`` the coefficients are the weighted least
    squares estimate ``beta_i = (X' W_i X)^{-1} X' W_i y`` with kernel
    weights of the distances to ``i`` (:func:`gwr_kernel_weights`).  With
    ``C_i = (X' W_i X)^{-1} X' W_i`` and the hat matrix ``S`` (row ``i`` is
    ``x_i' C_i``): ``sigma^2 = RSS / (n - 2 tr S + tr S'S)``, standard errors
    ``sqrt(sigma^2 diag(C_i C_i'))``, ``edf = n - 2 tr S + tr S'S``, ``enp =
    2 tr S - tr S'S``, ``AICc = n log(RSS/n) + n log(2 pi) + n (n + tr S) /
    (n - 2 - tr S)`` and the local ``R^2`` from the kernel-weighted total and
    residual sums of squares, exactly as ``GWmodel::gwr.basic``
    (Euclidean distances; Brunsdon, Fotheringham and Charlton 1996;
    Fotheringham, Brunsdon and Charlton 2002).

    :param y: Response (n,).
    :param X: Design matrix (n, p) including any intercept column.
    :param coords: (n, 2) coordinates.
    :param bw: Bandwidth (a distance, or a neighbour count when ``adaptive``).
    :param kernel: ``gaussian``, ``exponential``, ``bisquare``, ``tricube`` or
        ``boxcar``.
    :param adaptive: Adaptive (nearest-neighbour) bandwidth.
    :return: :class:`RichResult` with ``betas`` (n x p), ``se``, ``tvalues``,
        ``fitted``, ``residuals``, ``local_r2`` and ``diagnostics`` (``RSS``,
        ``AIC``, ``AICc``, ``BIC``, ``edf``, ``enp``, ``R2``, ``R2_adj``,
        ``trS``, ``trStS``, ``sigma2``).

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. E. (1996).
    Geographically weighted regression: a method for exploring spatial
    nonstationarity. *Geographical Analysis*, 28(4), 281-298.
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley, Chichester.

    Examples
    --------
    >>> P = [(x * 1.0, y * 1.0) for x in range(4) for y in range(4)]
    >>> X = [[1.0, 0.3 * i % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * r[1] + 0.1 * p[0] for r, p in zip(X, P)]
    >>> r = gwr_basic(y, X, P, 3.0, kernel="gaussian")
    >>> round(r.diagnostics["AICc"], 6)
    -20.824713
    """
    yv = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    Xm = [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    n, p = len(yv), len(Xm[0])
    if len(Xm) != n or len(P) != n:
        raise ValueError("y, X and coords must share n")
    D = [[math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) for j in range(n)] for i in range(n)]
    Wcols = [gwr_kernel_weights([D[j][i] for j in range(n)], bw, kernel, adaptive) for i in range(n)]
    betas, var, S = [], [], []
    for i in range(n):
        w = Wcols[i]
        xtw = [[Xm[j][a] * w[j] for j in range(n)] for a in range(p)]
        A = [[ssum(xtw[a][j] * Xm[j][c] for j in range(n)) for c in range(p)] for a in range(p)]
        Ai = [[float(v) for v in r] for r in inverse(A)]
        C = [[ssum(Ai[a][c] * xtw[c][j] for c in range(p)) for j in range(n)] for a in range(p)]
        betas.append([ssum(C[a][j] * yv[j] for j in range(n)) for a in range(p)])
        var.append([ssum(v * v for v in C[a]) for a in range(p)])
        S.append([ssum(Xm[i][a] * C[a][j] for a in range(p)) for j in range(n)])
    fitted = [ssum(Xm[i][a] * betas[i][a] for a in range(p)) for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    rss = ssum(v * v for v in resid)
    trS = ssum(S[i][i] for i in range(n))
    trStS = ssum(v * v for r in S for v in r)
    edf = n - 2.0 * trS + trStS
    enp = 2.0 * trS - trStS
    s2 = rss / edf
    se = [[math.sqrt(s2 * v) for v in r] for r in var]
    tv = [[betas[i][a] / se[i][a] if se[i][a] > 0 else float("nan") for a in range(p)] for i in range(n)]
    ybar = sum(yv) / n
    dyb = [(v - ybar) ** 2 for v in yv]
    dyh = [v * v for v in resid]
    # GWmodel: TSSw <- W %*% dybar2 with W[r, c] the weight of point r in the regression at c
    local_r2 = []
    for i in range(n):
        tssw = ssum(Wcols[c][i] * dyb[c] for c in range(n))
        rssw = ssum(Wcols[c][i] * dyh[c] for c in range(n))
        local_r2.append((tssw - rssw) / tssw if tssw > 0 else float("nan"))
    yss = ssum(dyb)
    r2 = 1.0 - rss / yss
    lg = n * math.log(rss / n) + n * math.log(2.0 * math.pi)
    diag = {
        "RSS": rss,
        "AIC": lg + n + trS,
        "AICc": lg + n * (n + trS) / (n - 2.0 - trS),
        "BIC": lg + math.log(n) * trS,
        "edf": edf,
        "enp": enp,
        "R2": r2,
        "R2_adj": 1.0 - (1.0 - r2) * (n - 1.0) / (edf - 1.0),
        "trS": trS,
        "trStS": trStS,
        "sigma2": s2,
    }
    return RichResult(
        payload={
            "betas": betas,
            "se": se,
            "tvalues": tv,
            "fitted": fitted,
            "residuals": resid,
            "local_r2": local_r2,
            "diagnostics": diag,
        }
    )


def cheatsheet() -> str:
    return "gwr_basic(y, X, coords, bw) -> GWR coefficients, SEs, t-values, local R2, AICc (GWmodel::gwr.basic)."
