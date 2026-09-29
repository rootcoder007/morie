# morie.fn -- function file (rootcoder007/morie)
"""GWR local coefficient estimates."""

import math

from ._qpcore import inverse, ssum
from .gwrbas import gwr_basic, gwr_kernel_weights


def _flat(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _rows(A):
    rows = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]


def _with_intercept(X, n):
    """Design rows; an intercept column is prepended when X has no constant column."""
    Xm = _rows(X)
    if len(Xm) != n:
        raise ValueError("X must have one row per observation")
    k = len(Xm[0])
    if any(len({r[c] for r in Xm}) == 1 and Xm[0][c] != 0.0 for c in range(k)):
        return Xm
    return [[1.0] + r for r in Xm]


def _setup(y, X, coords, bw, kernel, adaptive):
    """Response, design (with intercept), weight columns (Wc[i][j] = weight of point j at location i)."""
    yv = _flat(y)
    n = len(yv)
    Xm = _with_intercept(X, n)
    P = _rows(coords)
    D = [[math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) for j in range(n)] for i in range(n)]
    Wc = [gwr_kernel_weights(D[i], bw, kernel, adaptive) for i in range(n)]
    return yv, Xm, P, D, Wc


def _wls(Xm, w, y):
    """Local weighted least squares: (beta, C) with C = (X'WX)^{-1} X'W."""
    n, p = len(Xm), len(Xm[0])
    xtw = [[Xm[j][a] * w[j] for j in range(n)] for a in range(p)]
    A = inverse([[ssum(xtw[a][j] * Xm[j][c] for j in range(n)) for c in range(p)] for a in range(p)])
    C = [[ssum(A[a][c] * xtw[c][j] for c in range(p)) for j in range(n)] for a in range(p)]
    return [ssum(C[a][j] * y[j] for j in range(n)) for a in range(p)], C


def _fit(y, X, coords, bw, kernel, adaptive):
    yv, Xm, P, _, _ = _setup(y, X, coords, bw, kernel, adaptive)
    return gwr_basic(yv, Xm, P, bw, kernel=kernel, adaptive=adaptive)


def gwrcoef(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Local GWR coefficients ``beta_i = (X'W_iX)^{-1} X'W_i y`` at every data location.

    ``W_i`` holds the kernel weights of the distances to location ``i``
    (``GWmodel::gw.weight``; ``bw`` a distance, or a neighbour count when
    ``adaptive``); an intercept is prepended when ``X`` has no constant
    column. Thin front-end to :func:`morie.fn.gwrbas.gwr_basic`, which
    matches ``GWmodel::gwr.basic``. Returns the ``n x p`` list of rows.

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. E. (1996).
    Geographically weighted regression: a method for exploring spatial
    nonstationarity. *Geographical Analysis* 28, 281-298.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> [round(v, 10) for v in gwrcoef(y, X, P, 3.0, kernel="gaussian")[0]]
    [1.11197351, 2.0895217043]
    """
    return _fit(y, X, coords, bw, kernel, adaptive)["betas"]


gwrcoef_fn = gwrcoef


def cheatsheet() -> str:
    return "gwrcoef(y, X, coords, bw) -> local GWR coefficients (GWmodel::gwr.basic)."
