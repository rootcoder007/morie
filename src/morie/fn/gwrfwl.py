# morie.fn -- function file (rootcoder007/morie)
"""GWR Frisch-Waugh-Lovell local partitioned regression."""

from ._qpcore import ssum
from ._richresult import RichResult
from .gwrcoef import _rows, _setup, _wls


def _partial(Z1, C, v):
    """v minus its local WLS projection on Z1 (C = (Z1'WZ1)^{-1} Z1'W)."""
    n, k1 = len(v), len(C)
    b = [ssum(C[a][j] * v[j] for j in range(n)) for a in range(k1)]
    return [v[j] - ssum(Z1[j][a] * b[a] for a in range(k1)) for j in range(n)]


def gwrfwl(y, X1, X2, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Local Frisch-Waugh-Lovell partitioned regression: GWR coefficients of X2 net of X1.

    At each location i the response and the columns of X2 are
    residualised on X1 (an intercept is added when X1 has no
    constant column) by weighted least squares with the kernel weights
    W_i; the local coefficients of X2 are the W_i-weighted least
    squares of the partialled y on the partialled X2. By the
    Frisch-Waugh-Lovell theorem (weighted version) these equal the X2
    block of the full local estimate (Z'W_iZ)^{-1}Z'W_iy, Z = [X1,
    X2] (Frisch and Waugh 1933; Lovell 1963), which is returned alongside
    for checking.

    References
    ----------
    Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared
    with individual trends. *Econometrica* 1, 387-401.
    Lovell, M. C. (1963). Seasonal adjustment of economic time series and
    multiple regression analysis. *JASA* 58, 993-1010.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> X1 = [[P[i][0]] for i in range(16)]
    >>> r = gwrfwl(y, X1, X, P, 3.0, kernel="gaussian")
    >>> round(r["betas"][0][0], 10)
    2.0396391378
    """
    yv, Z1, P, _, Wc = _setup(y, X1, coords, bw, kernel, adaptive)
    X2m = _rows(X2)
    n, q = len(yv), len(X2m[0])
    k1 = len(Z1[0])
    betas, full = [], []
    for w in Wc:
        _, C = _wls(Z1, w, yv)
        yt = _partial(Z1, C, yv)
        cols = [_partial(Z1, C, [r[c] for r in X2m]) for c in range(q)]
        Xt = [[cols[c][j] for c in range(q)] for j in range(n)]
        betas.append(_wls(Xt, w, yt)[0])
        full.append(_wls([Z1[j] + X2m[j] for j in range(n)], w, yv)[0][k1:])
    return RichResult(payload={"betas": betas, "full_model_betas": full})


gwrfwl_fn = gwrfwl


def cheatsheet() -> str:
    return "gwrfwl(y, X1, X2, coords, bw) -> local FWL coefficients of X2 net of X1 (equal the full GWR block)."
