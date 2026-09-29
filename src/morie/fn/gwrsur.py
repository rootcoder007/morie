# morie.fn -- function file (rootcoder007/morie)
"""GWR seemingly-unrelated regression system."""

from ._qpcore import ssum
from ._richresult import RichResult
from .gwrcoef import _flat, _setup, _wls


def gwrsur(ys, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Geographically weighted seemingly-unrelated regressions with a common design.

    For m equations y_k = X beta_k(u, v) + e_k sharing the regressors,
    local feasible GLS (SUR) collapses to equation-by-equation local WLS
    (Zellner 1962: with identical regressors GLS equals OLS equation by
    equation), so the local coefficients are the GWR estimates of each
    equation; the system adds the local cross-equation residual covariance
    Sigma_i = sum_j w_ij e_j e_j' / sum_j w_ij (kernel-weighted, the
    GW-SUR error covariance of Chen et al. 2023 at location i).

    ys is a list of m response vectors. Returns betas (m lists
    of n x p rows), residuals and local_covariance (n matrices
    m x m).

    References
    ----------
    Zellner, A. (1962). An efficient method of estimating seemingly unrelated
    regressions and tests for aggregation bias. *JASA* 57, 348-368.
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> y2 = [0.5 - X[i][0] + 0.2 * P[i][1] + 0.03 * (i % 5) for i in range(16)]
    >>> r = gwrsur([y, y2], X, P, 3.0, kernel="gaussian")
    >>> round(r["local_covariance"][0][0][1], 12)
    -0.000908629427
    """
    Y = [_flat(v) for v in ys]
    _, Xm, _, _, Wc = _setup(Y[0], X, coords, bw, kernel, adaptive)
    n, p, m = len(Y[0]), len(Xm[0]), len(Y)
    betas = [[None] * n for _ in range(m)]
    for i, w in enumerate(Wc):
        for k in range(m):
            betas[k][i] = _wls(Xm, w, Y[k])[0]
    res = [[Y[k][i] - ssum(Xm[i][a] * betas[k][i][a] for a in range(p)) for i in range(n)] for k in range(m)]
    cov = []
    for w in Wc:
        sw = ssum(w)
        cov.append([[ssum(w[j] * res[a][j] * res[b][j] for j in range(n)) / sw for b in range(m)] for a in range(m)])
    return RichResult(payload={"betas": betas, "residuals": res, "local_covariance": cov})


gwrsur_fn = gwrsur


def cheatsheet() -> str:
    return "gwrsur(ys, X, coords, bw) -> GW-SUR: per-equation local WLS + local residual covariance."
