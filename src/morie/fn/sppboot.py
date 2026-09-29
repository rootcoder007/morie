# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel bootstrap."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from .sppanel import _cols, _demean, spatial_panel_ml
from .sppfe import _stack


def sppboot(y, X, W, time_id, unit_id, B=9, seed=0):
    r"""Residual bootstrap of the fixed-effects spatial lag panel model.

    The within spatial lag model is fitted (:func:`morie.fn.sppfe.sppfe`);
    with the within-transformed regressors X~, the centred residuals e
    and A = I - rho W, each replicate draws e* from e with
    replacement (Philox stream b of seed, index floor(u NT)), sets
    y*_t = A^{-1}(X~_t beta + e*_t) period by period and re-estimates the
    model; the bootstrap standard errors of rho and beta are the
    standard deviations (B - 1 divisor) over the replicates (Efron and
    Tibshirani 1993, sec. 9.5; Anselin 1990).

    References
    ----------
    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the
    Bootstrap*. Chapman and Hall.
    Anselin, L. (1990). Some robust approaches to testing and estimation in
    spatial econometrics. *Regional Science and Urban Economics* 20, 141-163.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> r = sppboot(y, X, W, tid, uid, B=5)
    >>> [round(v, 8) for v in r["rho_draws"]][:2]
    [-0.01760024, 0.06120544]
    """
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    fit = spatial_panel_ml(yv, Xm, Wm, N, model="lag", effects="individual")
    rho, beta = fit["rho"], fit["coefficients"]
    Xt = _cols([_demean(c, N, T, "individual") for c in _cols(Xm)])
    res = fit["residuals"]
    NT = N * T
    m = ssum(res) / NT
    ec = [v - m for v in res]
    Ai = inverse([[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(N)] for i in range(N)])
    mu = [ssum(r[k] * beta[k] for k in range(len(beta))) for r in Xt]
    rd, bd = [], []
    for b in range(int(B)):
        u = random_uniform(NT, seed=seed, stream=b)
        es = [ec[min(NT - 1, int(math.floor(float(v) * NT)))] for v in u]
        ys = []
        for t in range(T):
            v = [mu[t * N + i] + es[t * N + i] for i in range(N)]
            ys += [ssum(Ai[i][j] * v[j] for j in range(N)) for i in range(N)]
        f = spatial_panel_ml(ys, Xt, Wm, N, model="lag", effects="individual")
        rd.append(f["rho"])
        bd.append(list(f["coefficients"]))

    def sd(v):
        mv = ssum(v) / len(v)
        return math.sqrt(ssum((a - mv) ** 2 for a in v) / (len(v) - 1))

    return RichResult(
        payload={
            "rho": rho,
            "coefficients": beta,
            "rho_se": sd(rd),
            "coefficient_se": [sd([d[k] for d in bd]) for k in range(len(beta))],
            "rho_draws": rd,
            "coefficient_draws": bd,
        }
    )


sppboot_fn = sppboot


def cheatsheet() -> str:
    return "sppboot(y, X, W, time_id, unit_id, B=9) -> residual bootstrap SEs of the FE spatial lag panel."
