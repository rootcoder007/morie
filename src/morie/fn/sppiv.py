# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel IV/2SLS estimator."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .sppanel import _cols, _demean, _lag
from .sppfe import _stack


def sppiv(y, X, Z, W, time_id, unit_id, effects="individual"):
    r"""Spatial two-stage least squares for the (fixed-effects) spatial lag panel model.

    y_t = rho W y_t + X_t beta + effects + e_t; after the effects
    transformation W y is instrumented by H = [X, WX, W^2 X, Z]
    (Kelejian and Prucha 1998; Mutl and Pfaffermayr 2011), Z holding any
    additional exogenous instruments (None for none). With D = [Wy, X] and
    P_H the projection on H, (rho, beta) = (D'P_H D)^{-1} D'P_H y,
    sigma^2 = e'e / NT and standard errors from sigma^2 (D'P_H
    D)^{-1} (splm::spgm(model = "within", lag = TRUE, spatial.error =
    FALSE)).

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial two-stage
    least squares procedure for estimating a spatial autoregressive model
    with autoregressive disturbances. *J. Real Estate Finance Econ.* 17,
    99-121.
    Mutl, J. and Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord
    panel model. *Econometrics Journal* 14, 48-76.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> round(sppiv(y, X, None, W, tid, uid)["coefficients"][0], 8)
    -0.07889769
    """
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    xc = _cols(Xm)
    if Z is not None:
        _, Zm, _, _ = _stack(y, Z, time_id, unit_id)
        zc = _cols(Zm)
    else:
        zc = []
    if effects == "pooled":
        xc = [[1.0] * (N * T)] + xc
        tr = list
    else:

        def tr(v):
            return _demean(v, N, T, effects)

    yt = tr(yv)
    xt = [tr(c) for c in xc]
    lagged = []
    for c in xc:
        if len(set(c)) == 1:
            continue
        w1 = _lag(Wm, c, N, T)
        lagged += [tr(w1), tr(_lag(Wm, w1, N, T))]
    H = _cols(xt + lagged + [tr(c) for c in zc])
    D = _cols([_lag(Wm, yt, N, T)] + xt)
    n, h, q = N * T, len(H[0]), len(D[0])
    HtHi = inverse([[ssum(r[a] * r[b] for r in H) for b in range(h)] for a in range(h)])
    HtD = [[ssum(H[i][a] * D[i][b] for i in range(n)) for b in range(q)] for a in range(h)]
    Hty = [ssum(H[i][a] * yt[i] for i in range(n)) for a in range(h)]
    F = [[ssum(HtD[c][a] * HtHi[c][b] for c in range(h)) for b in range(h)] for a in range(q)]
    A = inverse([[ssum(F[a][c] * HtD[c][b] for c in range(h)) for b in range(q)] for a in range(q)])
    rhs = [ssum(F[a][c] * Hty[c] for c in range(h)) for a in range(q)]
    coef = [ssum(A[a][b] * rhs[b] for b in range(q)) for a in range(q)]
    res = [yt[i] - ssum(D[i][a] * coef[a] for a in range(q)) for i in range(n)]
    s2 = ssum(e * e for e in res) / n
    return RichResult(
        payload={
            "coefficients": coef,
            "rho": coef[0],
            "se": [math.sqrt(s2 * A[a][a]) for a in range(q)],
            "sigma2": s2,
            "residuals": res,
        }
    )


sppiv_fn = sppiv


def cheatsheet() -> str:
    return "sppiv(y, X, Z, W, time_id, unit_id) -> spatial 2SLS of the FE spatial lag panel, H = [X, WX, W^2X, Z]."
