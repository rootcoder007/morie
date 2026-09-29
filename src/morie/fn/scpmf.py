# morie.fn -- function file (rootcoder007/morie)
"""Spatial Poisson marginal effects: direct, indirect and total impacts of a spatial-lag count model."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .spcount import _mat


def _count_impacts(coef, rho, X, W):
    Xm, Wm = _mat(X), _mat(W)
    b = [float(v) for v in coef]
    n, p = len(Xm), len(b)
    if len(Wm) != n or any(len(r) != n for r in Wm) or any(len(r) != p for r in Xm):
        raise ValueError("X must be n x p with p = len(coef) and W n x n")
    rho = float(rho)
    Ai = inverse([[(1.0 if i == j else 0.0) - rho * Wm[i][j] for j in range(n)] for i in range(n)])
    xb = [ssum(Xm[i][a] * b[a] for a in range(p)) for i in range(n)]
    eta = [ssum(Ai[i][j] * xb[j] for j in range(n)) for i in range(n)]
    mu = [math.exp(v) for v in eta]
    md = ssum(mu[i] * Ai[i][i] for i in range(n)) / n
    mt = ssum(mu[i] * ssum(Ai[i]) for i in range(n)) / n
    direct = [md * bk for bk in b]
    total = [mt * bk for bk in b]
    return RichResult(
        payload={
            "direct": direct,
            "indirect": [t - d for t, d in zip(total, direct)],
            "total": total,
            "fitted": mu,
            "linear_predictor": eta,
        }
    )


def scpmf(coef, rho, X, W):
    r"""Marginal effects (impacts) of the spatial-lag Poisson model ``E y = exp((I - rho W)^{-1} X beta)``.

    The derivative of the mean of unit ``i`` with respect to regressor ``k``
    of unit ``j`` is ``S_k[i, j] = mu_i [(I - rho W)^{-1}]_{ij} beta_k``. As in
    LeSage and Pace (2009, section 2.7) the average direct impact is
    ``tr(S_k) / n``, the average total impact ``1' S_k 1 / n`` and the indirect
    (spillover) impact their difference. ``X`` holds the regressors in the
    columns matching ``coef`` (the intercept column's impact is returned too
    and is usually ignored).

    Parameters
    ----------
    coef : coefficients ``beta`` (length ``p``).
    rho : spatial lag parameter.
    X : ``n x p`` design matrix.
    W : ``n x n`` spatial weights matrix.

    Returns
    -------
    RichResult
        ``direct``, ``indirect``, ``total`` (one per coefficient), ``fitted``
        (``mu``) and ``linear_predictor``.

    References
    ----------
    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, section 2.7.

    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step
    estimator for a spatial lag model of counts. *Regional Science and Urban
    Economics* 40, 241-252.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> r = scpmf([0.2, 0.5], 0.3, [[1, 0.1], [1, 0.4], [1, 0.9]], W)
    >>> [round(v, 10) for v in (r.direct[1], r.indirect[1], r.total[1])]
    [0.9972864709, 0.3400792661, 1.337365737]
    """
    return _count_impacts(coef, rho, X, W)


scpmf_fn = scpmf


def cheatsheet() -> str:
    return "scpmf(coef, rho, X, W) -> direct/indirect/total impacts of the spatial-lag Poisson model."
