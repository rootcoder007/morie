# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a direct effect under interference, at a fixed network exposure."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_spillover"]


def _frac_treated(D, A):
    """Default exposure summary: fraction of neighbours treated."""
    n = len(D)
    out = []
    for i in range(n):
        deg = sum(A[i][j] for j in range(n) if j != i)
        s = sum(A[i][j] * D[j] for j in range(n) if j != i)
        out.append(s / deg if deg > 0 else 0.0)
    return out


def tmle_spillover(y, D, X, network, exposure_summary=None):
    """Targeted direct effect holding the neighbourhood exposure fixed.

    Under interference "the" treatment effect is not defined until the
    exposure of a unit's neighbours is pinned down, because a unit's
    outcome depends on the whole treatment vector.  The standard way out
    is an exposure mapping: reduce the neighbourhood to a scalar
    summary ``E_i`` and target the DIRECT effect at a fixed value of it,

        ``psi = E[Y(1, ebar)] - E[Y(0, ebar)]``,

    with ``ebar`` the sample-mean exposure.  The nuisance models both
    condition on ``E``: ``g(X, E)`` for the propensity and ``Q(D, E, X)``
    for the outcome, and the fluctuation is the point-treatment one.

    The variance cannot be the i.i.d. one -- neighbours' influence
    curves are correlated by construction -- so the reported SE is the
    network HAC form of Aronow & Samii, summing ``ic_i ic_j`` over pairs
    that are adjacent or identical.  On an edgeless network that
    collapses back to the i.i.d. variance.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    network : array-like, shape (n, n)
        Adjacency matrix; entry ``(i, j)`` non-zero if ``i`` and ``j``
        are neighbours.
    exposure_summary : callable or None
        ``exposure_summary(D, network) -> vector of length n``.  ``None``
        uses the fraction of neighbours treated.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``se_iid``, ``eps``, ``ebar``, ``n``.

    References
    ----------
    Aronow, P. M. & Samii, C. (2017).  Estimating average causal effects
    under general interference, with application to a social network
    experiment.  Annals of Applied Statistics 11(4):1912-1947.
    doi:10.1214/16-AOAS1005.  The targeting step is Sofrygin, O. & van
    der Laan, M. J. (2017), Semi-parametric estimation and inference for
    the mean outcome of the single time-point intervention in a causally
    connected population, Journal of Causal Inference 5(1).
    doi:10.1515/jci-2016-0003.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_spillover: y and D must share one length")
    Xm = C.mat(X)
    A = C.mat(network)
    if len(Xm) != n:
        raise ValueError("tmle_spillover: X must have one row per subject")
    if len(A) != n or len(A[0]) != n:
        raise ValueError("tmle_spillover: network must be n by n")
    E = _frac_treated(Dv, A) if exposure_summary is None else \
        [float(v) for v in exposure_summary(Dv, A)]
    if len(E) != n:
        raise ValueError("tmle_spillover: exposure_summary must return one value per unit")
    ebar = sum(E) / n
    W = [[1.0] + list(Xm[i]) + [E[i]] for i in range(n)]
    Wb = [[1.0] + list(Xm[i]) + [ebar] for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    qb, _, _, _ = S.ols([[Dv[i]] + list(W[i]) for i in range(n)], yv)
    Qobs = [C.dot([Dv[i]] + list(W[i]), qb) for i in range(n)]
    Q1 = [C.dot([1.0] + list(Wb[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(Wb[i]), qb) for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    ce = [v - m for v in ic]
    var = 0.0
    for i in range(n):
        for j in range(n):
            if i == j or A[i][j] != 0.0:
                var += ce[i] * ce[j]
    var /= float(n * n)
    se = math.sqrt(var) if var > 0.0 else float("nan")
    se_iid = math.sqrt(sum(v * v for v in ce) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "se_iid": se_iid, "eps": eps, "ebar": ebar, "n": n,
        "method": "TMLE for the direct effect under interference at fixed exposure"})


def cheatsheet():
    return "tmlspl: TMLE for a direct effect under network interference."
