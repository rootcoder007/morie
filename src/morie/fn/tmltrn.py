# morie.fn -- function file (rootcoder007/morie)
"""TMLE that transports a treatment effect from a source to a target population."""

import math

from . import _s04core as S4
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_transportability"]


def tmle_transportability(y, D, X, S):
    """Targeted ``E[Y(1) - Y(0) | S = 0]`` from outcomes seen only when ``S = 1``.

    Transport is an extrapolation in the covariate distribution, not in
    the outcome model: the effect is identified in the target population
    only under S-admissibility, i.e. the conditional effect given ``X``
    is the same in both populations and the target's covariate support
    is contained in the source's.  What changes relative to a plain
    subgroup analysis is the weight -- the source rows have to be
    reweighted by the SAMPLING ODDS

        ``H = I(S = 1)/P(S = 0) * (1 - p(X))/p(X) *
              [D/g(X) - (1 - D)/(1 - g(X))]``,

    with ``p(X) = P(S = 1 | X)``.  Outcomes in the target population are
    never used and may be any placeholder; only its covariates enter.
    ``psi`` is the mean of the targeted contrast over the target rows.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome; entries with ``S = 0`` are ignored.
    D : array-like, shape (n,)
        Binary treatment; entries with ``S = 0`` are ignored.
    X : array-like, shape (n, p)
        Covariates, in both populations.
    S : array-like, shape (n,)
        1 for a source (trial) row, 0 for a target-population row.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_source``, ``n_target``, ``n``.

    References
    ----------
    Rudolph, K. E. & van der Laan, M. J. (2017).  Robust estimation of
    encouragement design intervention effects transported across sites.
    Journal of the Royal Statistical Society Series B 79(5):1509-1525.
    doi:10.1111/rssb.12213.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    Sv = C.vec(S)
    n = len(yv)
    if n == 0 or len(Dv) != n or len(Sv) != n:
        raise ValueError("tmle_transportability: y, D and S must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_transportability: X must have one row per subject")
    src = [i for i in range(n) if Sv[i] > 0.5]
    tgt = [i for i in range(n) if Sv[i] <= 0.5]
    if len(src) < 2 or len(tgt) < 1:
        raise ValueError("tmle_transportability: need at least two source and one target row")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    pb = S4.glmbin(W, Sv)
    p = [S4.clip(S4.expit(C.dot(W[i], pb)), 0.025, 0.975) for i in range(n)]
    gb = S4.glmbin([W[i] for i in src], [Dv[i] for i in src])
    g = [S4.clip(S4.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    qb, _, _, _ = S4.ols([[Dv[i]] + list(W[i]) for i in src], [yv[i] for i in src])
    Q1 = [C.dot([1.0] + list(W[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]), qb) for i in range(n)]
    Qobs = [Q1[i] if Dv[i] > 0.5 else Q0[i] for i in range(n)]
    pt = len(tgt) / float(n)
    odds = [(1.0 - p[i]) / p[i] for i in range(n)]
    H = [Sv[i] / pt * odds[i] * (Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]))
         for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n) if Sv[i] > 0.5) / den \
        if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps * odds[i] / (pt * g[i]) for i in range(n)]
    Q0s = [Q0[i] - eps * odds[i] / (pt * (1.0 - g[i])) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in tgt) / len(tgt)
    ic = []
    for i in range(n):
        r = (yv[i] - Qobs[i] - eps * H[i]) if Sv[i] > 0.5 else 0.0
        ic.append(H[i] * r + (1.0 - Sv[i]) / pt * (Q1[i] - Q0[i] - psi))
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps,
        "n_source": float(len(src)), "n_target": float(len(tgt)), "n": n,
        "method": "TMLE transporting a treatment effect to a target population"})


def cheatsheet():
    return "tmltrn: TMLE transporting an effect across populations."
