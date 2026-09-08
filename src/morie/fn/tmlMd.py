# morie.fn -- function file (rootcoder007/morie)
"""TMLE for natural direct and indirect effects."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_mediation"]


def tmle_mediation(Y, X, M, Cc):
    """Natural direct and indirect effects, both targeted.

    Two nuisance fits and two fluctuations: one for the outcome given
    treatment, mediator and covariates, one for the mediator density
    given treatment and covariates.  Targeting both is what buys the
    double robustness -- a mis-specified mediator model can be rescued
    by a correct outcome model and the other way round, which a
    plug-in product-of-coefficients estimator cannot do.

    Formula: ``NDE = E[Y(1, M_0) - Y(0, M_0)]``,
    ``NIE = E[Y(1, M_1) - Y(1, M_0)]``, targeted with the clever
    covariate ``H = D / g - (1 - D) / (1 - g)``.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Binary treatment.
    M : array-like, shape (n,)
        Mediator.
    Cc : array-like, shape (n, p)
        Baseline covariates.

    Returns
    -------
    RichResult
        ``estimate`` (NDE), ``nie``, ``total``, ``se``, ``eps``, ``n``.

    References
    ----------
    Zheng, W. & van der Laan, M. J. (2012).  Targeted maximum
    likelihood estimation of natural direct effects.  International
    Journal of Biostatistics 8(1):1-40.
    """
    yv = C.vec(Y)
    Dv = C.vec(X)
    Mv = C.vec(M)
    n = len(yv)
    W = C.cbind1(C.mat(Cc))
    ref = [i for i in range(n) if Dv[i] <= 0.5]
    trt = [i for i in range(n) if Dv[i] > 0.5]
    m0b, _, _, _ = S.ols([W[i] for i in ref], [Mv[i] for i in ref])
    m1b, _, _, _ = S.ols([W[i] for i in trt], [Mv[i] for i in trt])
    M0 = [C.dot(W[i], m0b) for i in range(n)]
    M1 = [C.dot(W[i], m1b) for i in range(n)]
    des = [[Dv[i], Mv[i]] + list(W[i]) for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    Q = [C.dot(des[i], qb) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Q[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1M0 = [C.dot([1.0, M0[i]] + list(W[i]), qb) + eps / g[i] for i in range(n)]
    Q0M0 = [C.dot([0.0, M0[i]] + list(W[i]), qb) - eps / (1.0 - g[i]) for i in range(n)]
    Q1M1 = [C.dot([1.0, M1[i]] + list(W[i]), qb) + eps / g[i] for i in range(n)]
    nde = sum(Q1M0[i] - Q0M0[i] for i in range(n)) / n
    nie = sum(Q1M1[i] - Q1M0[i] for i in range(n)) / n
    Qs = [Q[i] + eps * H[i] for i in range(n)]
    ic = [H[i] * (yv[i] - Qs[i]) + Q1M0[i] - Q0M0[i] - nde for i in range(n)]
    mi = sum(ic) / n
    se = math.sqrt(sum((v - mi) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": nde, "nie": nie, "total": nde + nie, "se": se,
        "eps": eps, "n": n,
        "method": "TMLE for natural direct and indirect effects"})


tmlemediation = tmle_mediation


def cheatsheet():
    return "tmlMd: TMLE for natural direct and indirect effects."
