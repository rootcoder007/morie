# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural mediation by IPTW."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["msm_mediation"]


def msm_mediation(y, A, M, H):
    """Direct and indirect effects from weighted marginal models.

    Weighting rather than conditioning is what lets the mediator model
    stay marginal.  Conditioning on a confounder of the mediator-outcome
    link would also condition on anything the exposure caused; the
    weights remove the confounding while leaving the exposure effect on
    the mediator intact, which is exactly the path being measured.

    Formula: weights ``1 / P(A|H)`` from a logistic propensity, then
    weighted fits ``Y = th0 + th1 a + th2 m + th3 a m`` and
    ``M = b0 + b1 a``, giving ``NDE = th1 + th3 (b0 + b1 a*)`` and
    ``NIE = b1 (th2 + th3 a)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    A : array-like, shape (n,)
        Binary exposure.
    M : array-like, shape (n,)
        Mediator.
    H : array-like, shape (n, p)
        Confounder history used for the weights.

    Returns
    -------
    RichResult
        ``estimate`` (NDE), ``nie``, ``total``, ``theta``, ``beta``,
        ``w_mean``, ``n``.

    References
    ----------
    VanderWeele, T. J. & Vansteelandt, S. (2010).  Odds ratios for
    mediation analysis for a dichotomous outcome.  American Journal of
    Epidemiology 172:1339-1348.  The weighted marginal structural form
    is Robins, J. M., Hernan, M. A. & Brumback, B. (2000), Marginal
    structural models and causal inference in epidemiology,
    Epidemiology 11:550-560.
    """
    yv = C.vec(y)
    Av = C.vec(A)
    Mv = C.vec(M)
    n = len(yv)
    Hm = C.cbind1(C.mat(H))
    gb = S.glmbin(Hm, Av)
    g = [S.clip(S.expit(C.dot(Hm[i], gb)), 0.025, 0.975) for i in range(n)]
    w = [1.0 / g[i] if Av[i] > 0.5 else 1.0 / (1.0 - g[i]) for i in range(n)]
    sw = [math.sqrt(t) for t in w]
    XO = [[sw[i], sw[i] * Av[i], sw[i] * Mv[i], sw[i] * Av[i] * Mv[i]] for i in range(n)]
    theta, _, _, _ = S.ols(XO, [sw[i] * yv[i] for i in range(n)])
    XM = [[sw[i], sw[i] * Av[i]] for i in range(n)]
    beta, _, _, _ = S.ols(XM, [sw[i] * Mv[i] for i in range(n)])
    nde = theta[1] + theta[3] * beta[0]
    nie = beta[1] * (theta[2] + theta[3])
    return RichResult(payload={
        "estimate": nde, "nie": nie, "total": nde + nie, "theta": theta,
        "beta": beta, "w_mean": sum(w) / n, "n": n,
        "method": "Marginal structural mediation by IPTW"})


msmmediation = msm_mediation


def cheatsheet():
    return "medmsm: Marginal structural mediation by IPTW."
