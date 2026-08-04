# morie.fn -- function file (rootcoder007/morie)
"""Marginal risk difference by targeted maximum likelihood."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmlerd", "tmle_marginal_rd"]


def tmlerd(Y, A, QAW, Q1W, Q0W, g1W, gbound=0.025, level=0.95):
    """Targeted estimate of the marginal risk difference E[Y_1] - E[Y_0].

    The reason to target rather than simply average the initial
    predictions is DOUBLE ROBUSTNESS with valid inference: the plug-in
    mean of a machine-learning fit is biased at a rate the fit's own
    standard errors know nothing about, whereas the targeted estimate
    solves the efficient score equation and its influence curve gives
    an honest standard error.  ``psi_init`` is returned alongside so
    the size of that correction is visible.

    Formula: fluctuate on H1 = A/g1(W), H0 = (1-A)/g0(W);
             psi = mean(Q*(1,W)) - mean(Q*(0,W));
             IC = (A/g1)(Y - Q*(A,W)) + Q*(1,W) - mu1
                  - [((1-A)/g0)(Y - Q*(A,W)) + Q*(0,W) - mu0];
             se = sqrt(var(IC)/n)

    Parameters
    ----------
    Y : array-like
        Outcome in [0, 1].
    A : array-like
        Binary treatment in {0, 1}.
    QAW, Q1W, Q0W : array-like
        Initial predictions of E[Y | A, W], E[Y | 1, W], E[Y | 0, W].
    g1W : array-like
        Initial propensity P(A = 1 | W).
    gbound : float
        Propensity truncation level.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``p_value``,
        ``mu1``, ``mu0``, ``psi_init``, ``epsilon``, ``ic_mean``,
        ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan), whose fluctuation is
    ``glm(Ystar ~ -1 + offset(qlogis(Q[,"QAW"])) + H0W + H1W,
    family = binomial)`` with H1W = A/g1W and H0W = (1-A)/g0W, and
    whose ``calcParameters`` forms
    ``IC.EY1 <- Delta/pDelta1*(Y-Q[,"QAW"]) + Q[,"Q1W"] - mu1`` with
    ``var.psi <- var(IC)/n``.  That package is van der Laan's group's
    own software for van der Laan & Rose (2011), Targeted Learning.
    """
    Y = C.vec(Y)
    A = C.vec(A)
    n = len(Y)
    for v, nm in ((A, "A"), (QAW, "QAW"), (Q1W, "Q1W"), (Q0W, "Q0W"),
                  (g1W, "g1W")):
        if len(C.vec(v)) != n:
            raise ValueError("%s must have one entry per observation" % nm)
    if any(v not in (0.0, 1.0) for v in A):
        raise ValueError("A must be binary 0/1")
    if any(v < 0.0 or v > 1.0 for v in Y):
        raise ValueError("Y must lie in [0, 1]")
    if n < 2:
        raise ValueError("at least two observations are required")
    QAW = C.vec(QAW)
    Q1W = C.vec(Q1W)
    Q0W = C.vec(Q0W)
    g1W = C.vec(g1W)
    fit = T.target(Y, A, QAW, Q1W, Q0W, g1W, gbound)
    mu1, mu0, ic1, ic0 = T.curves(Y, A, fit)
    ic = [ic1[i] - ic0[i] for i in range(n)]
    psi = mu1 - mu0
    se = math.sqrt(C.var(ic, 1) / n)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    init = sum(Q1W) / n - sum(Q0W) / n
    return RichResult(payload={
        "estimate": psi, "se": se, "ci_lower": psi - z * se,
        "ci_upper": psi + z * se,
        "p_value": 2.0 * (1.0 - C.pnorm(abs(psi / se))) if se > 0 else 0.0,
        "mu1": mu1, "mu0": mu0, "psi_init": init,
        "epsilon": fit["epsilon"], "ic_mean": sum(ic) / n, "n": float(n),
        "method": "TMLE marginal risk difference"})


tmle_marginal_rd = tmlerd


def cheatsheet():
    return "tmlmrd: psi = mean Q*(1,W) - mean Q*(0,W); se from the influence curve"
