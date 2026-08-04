# morie.fn -- function file (rootcoder007/morie)
"""Marginal odds ratio by targeted maximum likelihood."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmleor", "tmle_marginal_or"]


def tmleor(Y, A, QAW, Q1W, Q0W, g1W, gbound=0.025, level=0.95):
    """Targeted marginal odds ratio, with inference on the log scale.

    The MARGINAL odds ratio is not the conditional one and does not
    equal the coefficient of a logistic regression: odds ratios are
    non-collapsible, so averaging over W changes the number even with
    no confounding at all.  This estimates the marginal parameter.

    Inference is done on the LOG scale and exponentiated, because the
    sampling distribution of an odds ratio is badly skewed and a
    symmetric interval on the ratio itself can cover negative values.

    Formula: psi = [mu1/(1-mu1)] / [mu0/(1-mu0)];
             IC_logOR = (1/(mu1(1-mu1))) IC_1 - (1/(mu0(1-mu0))) IC_0;
             CI = exp(log psi -+ z se_log)

    Parameters
    ----------
    Y : array-like
        Binary outcome.
    A : array-like
        Binary treatment.
    QAW, Q1W, Q0W : array-like
        Initial outcome predictions.
    g1W : array-like
        Initial propensity.
    gbound : float
        Propensity truncation level.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``log_or``, ``se_log``, ``ci_lower``,
        ``ci_upper``, ``p_value``, ``mu1``, ``mu0``, ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan), whose ``calcParameters``
    sets ``OR$psi <- mu1/(1-mu1)/(mu0/(1-mu0))`` and builds
    ``IC.logOR <- 1/(mu1*(1-mu1)) * (A/g1W*(Y-Q[,"QAW"]) + Q[,"Q1W"])
    - ...`` on the log scale.  Van der Laan's group's own software for
    van der Laan & Rose (2011), Targeted Learning.
    """
    Y = C.vec(Y)
    A = C.vec(A)
    n = len(Y)
    QAW = C.vec(QAW)
    Q1W = C.vec(Q1W)
    Q0W = C.vec(Q0W)
    g1W = C.vec(g1W)
    if any(len(v) != n for v in (A, QAW, Q1W, Q0W, g1W)):
        raise ValueError("every argument must have one entry per observation")
    if any(v not in (0.0, 1.0) for v in A):
        raise ValueError("A must be binary 0/1")
    if any(v < 0.0 or v > 1.0 for v in Y):
        raise ValueError("Y must lie in [0, 1]")
    if n < 2:
        raise ValueError("at least two observations are required")
    fit = T.target(Y, A, QAW, Q1W, Q0W, g1W, gbound)
    mu1, mu0, ic1, ic0 = T.curves(Y, A, fit)
    if not 0.0 < mu1 < 1.0 or not 0.0 < mu0 < 1.0:
        raise ValueError(
            "a targeted mean hit 0 or 1; the odds ratio is undefined")
    psi = (mu1 / (1.0 - mu1)) / (mu0 / (1.0 - mu0))
    ic = [ic1[i] / (mu1 * (1.0 - mu1)) - ic0[i] / (mu0 * (1.0 - mu0))
          for i in range(n)]
    sel = math.sqrt(C.var(ic, 1) / n)
    lp = math.log(psi)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": psi, "log_or": lp, "se_log": sel,
        "ci_lower": math.exp(lp - z * sel),
        "ci_upper": math.exp(lp + z * sel),
        "p_value": 2.0 * (1.0 - C.pnorm(abs(lp / sel))) if sel > 0 else 0.0,
        "mu1": mu1, "mu0": mu0, "n": float(n),
        "method": "TMLE marginal odds ratio, inference on the log scale"})


tmle_marginal_or = tmleor


def cheatsheet():
    return "tmlmro: marginal (not conditional) OR; log-scale CI, non-collapsible"
