# morie.fn -- function file (rootcoder007/morie)
"""Controlled direct effect by targeted maximum likelihood."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmlecde", "tmle_controlled_direct"]


def tmlecde(Y, A, M, QAM, Q1m, Q0m, g1W, hmW, m=1, gbound=0.025,
            level=0.95):
    """Controlled direct effect: the treatment effect with the mediator held fixed.

    The CDE is not the total effect minus an indirect effect; it is
    the effect of A when M is SET to a fixed value m for everyone, so
    it needs the mediator mechanism as a second nuisance and its
    clever covariate carries BOTH densities in the denominator.  That
    product is why positivity is so much more demanding here than for
    a total effect -- ``min_denominator`` is returned for exactly that
    reason.

    Formula: H_a = 1{A = a, M = m} / (g_a(W) h_m(A, W));
             psi = E[Q*(1, m, W)] - E[Q*(0, m, W)];
             IC = H_1(Y - Q*) + Q*(1,m,W) - mu1
                  - [H_0(Y - Q*) + Q*(0,m,W) - mu0]

    Parameters
    ----------
    Y : array-like
        Outcome in [0, 1].
    A : array-like
        Binary treatment.
    M : array-like
        Mediator level of each observation.
    QAM : array-like
        Initial E[Y | A, M, W] at the observed (A, M).
    Q1m, Q0m : array-like
        Initial E[Y | A = 1, M = m, W] and E[Y | A = 0, M = m, W].
    g1W : array-like
        Initial P(A = 1 | W).
    hmW : array-like
        Initial P(M = m | A, W) at each observation's own A.
    m : float
        The level the mediator is set to.
    gbound : float
        Truncation on both nuisance probabilities.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``mu1``,
        ``mu0``, ``epsilon``, ``min_denominator``, ``max_weight``,
        ``n``.

    References
    ----------
    van der Laan & Petersen (2008), Direct effect models,
    International Journal of Biostatistics 4(1), Article 23 -- the
    row's own citation, which defines the controlled direct effect and
    its targeted estimator.  That paper was NOT obtainable, so the
    estimator is assembled from the components verified in the CRAN
    package ``tmle`` 2.1.1 (Gruber & van der Laan), which was fetched
    and read: the same logistic fluctuation on an offset of the
    initial fit, with the clever covariate's denominator extended from
    g_a(W) to g_a(W) h_m(A, W) as the CDE identification requires.
    """
    Y = C.vec(Y)
    n = len(Y)
    A = C.vec(A)
    M = C.vec(M)
    QAM = C.vec(QAM)
    Q1m = C.vec(Q1m)
    Q0m = C.vec(Q0m)
    g1W = C.vec(g1W)
    hmW = C.vec(hmW)
    if any(len(v) != n for v in (A, M, QAM, Q1m, Q0m, g1W, hmW)):
        raise ValueError("every argument must have one entry per observation")
    if any(v not in (0.0, 1.0) for v in A):
        raise ValueError("A must be binary 0/1")
    if any(v < 0.0 or v > 1.0 for v in Y):
        raise ValueError("Y must lie in [0, 1]")
    if n < 2:
        raise ValueError("at least two observations are required")
    m = float(m)
    g1 = [T.bound(v, gbound, 1.0 - gbound) for v in g1W]
    g0 = [1.0 - v for v in g1]
    h = [T.bound(v, gbound, 1.0) for v in hmW]
    at = [1.0 if M[i] == m else 0.0 for i in range(n)]
    H1 = [at[i] * A[i] / (g1[i] * h[i]) for i in range(n)]
    H0 = [at[i] * (1.0 - A[i]) / (g0[i] * h[i]) for i in range(n)]
    off = [T.logit(v) for v in QAM]
    e = [0.0, 0.0]
    for _ in range(100):
        gr = [0.0, 0.0]
        Hm = [[1e-10, 0.0], [0.0, 1e-10]]
        for i in range(n):
            mu = T.expit(off[i] + e[0] * H0[i] + e[1] * H1[i])
            r = Y[i] - mu
            w = mu * (1.0 - mu)
            gr[0] += H0[i] * r
            gr[1] += H1[i] * r
            Hm[0][0] += H0[i] * H0[i] * w
            Hm[0][1] += H0[i] * H1[i] * w
            Hm[1][0] += H0[i] * H1[i] * w
            Hm[1][1] += H1[i] * H1[i] * w
        st = C.solvev(Hm, gr)
        e = [e[0] + st[0], e[1] + st[1]]
        if max(abs(st[0]), abs(st[1])) < 1e-12:
            break
    QAs = [T.expit(off[i] + e[0] * H0[i] + e[1] * H1[i]) for i in range(n)]
    Q1s = [T.expit(T.logit(Q1m[i]) + e[1] / (g1[i] * h[i]))
           for i in range(n)]
    Q0s = [T.expit(T.logit(Q0m[i]) + e[0] / (g0[i] * h[i]))
           for i in range(n)]
    mu1 = sum(Q1s) / n
    mu0 = sum(Q0s) / n
    ic = [H1[i] * (Y[i] - QAs[i]) + Q1s[i] - mu1
          - (H0[i] * (Y[i] - QAs[i]) + Q0s[i] - mu0) for i in range(n)]
    psi = mu1 - mu0
    se = math.sqrt(C.var(ic, 1) / n)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    den = [min(g1[i], g0[i]) * h[i] for i in range(n)]
    return RichResult(payload={
        "estimate": psi, "se": se, "ci_lower": psi - z * se,
        "ci_upper": psi + z * se, "mu1": mu1, "mu0": mu0,
        "epsilon": e, "min_denominator": min(den),
        "max_weight": max(max(H1), max(H0)), "n": float(n),
        "method": "TMLE controlled direct effect at a fixed mediator level"})


tmle_controlled_direct = tmlecde


def cheatsheet():
    return "tmlcde: H_a = 1{A=a,M=m}/(g_a h_m); positivity needs BOTH densities"
