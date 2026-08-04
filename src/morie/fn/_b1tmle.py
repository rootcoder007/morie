# morie.fn -- function file (rootcoder007/morie)
"""Shared targeted-maximum-likelihood machinery for the big1/s05 batch.

Internal only.  One targeting step and one influence curve, used by
every tmle* module in this batch rather than copied into each.  Mirrors
``aaa_b1_tmle.R`` on the R side.
"""

import math

from . import _tail1core as C

__all__ = []

_EPS = 1e-12


def bound(v, lo, hi):
    return min(hi, max(lo, v))


def logit(p):
    p = bound(p, _EPS, 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def expit(x):
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def target(Y, A, QAW, Q1W, Q0W, g1W, gbound=0.025, iters=100, tol=1e-12):
    """One TMLE fluctuation step; returns the targeted predictions.

    The fluctuation is a two-covariate logistic regression of Y on the
    clever covariates H0W = (1-A)/g0W and H1W = A/g1W with NO intercept
    and the initial fit as an offset on the logit scale -- exactly the
    ``glm(Ystar ~ -1 + offset(qlogis(Q)) + H0W + H1W, family = binomial)``
    of van der Laan's own ``tmle`` package.

    Newton-Raphson with a FIXED iteration cap, so both language arms
    land on the same iterate.
    """
    n = len(Y)
    g1 = [bound(v, gbound, 1.0 - gbound) for v in g1W]
    g0 = [1.0 - v for v in g1]
    H1 = [A[i] / g1[i] for i in range(n)]
    H0 = [(1.0 - A[i]) / g0[i] for i in range(n)]
    off = [logit(v) for v in QAW]
    e = [0.0, 0.0]
    for _ in range(int(iters)):
        gr = [0.0, 0.0]
        Hm = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(n):
            eta = off[i] + e[0] * H0[i] + e[1] * H1[i]
            mu = expit(eta)
            r = Y[i] - mu
            w = mu * (1.0 - mu)
            gr[0] += H0[i] * r
            gr[1] += H1[i] * r
            Hm[0][0] += H0[i] * H0[i] * w
            Hm[0][1] += H0[i] * H1[i] * w
            Hm[1][0] += H0[i] * H1[i] * w
            Hm[1][1] += H1[i] * H1[i] * w
        # Ridge the information by a hair: with a rare treatment one of
        # the two clever covariates can be identically zero, and the
        # information is then exactly singular.
        Hm[0][0] += 1e-10
        Hm[1][1] += 1e-10
        st = C.solvev(Hm, gr)
        e = [e[0] + st[0], e[1] + st[1]]
        if max(abs(st[0]), abs(st[1])) < tol:
            break
    QAs = [expit(off[i] + e[0] * H0[i] + e[1] * H1[i]) for i in range(n)]
    Q1s = [expit(logit(Q1W[i]) + e[1] / g1[i]) for i in range(n)]
    Q0s = [expit(logit(Q0W[i]) + e[0] / g0[i]) for i in range(n)]
    return {"epsilon": e, "QAstar": QAs, "Q1star": Q1s, "Q0star": Q0s,
            "g1": g1, "g0": g0, "H1": H1, "H0": H0}


def curves(Y, A, fit):
    """Efficient influence curves of E[Y_1] and E[Y_0] after targeting."""
    n = len(Y)
    Q1s = fit["Q1star"]
    Q0s = fit["Q0star"]
    QAs = fit["QAstar"]
    g1 = fit["g1"]
    g0 = fit["g0"]
    mu1 = sum(Q1s) / n
    mu0 = sum(Q0s) / n
    ic1 = [A[i] / g1[i] * (Y[i] - QAs[i]) + Q1s[i] - mu1 for i in range(n)]
    ic0 = [(1.0 - A[i]) / g0[i] * (Y[i] - QAs[i]) + Q0s[i] - mu0
           for i in range(n)]
    return mu1, mu0, ic1, ic0
