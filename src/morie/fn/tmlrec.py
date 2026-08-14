# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the marginal rate ratio of a recurrent event."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_recurrent"]


def tmle_recurrent(time, event, D, X):
    """Targeted marginal rate ratio for recurrent events.

    With recurrent events the natural marginal summary is the RATE, not
    a hazard ratio: the mean function of the counting process is
    identified without any assumption about the dependence between a
    subject's successive events, which is exactly what makes the
    Lin-Wei-Yang-Ying rate model usable when the within-subject
    dependence is unknown.  Each subject contributes a rate
    ``N_i / T_i``; the outcome regression is on that rate, the clever
    covariate is the point-treatment one
    ``H = D/g - (1 - D)/(1 - g)``, and

        ``mu_a = mean_i Q*(a, X_i)``,  ``estimate = mu_1 / mu_0``.

    The standard error is the delta-method combination of the two arms'
    influence curves, so it is a ratio SE and not a difference SE.

    Parameters
    ----------
    time : array-like, shape (n,)
        Follow-up duration of each subject; must be positive.
    event : array-like, shape (n,)
        Number of events observed for each subject.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Baseline covariates.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``mu1``, ``mu0``, ``eps``, ``n``.

    References
    ----------
    Lin, D. Y., Wei, L. J., Yang, I. & Ying, Z. (2000).  Semiparametric
    regression for the mean and rate functions of recurrent events.
    Journal of the Royal Statistical Society Series B 62(4):711-730.
    doi:10.1111/1467-9868.00259.  The targeting step is van der Laan,
    M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    tv = C.vec(time)
    ev = C.vec(event)
    Dv = C.vec(D)
    n = len(tv)
    if n == 0 or len(ev) != n or len(Dv) != n:
        raise ValueError("tmle_recurrent: time, event and D must share one length")
    if any(v <= 0.0 for v in tv):
        raise ValueError("tmle_recurrent: follow-up time must be positive")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_recurrent: X must have one row per subject")
    rate = [ev[i] / tv[i] for i in range(n)]
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    qb, _, _, _ = S.ols([[Dv[i]] + list(W[i]) for i in range(n)], rate)
    Q1 = [C.dot([1.0] + list(W[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]), qb) for i in range(n)]
    Qobs = [Q1[i] if Dv[i] > 0.5 else Q0[i] for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (rate[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    mu1 = sum(Q1s) / n
    mu0 = sum(Q0s) / n
    if mu0 == 0.0:
        raise ValueError("tmle_recurrent: the control-arm rate is zero; no rate ratio")
    ic1 = [Dv[i] / g[i] * (rate[i] - Qobs[i] - eps * H[i]) + Q1s[i] - mu1 for i in range(n)]
    ic0 = [(1.0 - Dv[i]) / (1.0 - g[i]) * (rate[i] - Qobs[i] - eps * H[i]) + Q0s[i] - mu0
           for i in range(n)]
    ic = [ic1[i] / mu0 - mu1 * ic0[i] / (mu0 * mu0) for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": mu1 / mu0, "se": se, "mu1": mu1, "mu0": mu0, "eps": eps, "n": n,
        "method": "TMLE for the marginal recurrent-event rate ratio"})


def cheatsheet():
    return "tmlrec: TMLE for the marginal recurrent-event rate ratio."

# public names resolved by fn/_lazy_map.json
tmlerecurrent = tmle_recurrent
