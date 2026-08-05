# morie.fn -- function file (rootcoder007/morie)
"""RCT-assisted TMLE: borrow observational strength for a trial-population ATE."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_rct_assisted"]


def tmle_rct_assisted(y_rct, y_obs, D, X):
    """Targeted trial-population ATE with an observational arm for ``Q`` only.

    The asymmetry is deliberate.  The observational rows are allowed to
    sharpen the outcome regression, because a wrong ``Q`` is repaired by
    the targeting step; they are NOT allowed into the clever covariate,
    because their treatment mechanism is unknown and a wrong ``g`` in a
    randomised trial would throw away the one thing the trial
    guarantees.  So ``g`` is the trial's own empirical randomisation
    probability, constant across trial rows, and the clever covariate is

        ``H = I(trial)/P(trial) * [D/g - (1 - D)/(1 - g)]``,

    which makes the target the ATE in the TRIAL population.  If the
    observational rows are biased the estimator is still consistent; it
    only loses the efficiency the borrowing was meant to buy.

    Rows are stacked trial-first: ``D`` and ``X`` must have
    ``len(y_rct) + len(y_obs)`` rows in that order.

    Parameters
    ----------
    y_rct : array-like, shape (n1,)
        Trial outcomes.
    y_obs : array-like, shape (n2,)
        Observational outcomes.
    D : array-like, shape (n1 + n2,)
        Binary treatment, trial rows first.
    X : array-like, shape (n1 + n2, p)
        Covariates, trial rows first.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``g_rct``, ``n_rct``, ``n_obs``,
        ``n``.

    References
    ----------
    Athey, S., Chetty, R., Imbens, G. W. & Kang, H. (2025).  The
    surrogate index: combining short-term proxies to estimate long-term
    treatment effects more rapidly and precisely.  Review of Economic
    Studies 93(4):2284-2312.  doi:10.1093/restud/rdaf087.  The targeting
    step is van der Laan, M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    y1 = C.vec(y_rct)
    y2 = C.vec(y_obs)
    yv = y1 + y2
    Dv = C.vec(D)
    n1 = len(y1)
    n = len(yv)
    if n1 < 2:
        raise ValueError("tmle_rct_assisted: need at least two trial rows")
    if len(Dv) != n:
        raise ValueError("tmle_rct_assisted: D must have one entry per stacked row")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_rct_assisted: X must have one row per stacked row")
    Sv = [1.0 if i < n1 else 0.0 for i in range(n)]
    W = [[1.0] + list(Xm[i]) + [Sv[i]] for i in range(n)]
    g0 = sum(Dv[i] for i in range(n1)) / n1
    g0 = S.clip(g0, 0.025, 0.975)
    qb, _, _, _ = S.ols([[Dv[i]] + list(W[i]) for i in range(n)], yv)
    Q1 = [C.dot([1.0] + list(W[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]), qb) for i in range(n)]
    Qobs = [Q1[i] if Dv[i] > 0.5 else Q0[i] for i in range(n)]
    pt = n1 / float(n)
    H = [Sv[i] / pt * (Dv[i] / g0 - (1.0 - Dv[i]) / (1.0 - g0)) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps * Sv[i] / (pt * g0) for i in range(n)]
    Q0s = [Q0[i] - eps * Sv[i] / (pt * (1.0 - g0)) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n1)) / n1
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Sv[i] / pt * (Q1s[i] - Q0s[i] - psi)
          for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "g_rct": g0,
        "n_rct": float(n1), "n_obs": float(n - n1), "n": n,
        "method": "RCT-assisted TMLE for the trial-population ATE"})


def cheatsheet():
    return "tmlrct: RCT-assisted TMLE borrowing observational strength."
