# morie.fn -- function file (rootcoder007/morie)
"""Adaptively-weighted AIPW for data collected by a bandit.

Source opened: Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S. and
Athey, S. (2021).  Confidence intervals for policy evaluation in
adaptive experiments.  *PNAS* 118(15), e2014602118; working paper
arXiv:1911.02768, page 8, equation (12).  When the assignment
probabilities e_t are chosen adaptively, the ordinary AIPW average is
not asymptotically normal because sum_t 1/e_t need not concentrate.  The
paper fixes the denominator by construction with the stick-breaking
recursion

    h_t^2 / e_t = ( 1 - sum_{s < t} h_s^2 / e_s ) lambda_t,
    0 <= lambda_t < 1 for t < T,  lambda_T = 1,

so that sum_t h_t^2 / e_t = 1 exactly, at every sample size and not
merely asymptotically.  The allocation rate used here is
lambda_t = 1/(T - t + 1), the smallest rate permitted by the lower bound
of the paper's Theorem 3, condition (14).

The score is the standard AIPW contrast

    psi_t = m1_t - m0_t + D_t (y_t - m1_t)/e_t
                        - (1 - D_t)(y_t - m0_t)/(1 - e_t),

with m1, m0 from a least-squares outcome regression per arm, and the
estimator is the h-weighted mean sum h_t psi_t / sum h_t.  A constant
assignment probability makes every h_t equal, at which point the
estimator collapses to the unweighted AIPW mean -- the degenerate check
this module is anchored on.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_bandit_did"]


def dr_bandit_did(y, D_t, X=None, pi_t=None):
    """Adaptively-weighted AIPW effect under a known assignment sequence.

    Parameters
    ----------
    y : array-like
        Outcome, one entry per time step, in assignment order.
    D_t : array-like
        Binary assignment actually made at each step.
    X : 2-D array-like, optional
        Covariates for the per-arm outcome regressions.
    pi_t : array-like, optional
        Known assignment probability e_t at each step, strictly inside
        (0, 1).  ``None`` uses the realised treated share, which is the
        non-adaptive case.

    Returns
    -------
    result : dict
        Keys: estimate, se, aipw_unweighted, h, sum_h2_over_e, n_treat,
        n.

    References
    ----------
    Hadad, Hirshberg, Zhan, Wager & Athey (2021), PNAS 118(15):
    e2014602118, doi:10.1073/pnas.2014602118; arXiv:1911.02768 eq. (12).
    """
    yv = k.vec(y)
    dv = k.vec(D_t)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(dv) != n:
        raise ValueError("y and D_t must have the same length")
    s = sum(dv)
    if s <= 0.0 or s >= float(n):
        raise ValueError("D_t must contain both arms")
    if pi_t is None:
        e = [s / float(n)] * n
    else:
        e = k.vec(pi_t)
        if len(e) != n:
            raise ValueError("pi_t must have the same length as y")
    for v in e:
        if not (0.0 < v < 1.0):
            raise ValueError("pi_t must lie strictly inside (0, 1)")
    Z = k.design(X, n)
    i1 = [i for i in range(n) if dv[i] >= 0.5]
    i0 = [i for i in range(n) if dv[i] < 0.5]
    b1 = k.lstsq([Z[i] for i in i1], [yv[i] for i in i1])
    b0 = k.lstsq([Z[i] for i in i0], [yv[i] for i in i0])
    m1 = k.matvec(Z, b1)
    m0 = k.matvec(Z, b0)
    psi = []
    for i in range(n):
        psi.append(m1[i] - m0[i]
                   + dv[i] * (yv[i] - m1[i]) / e[i]
                   - (1.0 - dv[i]) * (yv[i] - m0[i]) / (1.0 - e[i]))
    h = []
    acc = 0.0
    for i in range(n):
        lam = 1.0 / float(n - i)
        q = (1.0 - acc) * lam
        if q < 0.0:
            q = 0.0
        acc += q
        h.append((q * e[i]) ** 0.5)
    sh = sum(h)
    if sh <= 0.0:
        raise ValueError("degenerate weights: every h_t is zero")
    est = 0.0
    for i in range(n):
        est += h[i] * psi[i]
    est = est / sh
    v = 0.0
    for i in range(n):
        v += (h[i] * (psi[i] - est)) ** 2
    return RichResult(
        title="Adaptively-weighted AIPW",
        summary_lines=[("sum h^2/e", acc)],
        payload={
            "estimate": est,
            "se": (v ** 0.5) / sh,
            "aipw_unweighted": k.mean(psi),
            "h": h,
            "sum_h2_over_e": acc,
            "n_treat": s,
            "n": n,
            "method": "DR for adaptive bandit-DiD",
        },
    )


def cheatsheet():
    return "drbnk: DR for adaptive bandit-DiD"


# compact alias per ledger/NAMING.md
drbanditdid = dr_bandit_did
