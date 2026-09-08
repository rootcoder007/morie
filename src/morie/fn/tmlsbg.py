# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the average treatment effect inside a named subgroup."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_subgroup"]


def tmle_subgroup(y, D, X, subgroup):
    """Targeted ``E[Y(1) - Y(0) | S = 1]`` for a pre-specified subgroup.

    A subgroup effect is not the full-sample effect restricted to a few
    rows: the target changes, so the clever covariate has to change with
    it.  Conditioning on ``S = 1`` divides the efficient influence
    function by ``P(S = 1)`` and zeroes it outside the subgroup, giving

        ``H = I(S = 1)/P(S = 1) * [D/g(X) - (1 - D)/(1 - g(X))]``.

    The nuisance models are still fitted on the whole sample -- that is
    the point of doing it this way rather than subsetting first, since
    the outcome regression borrows information from outside the subgroup
    while the target parameter does not.  ``psi`` is the mean of the
    targeted contrast over the subgroup rows only.  With ``S`` all ones
    this reduces exactly to the full-sample TMLE.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    subgroup : array-like, shape (n,)
        1 for subgroup membership, 0 otherwise.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_sub``, ``n``.

    References
    ----------
    Chernozhukov, V., Demirer, M., Duflo, E. & Fernandez-Val, I. (2025).
    Fisher-Schultz lecture: generic machine learning inference on
    heterogeneous treatment effects in randomized experiments.
    Econometrica 93(4):1121-1164.  doi:10.3982/ECTA19303.  The targeting
    step is van der Laan, M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    Sv = C.vec(subgroup)
    n = len(yv)
    if n == 0 or len(Dv) != n or len(Sv) != n:
        raise ValueError("tmle_subgroup: y, D and subgroup must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_subgroup: X must have one row per subject")
    ps = sum(Sv) / n
    if ps <= 0.0:
        raise ValueError("tmle_subgroup: the subgroup is empty")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    des = [[Dv[i]] + list(W[i]) for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)
    Q1 = [C.dot([1.0] + list(W[i]), qb) for i in range(n)]
    Q0 = [C.dot([0.0] + list(W[i]), qb) for i in range(n)]
    Qobs = [Q1[i] if Dv[i] > 0.5 else Q0[i] for i in range(n)]
    H = [Sv[i] / ps * (Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i])) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps * Sv[i] / (ps * g[i]) for i in range(n)]
    Q0s = [Q0[i] - eps * Sv[i] / (ps * (1.0 - g[i])) for i in range(n)]
    psi = sum(Sv[i] * (Q1s[i] - Q0s[i]) for i in range(n)) / (ps * n)
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Sv[i] / ps * (Q1s[i] - Q0s[i] - psi)
          for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n_sub": float(sum(Sv)), "n": n,
        "method": "TMLE for the average treatment effect within a subgroup"})


def cheatsheet():
    return "tmlsbg: TMLE for a subgroup-conditional treatment effect."

# public names resolved by fn/_lazy_map.json
tmlesubgroup = tmle_subgroup
