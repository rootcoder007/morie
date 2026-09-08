# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the value of the optimal individualized treatment rule."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_individual_regime"]


def tmle_individual_regime(y, D, W, X):
    """Targeted estimate of ``E[Y(d*)]`` for the estimated optimal rule.

    The rule is read off the blip (contrast) function rather than off the
    outcome regression directly, because only the sign of the blip
    matters: ``d*(V) = I(B(V) > 0)`` with ``B(V) = E[Y(1) - Y(0) | V]``.
    Once the rule is fixed the target is an ordinary mean under a known
    deterministic regime, so the clever covariate collapses to

        ``H_i = I(D_i = d*(V_i)) / g_{d*(V_i)}(W_i)``

    and a linear fluctuation ``Q* = Q + eps H`` with
    ``eps = sum H (y - Q) / sum H^2`` solves the efficient score.  The
    plug-in is ``psi = mean_i Q*(d*(V_i), W_i)``, which at the rule
    equals ``mean_i [Q(d_i, W_i) + eps / g_{d_i}(W_i)]``.

    Note that the rule is estimated from the same data; the reported
    standard error is the non-uniform influence-curve SE, which is
    honest only away from the non-uniqueness boundary ``B(V) = 0``
    (Luedtke & van der Laan 2016 give the boundary correction).

    Determinism: the blip regression and the outcome regression are both
    ordinary least squares through the shared modified-Gram-Schmidt
    solver, and the propensity is fixed-iteration IRLS.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    W : array-like, shape (n, p)
        Covariates entering the nuisance models.
    X : array-like, shape (n, q)
        Covariates the rule is allowed to depend on.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_treated``, ``n``.

    References
    ----------
    Luedtke, A. R. & van der Laan, M. J. (2016).  Statistical inference
    for the mean outcome under a possibly non-unique optimal treatment
    strategy.  Annals of Statistics 44(2):713-742.
    doi:10.1214/15-AOS1384.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_individual_regime: y and D must share one length")
    Wm = C.mat(W)
    Vm = C.mat(X)
    if len(Wm) != n or len(Vm) != n:
        raise ValueError("tmle_individual_regime: W and X must have one row per subject")
    Wd = [[1.0] + list(Wm[i]) for i in range(n)]
    gb = S.glmbin(Wd, Dv)
    g = [S.clip(S.expit(C.dot(Wd[i], gb)), 0.025, 0.975) for i in range(n)]
    des = [[Dv[i]] + list(Wd[i]) + [Dv[i] * v for v in Wm[i]] for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)

    def qhat(i, d):
        return C.dot([d] + list(Wd[i]) + [d * v for v in Wm[i]], qb)

    blip = [qhat(i, 1.0) - qhat(i, 0.0) for i in range(n)]
    bd = [[1.0] + list(Vm[i]) for i in range(n)]
    bb, bfit, _, _ = S.ols(bd, blip)
    rule = [1.0 if bfit[i] > 0.0 else 0.0 for i in range(n)]

    gd = [g[i] if rule[i] > 0.5 else 1.0 - g[i] for i in range(n)]
    H = [(1.0 if abs(Dv[i] - rule[i]) < 0.5 else 0.0) / gd[i] for i in range(n)]
    Qd = [qhat(i, rule[i]) for i in range(n)]
    Qobs = [qhat(i, Dv[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Qds = [Qd[i] + eps / gd[i] for i in range(n)]
    psi = sum(Qds) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Qds[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps,
        "n_treated": float(sum(rule)), "n": n,
        "method": "TMLE for the value of the estimated optimal individualized rule"})


def cheatsheet():
    return "tmlitr: TMLE for the value of the optimal individualized rule."
