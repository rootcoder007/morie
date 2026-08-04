# morie.fn -- slice s03 (rootcoder007/morie)
"""TRPO's constrained surrogate and its step size.

Source consulted (FETCHED): Schulman, J., Levine, S., Moritz, P.,
Jordan, M. and Abbeel, P. (2015).  Trust region policy optimization.
*ICML* 37, 1889-1897 (arXiv:1502.05477).  The practical problem the
paper solves, its equation (14), is

    maximize_theta   E[ pi_theta(a|s) / pi_theta_old(a|s) * A ]
    subject to       Dbar_KL^(rho_old)(theta_old, theta) <= delta

-- note that this is the *average* KL over the state distribution,
which the paper substitutes for the maximum KL of its equation (12)
because "the max ... is impractical to solve".  Section 6 then solves
the quadratic approximation in closed form: with g the gradient of the
surrogate and F the Fisher information (the Hessian of the KL), the
step is

    s = sqrt( 2 delta / (x' F x) ) x,   x = F^(-1) g

which is the largest step in the direction x that stays inside the
trust region.  Both the objective and, when g and F are given, that
step length are returned.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["trpo"]


def trpo(env, policy=None, kl_max=0.01, ratio=None, adv=None, kl=None,
         g=None, F=None):
    """Surrogate value, KL, and the trust-region step length.

    Parameters
    ----------
    env : array-like
        The advantages A_t.
    policy : array-like, optional
        The probability ratios r_t.
    kl_max : float
        The trust-region radius delta.
    ratio, adv, kl : array-like, optional
        Explicit ratios, advantages, and per-state KL divergences.
    g : array-like, optional
        Gradient of the surrogate.
    F : 2-D array-like, optional
        Fisher information matrix.

    Returns
    -------
    RichResult with payload:
        estimate   : the surrogate objective
        kl_mean    : mean KL over states
        feasible   : whether the KL constraint holds
        step       : the natural-gradient step (empty without g and F)
        step_size  : sqrt(2 delta / x' F x)
    """
    a = k.vec(adv if adv is not None else env)
    r = k.vec(ratio if ratio is not None else policy)
    n = len(a)
    s = 0.0
    for i in range(n):
        s += r[i] * a[i]
    surr = s / n if n else float("nan")
    klm = k.mean(k.vec(kl)) if kl is not None else float("nan")
    step = []
    ss = float("nan")
    if g is not None and F is not None:
        gv = k.vec(g)
        Fm = k.mat(F)
        x = k.ridgesolve(Fm, gv, 1e-10)
        Fx = k.matvec(Fm, x)
        q = 0.0
        for i in range(len(x)):
            q += x[i] * Fx[i]
        if q > 0.0:
            ss = math.sqrt(2.0 * float(kl_max) / q)
            step = [ss * v for v in x]
    return RichResult(
        title="TRPO constrained surrogate",
        summary_lines=[("surrogate", surr), ("mean KL", klm)],
        payload={
            "estimate": surr,
            "surrogate": surr,
            "kl_mean": klm,
            "feasible": (klm <= float(kl_max)) if klm == klm else True,
            "step": step,
            "step_size": ss,
            "delta": float(kl_max),
            "n": n,
            "method": "TRPO surrogate under a mean-KL trust region (eq. 14; step from sec. 6)",
        },
    )


def cheatsheet():
    return "trpoc: Trust region policy optimization"
