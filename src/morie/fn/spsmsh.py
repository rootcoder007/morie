# morie.fn -- function file (rootcoder007/morie)
"""Shifted (stochastic) intervention parameter for a continuous exposure."""

from math import exp, fsum, pi, sqrt

from ._richresult import RichResult
from ._spx import dot, lstsq, mat, vec

__all__ = [
    "spsm_shifted_intervention",
    "shiftint",
]


def spsm_shifted_intervention(y, a, h, delta=1.0, trim=None):
    """IPW estimate of ``E[Y(A + delta)]`` under a shifted intervention.

    NOT IN SCHABENBERGER & GOTWAY -- this is causal inference. The
    estimand and the density-ratio weight are Diaz, I. & van der Laan,
    M. J. (2012), "Population intervention causal effects based on
    stochastic interventions", *Biometrics* 68:541-549, and Diaz &
    van der Laan (2018), Ch. 14 of *Targeted Learning in Data Science*,
    Springer -- named from the general literature and NOT verified
    against a PDF in this corpus.

    A shifted intervention does not set the exposure to a fixed value; it
    moves everyone by delta, d(a, h) = a + delta. That is the point: for a
    continuous exposure the fixed-value estimand E[Y(a)] needs positivity
    at every a, which real data never have, while the shifted estimand
    needs only that a + delta stays in the support.

    For an additive shift the identified functional is

        psi = E[ {g(A - delta | H) / g(A | H)} Y ],

    the density ratio evaluated at the BACK-shifted exposure. Back-shifted,
    not forward-shifted: the change of variables in the derivation runs
    the other way, and using g(A + delta|H)/g(A|H) is the sign error this
    estimand invites.

    g is a Gaussian working model, A | H ~ N(H'gamma, tau^2) fitted by
    OLS with an intercept, so the weight collapses to

        w = exp{ (delta / tau^2) (A - H'gamma - delta/2) }.

    This is a WORKING MODEL, not a guarantee; psi is consistent only if it
    is right (or if a doubly robust outcome model is added, which this
    function does not do). ``max_weight`` is returned because a large
    maximum weight means one observation is carrying the estimate, and
    `trim` caps the weights when it does.

    Parameters
    ----------
    y : (n,) array-like
        Outcome.
    a : (n,) array-like
        Continuous exposure.
    h : (n, k) array-like or None
        Covariates WITHOUT an intercept column.
    delta : float
        Additive shift.
    trim : float, optional
        Upper cap on the weights.

    Returns
    -------
    RichResult
        ``psi``, ``naive_mean``, ``weights``, ``max_weight``, ``tau2``,
        ``gamma``, ``delta``, ``n``, ``method``.
    """
    yv = vec(y, "y")
    av = vec(a, "a")
    n = len(yv)
    if len(av) != n:
        raise ValueError("`y` and `a` must have the same length")
    if n < 4:
        raise ValueError("at least 4 observations are needed")
    if h is None:
        hm = [[] for _ in range(n)]
    else:
        hm = mat(h, "h")
        if len(hm) != n:
            raise ValueError("`h` has %d rows but `y` has %d values"
                             % (len(hm), n))
    d = float(delta)
    des = [[1.0] + list(hm[i]) for i in range(n)]
    k = len(des[0])
    if n <= k:
        raise ValueError("need more observations than covariates + 1")
    gam = lstsq(des, av)
    res = [av[i] - dot(des[i], gam) for i in range(n)]
    tau2 = fsum([t * t for t in res]) / (n - k)
    if tau2 <= 0:
        raise ValueError("the exposure is perfectly predicted by `h`; "
                         "no shift is identified")

    w = [exp((d / tau2) * (res[i] - 0.5 * d)) for i in range(n)]
    if trim is not None:
        cap = float(trim)
        if cap <= 0:
            raise ValueError("`trim` must be positive")
        w = [min(t, cap) for t in w]
    psi = fsum([w[i] * yv[i] for i in range(n)]) / n

    return RichResult(payload={
        "psi": psi,
        "naive_mean": fsum(yv) / n,
        "weights": w,
        "max_weight": max(w),
        "mean_weight": fsum(w) / n,
        "tau2": tau2,
        "gamma": gam,
        "delta": d,
        "weight_uses_back_shifted_density": True,
        "gaussian_working_model": True,
        "n": n,
        "method": ("Shifted-intervention IPW psi = E[g(A-delta|H)/g(A|H) Y] "
                   "with a Gaussian exposure model (Diaz & van der Laan "
                   "2012, 2018); NOT in Schabenberger & Gotway"),
    })


def cheatsheet():
    return "spsmsh: shifted-intervention causal parameter"


# compact alias per ledger/NAMING.md
shiftint = spsm_shifted_intervention
