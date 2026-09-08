# morie.fn -- function file (rootcoder007/morie)
"""Posterior predictive p-value from replicated test quantities."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ppcrep", "posterior_predictive_replication"]


def ppcrep(t_obs, t_rep):
    """Posterior predictive p-value comparing T(y, theta) to T(y_rep, theta).

    The comparison is made WITHIN each draw: T(y, theta^s) against
    T(y_rep^s, theta^s), the same theta on both sides.  Comparing the
    observed statistic to the whole replicate distribution instead --
    the frequentist reflex -- ignores that the test quantity may depend
    on theta, and gives a different number whenever it does.

    An extreme value in EITHER tail is evidence of misfit, so both the
    one-sided value and min(p, 1-p) doubled are returned; BDA3 warns
    that values below 0.01 or above 0.99 indicate major failures.

    Formula: p_B = Pr( T(y_rep, theta) >= T(y, theta) | y )
                 ~= (1/S) sum_s 1{ T(y_rep^s, theta^s) >= T(y, theta^s) }

    Parameters
    ----------
    t_obs : array-like or float
        T(y, theta^s) for each draw; a scalar if T does not depend on
        theta.
    t_rep : array-like
        T(y_rep^s, theta^s), one per draw.

    Returns
    -------
    RichResult
        ``p_value``, ``p_two_sided``, ``n_extreme``, ``t_obs_mean``,
        ``t_rep_mean``, ``S``, ``extreme`` (1 when outside
        [0.01, 0.99]).

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 6.3, which forms the
    comparison "as a scatterplot of the values T(y, theta^s) vs.
    T(y_rep s, theta^s)" and, under "Interpreting posterior predictive
    p-values", flags tail-area probabilities "less than 0.01 or more
    than 0.99" as major failures.  Fetched as the full text of the book
    from the author's own copy.
    """
    tr = C.vec(t_rep)
    S = len(tr)
    if S < 2:
        raise ValueError("at least two replicates are required")
    to = C.vec(t_obs)
    if len(to) == 1:
        to = to * S
    if len(to) != S:
        raise ValueError("t_obs must be a scalar or have one value per draw")
    k = sum(1 for s in range(S) if tr[s] >= to[s])
    p = k / S
    return RichResult(payload={
        "p_value": p, "p_two_sided": 2.0 * min(p, 1.0 - p),
        "n_extreme": float(k), "t_obs_mean": sum(to) / S,
        "t_rep_mean": sum(tr) / S, "S": float(S),
        "extreme": 1.0 if (p < 0.01 or p > 0.99) else 0.0,
        "method": "Posterior predictive p-value, BDA3 Section 6.3"})


posterior_predictive_replication = ppcrep


def cheatsheet():
    return "posrr: p_B = mean_s 1{T(yrep^s, th^s) >= T(y, th^s)}, paired by draw"
