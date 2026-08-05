# morie.fn -- function file (rootcoder007/morie)
"""Post-stratification / coverage correction of survey weights."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["coverage_correction"]


def coverage_correction(y, weights, target_totals, strata=None):
    """
    Coverage correction of sampling weights to known population totals

    Formula: w_i' = w_i * (N_target_h / hat N_h)

    Within each post-stratum h the weights are rescaled so that they
    sum exactly to the known population count.  By construction the
    adjusted weights reproduce the control totals, which is the check
    the method is built to satisfy.

    Parameters
    ----------
    y : array-like
        Survey variable, length n.
    weights : array-like
        Design weights before correction, length n.
    target_totals : array-like
        Known population count per post-stratum, in the order the
        stratum labels first appear in ``strata``.
    strata : array-like or None
        Post-stratum label per unit.  None puts everyone in one stratum.

    Returns
    -------
    result : dict
        Keys: estimate (post-stratified mean of y), w_adj, factors,
        total, n.

    References
    ----------
    Sarndal, Swensson & Wretman (1992), Model Assisted Survey Sampling,
    Springer, section 7.6.
    """
    y = core.vec(y)
    w = core.vec(weights)
    tt = core.vec(target_totals)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(w) != n:
        raise ValueError("y and weights must have the same length")
    ids = [0] * n if strata is None else list(strata)
    if len(ids) != n:
        raise ValueError("y and strata must have the same length")
    keys = []
    for k in ids:
        if k not in keys:
            keys.append(k)
    if len(tt) != len(keys):
        raise ValueError("target_totals must have one entry per stratum")
    factors = []
    w_adj = list(w)
    for j, k in enumerate(keys):
        idx = [i for i in range(n) if ids[i] == k]
        nh = sum(w[i] for i in idx)
        if nh <= 0.0:
            raise ValueError("stratum has zero estimated size; cannot correct")
        f = tt[j] / nh
        factors.append(f)
        for i in idx:
            w_adj[i] = w[i] * f
    tot = sum(w_adj)
    est = sum(w_adj[i] * y[i] for i in range(n)) / tot
    return RichResult(payload={
        "estimate": est,
        "w_adj": w_adj,
        "factors": factors,
        "total": tot,
        "n": n,
        "method": "coverage correction of weights to known population totals",
    })


def cheatsheet():
    return "covpop: coverage correction of survey weights to control totals"


# compact alias per ledger/NAMING.md
coveragecorrection = coverage_correction
