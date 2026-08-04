# morie.fn -- function file (rootcoder007/morie)
"""Monotone instrumental variable bounds."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mivbound", "bound_monot_inst"]


def mivbound(lower, upper, prob):
    """Intersect per-instrument bounds under a monotone instrument.

    A monotone instrumental variable is one that shifts the mean
    counterfactual weakly in a known direction rather than not at all:

        v'' >= v'  implies  E[Y(t) | V = v''] >= E[Y(t) | V = v'].

    That turns the collection of per-value bounds into a single sharper
    one, because a low value of V can borrow the largest lower bound from
    below it and the smallest upper bound from above it:

        LB(t | V = v) = max_{u <= v} LB(t | V = u)
        UB(t | V = v) = min_{u >= v} UB(t | V = u)
        E[Y(t)] in [ sum_v P(V=v) LB(t|v),  sum_v P(V=v) UB(t|v) ]

    with the instrument values supplied already in increasing order.

    Parameters
    ----------
    lower, upper : array-like
        Per-value bounds LB(t|V=v), UB(t|V=v) in increasing order of v.
    prob : array-like
        P(V = v), same order; normalised internally.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``lowerv``, ``upperv``,
        ``prob``, ``k``.

    References
    ----------
    Manski, C. F. and Pepper, J. V. (2000), "Monotone instrumental
    variables: with an application to the returns to schooling",
    Econometrica 68(4), 997-1010.  Standard published form.  The article
    could not be obtained for this implementation: JSTOR (stable/2999452)
    returned a 5.8 kB access stub, two university course copies returned
    empty responses, and the NBER technical working paper t0224 downloaded
    as a zero-page PDF.  Only the bound stated in this docstring is
    claimed, and it is stated in full so it can be checked against the
    article by anyone who has it.
    """
    L = C.vec(lower)
    U = C.vec(upper)
    p = C.vec(prob)
    k = len(L)
    if len(U) != k or len(p) != k:
        raise ValueError("lower, upper and prob must have the same length")
    if k == 0:
        raise ValueError("need at least one instrument value")
    if any(v < 0.0 for v in p):
        raise ValueError("probabilities must be non-negative")
    tot = sum(p)
    if tot <= 0.0:
        raise ValueError("probabilities must not all be zero")
    p = [v / tot for v in p]
    Lv = []
    run = L[0]
    for v in range(k):
        run = max(run, L[v])
        Lv.append(run)
    Uv = [0.0] * k
    run = U[k - 1]
    for v in range(k - 1, -1, -1):
        run = min(run, U[v])
        Uv[v] = run
    lb = sum(p[v] * Lv[v] for v in range(k))
    ub = sum(p[v] * Uv[v] for v in range(k))
    return RichResult(payload={
        "lower": lb, "upper": ub, "width": ub - lb, "lowerv": Lv,
        "upperv": Uv, "prob": p, "k": k,
        "method": "Monotone instrumental variable bounds (Manski-Pepper 2000)"})


bound_monot_inst = mivbound


boundmonotinst = mivbound


def cheatsheet():
    return "bdmnts: Monotone instrumental variable bounds."
