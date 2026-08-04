# morie.fn -- function file (rootcoder007/morie)
"""Monotone treatment selection bounds."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mtsbound", "bound_monot_selection"]


def mtsbound(y, z, d, ymin, ymax):
    """Bounds on E[Y(d)] under monotone selection into treatment.

    Monotone treatment selection assumes that units choosing higher
    treatment levels would have had weakly higher outcomes at *every*
    counterfactual level: for t'' >= t',

        E[Y(t) | Z = t''] >= E[Y(t) | Z = t']   for all t.

    Taking t' or t'' equal to the level t makes the observed conditional
    mean E[Y | Z = t] an upper bound on the counterfactual mean in the
    groups that selected less, and a lower bound in those that selected
    more.  Averaging over the selection distribution,

        E[Y(d)] <= sum_{t <= d} P(Z=t) E[Y|Z=d] + sum_{t > d} P(Z=t) ymax
        E[Y(d)] >= sum_{t <  d} P(Z=t) ymin     + sum_{t >= d} P(Z=t) E[Y|Z=d]

    where the support endpoints fill the side selection monotonicity
    leaves open.  Combining this with monotone treatment response
    tightens both sides; that combination is the companion routine
    ``mtrbound``.

    Parameters
    ----------
    y : array-like
        Observed outcomes.
    z : array-like
        Observed treatment levels.
    d : float
        Level whose mean counterfactual is bounded; must be realised.
    ymin, ymax : float
        A priori support of the outcome.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``condmean``, ``pbelow``,
        ``pat``, ``pabove``, ``n``, ``d``.

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
    y = C.vec(y)
    z = C.vec(z)
    n = len(y)
    if len(z) != n:
        raise ValueError("y and z must have the same length")
    lo, hi = float(ymin), float(ymax)
    if lo > hi:
        raise ValueError("ymin must not exceed ymax")
    d = float(d)
    at = [i for i in range(n) if z[i] == d]
    if not at:
        raise ValueError("no unit is observed at treatment level d")
    cm = sum(y[i] for i in at) / len(at)
    pb = sum(1 for i in range(n) if z[i] < d) / n
    pa = sum(1 for i in range(n) if z[i] > d) / n
    pat = len(at) / n
    ub = (pb + pat) * cm + pa * hi
    lb = pb * lo + (pat + pa) * cm
    return RichResult(payload={
        "lower": lb, "upper": ub, "width": ub - lb, "condmean": cm,
        "pbelow": pb, "pat": pat, "pabove": pa, "n": n, "d": d,
        "method": "Monotone treatment selection bounds (Manski-Pepper 2000)"})


bound_monot_selection = mtsbound


def cheatsheet():
    return "bdmnsl: Monotone treatment selection bounds."
