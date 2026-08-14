"""Manski worst-case bound for partially observed outcomes (Manski 2007)."""

from ._richresult import RichResult

__all__ = ["bdtrns", "manski_bound"]


def bdtrns(y_obs, p_obs, y_lo, y_hi):
    """
    Identification region for a mean with missing (non-transported)
    outcomes, using the empirical evidence alone.

    Manski (2007), Eqs. 2.8-2.9: with z = 1 indicating observation
    and g(y) bounded in [g0, g1], the Law of Iterated Expectations
    gives E[g(y)|x] = E[g|x, z=1] P(z=1|x) + E[g|x, z=0] P(z=0|x);
    the data reveal nothing about the second conditional
    expectation, which can take any value in [g0, g1], so the
    identification region is the interval

        [ E[g|z=1] P(z=1) + g0 P(z=0),
          E[g|z=1] P(z=1) + g1 P(z=0) ],

    whose width is exactly (g1 - g0) P(z=0) -- shrinking to a point
    as observability approaches 1 and widening to the full outcome
    range as it approaches 0.  This is the "no assumptions" bound
    used for extrapolation/transport across populations (his Ch. 1.4
    and Ch. 2).

    Sources
    -------
    Manski, C. F. (2007). *Identification for Prediction and
    Decision*. Harvard University Press, Secs. 1.4, 2.1, Eqs.
    2.8-2.9 (local copy
    fetched-wave3/Identification_for_Prediction_and_Decision..pdf).

    Parameters
    ----------
    y_obs : sequence of float
        Observed outcomes (the z = 1 subsample).
    p_obs : float
        Observation probability P(z = 1) in [0, 1].
    y_lo, y_hi : float
        Logical outcome bounds g0 <= y <= g1.

    Returns
    -------
    RichResult
        Keys: lower, upper, width, observed_mean, p_obs.
    """
    yv = [float(v) for v in y_obs]
    if not yv:
        raise ValueError("need at least one observed outcome")
    p = float(p_obs)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p_obs must be in [0, 1]")
    g0 = float(y_lo)
    g1 = float(y_hi)
    if g1 < g0:
        raise ValueError("y_hi must be >= y_lo")
    if any(v < g0 - 1e-12 or v > g1 + 1e-12 for v in yv):
        raise ValueError("observed outcomes violate the stated bounds")
    m = sum(yv) / len(yv)
    lo = m * p + g0 * (1.0 - p)
    hi = m * p + g1 * (1.0 - p)
    return RichResult(payload={
        "lower": lo,
        "upper": hi,
        "width": hi - lo,
        "observed_mean": m,
        "p_obs": p,
        "method": "Manski (2007) worst-case bound (Eqs. 2.8-2.9)",
    })


# long descriptive alias (stub-era name)
manski_bound = bdtrns


def cheatsheet():
    return "bdtrns: [m p + g0 (1-p), m p + g1 (1-p)]; width = (g1-g0)(1-p)"

# public names resolved by fn/_lazy_map.json
bound_transport = bdtrns
boundtransport = bdtrns
