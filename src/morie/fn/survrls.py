# morie.fn -- function file (rootcoder007/morie)
"""Restricted mean survival time over a fixed horizon."""

from .rmst import rmst_estimate

from ._richresult import RichResult

__all__ = ["restricted_lifetime"]


def restricted_lifetime(fit, event, t_star):
    """Area under the Kaplan-Meier curve out to ``t_star``.

    RMST answers what a hazard ratio does not: how much longer, on
    average, over a stated window -- and it needs no proportional-hazards
    assumption.  The horizon is not optional in substance, because past
    the last observed time the curve is a plateau the data cannot pin
    down, so ``t_star`` is required here rather than defaulted.

    The estimator already exists in the tree, so this module is a thin
    alias for ``rmst.rmst_estimate`` rather than a second implementation.

    Formula: ``RMST(t*) = integral_0^{t*} S(u) du``, the survival curve
    being a step function integrated as summed rectangles.

    Parameters
    ----------
    fit : array-like
        Observed event or censoring times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.
    t_star : float
        Restriction horizon, positive.

    Returns
    -------
    RichResult
        ``estimate`` (RMST), ``tau``, ``n_events``, ``method``.

    References
    ----------
    Royston, P. & Parmar, M. K. B. (2013).  Restricted mean survival time:
    an alternative to the hazard ratio for the design and analysis of
    randomized trials with a time-to-event outcome.  BMC Medical Research
    Methodology 13:152.  <https://doi.org/10.1186/1471-2288-13-152>
    Klein, J. P. & Moeschberger, M. L. (2003).  Survival Analysis, 2nd
    edition.  Springer, section 4.5.
    """
    if float(t_star) <= 0.0:
        raise ValueError("restricted_lifetime: t_star must be positive")
    r = rmst_estimate(fit, event, tau=float(t_star))
    d = r.to_dict()
    return RichResult(payload={
        "estimate": float(d["value"]), "tau": float(d["tau"]),
        "n_events": int(d["n_events"]),
        "method": "RMST(t*) = area under the KM curve [Royston & Parmar 2013]"})


# CANONICAL TEST
# >>> # three uncensored events at 1,2,3: S steps 1 -> 2/3 -> 1/3 -> 0,
# >>> # so the area to 3 is exactly 1 + 2/3 + 1/3 = 2
# >>> r = restricted_lifetime([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 3.0)
# >>> assert abs(r["estimate"] - 2.0) < 1e-12
# >>> # horizon before any event: RMST is exactly the horizon
# >>> assert abs(restricted_lifetime([2.0, 4.0], [1.0, 1.0], 1.0)["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "survrls(fit, event, t_star): RMST to t_star (alias of rmst)."
