# morie.fn -- function file (rootcoder007/morie)
"""NLS estimator for parametric survival."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["nonlinear_least_squares_surv"]


def nonlinear_least_squares_surv(time, event, model="weibull", t_grid=None):
    r"""Least-squares fit of a parametric survival curve to the
    Kaplan-Meier estimate:

    .. math:: \min_\theta \sum_j
              \big[\hat S_{KM}(t_j) - S(t_j;\theta)\big]^2 .

    Fits the CURVE rather than the data, and that distinction is the
    whole caveat. Maximum likelihood uses each observation and its
    censoring status; this uses only the fitted step function
    evaluated on a grid, so it discards information and its standard
    errors are not the likelihood ones. Points are also not
    independent -- successive Kaplan-Meier values share the same
    at-risk sets -- so the sum of squares is not a valid objective
    for inference.

    It survives because it is robust to things maximum likelihood is
    not: it needs no correct likelihood, converges from poor starts,
    and gives a serviceable check on a parametric fit by eye. Used
    for inference it is simply wrong, and ``valid_for_inference``
    says so.

    Parameters
    ----------
    time, event : array-like
        Times and 0/1 indicators.
    model : {"weibull", "exponential"}
        Parametric family.
    t_grid : array-like, optional
        Fitting grid.

    Returns
    -------
    RichResult
        keys: ``params``, ``model``, ``sse``, ``t_grid``,
        ``fitted``, ``km``, ``valid_for_inference`` (False), ``n``,
        ``method``.
    """
    from scipy import optimize

    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    n = tv.size
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    if ev.size != n:
        raise ValueError(f"event has {ev.size} entries for {n} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if model not in ("weibull", "exponential"):
        raise ValueError("model must be 'weibull' or 'exponential'.")
    uniq = np.unique(tv)
    km = np.ones(uniq.size)
    s = 1.0
    for i, v in enumerate(uniq):
        at = float(np.sum(tv >= v))
        d = float(np.sum((tv == v) & (ev == 1.0)))
        if at > 0 and d > 0:
            s *= 1.0 - d / at
        km[i] = s
    g = uniq if t_grid is None else \
        np.atleast_1d(np.asarray(t_grid, dtype=float))
    target = np.interp(g, uniq, km)

    if model == "exponential":
        def surv(p, t):
            return np.exp(-np.exp(p[0]) * t)
        p0 = np.array([np.log(1.0 / max(tv.mean(), 1e-6))])
    else:
        def surv(p, t):
            return np.exp(-(t / np.exp(p[0])) ** np.exp(p[1]))
        p0 = np.array([np.log(max(tv.mean(), 1e-6)), 0.0])

    res = optimize.least_squares(lambda p: surv(p, g) - target, p0)
    params = np.exp(res.x)
    return RichResult(payload={
        "params": params, "model": model, "sse": float(np.sum(res.fun ** 2)),
        "t_grid": g, "fitted": surv(res.x, g), "km": target,
        "valid_for_inference": False,
        "why_not": "it fits the CURVE, not the data: it ignores censoring "
                   "status per observation and the Kaplan-Meier points are "
                   "not independent, so the sum of squares is not a likelihood",
        "use_for": "a robust visual check on a parametric fit",
        "n": int(n),
        "method": "Least squares of a parametric survival curve against Kaplan-Meier"})


def cheatsheet():
    return "survnls: fine as a fit check, invalid for inference -- KM points are not independent"
