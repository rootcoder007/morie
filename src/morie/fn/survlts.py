# morie.fn -- function file (rootcoder007/morie)
"""Smoothed life-table estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["life_table_smoothed"]


def life_table_smoothed(time, event, bandwidth=None, grid=None):
    r"""Kernel-smoothed survival curve.

    The Kaplan-Meier estimator is a STEP function, and a step
    function is a poor estimate of a continuous survival curve: it is
    not differentiable, so no hazard can be read off it, and it
    cannot be interpolated between event times. Smoothing solves both
    -- it is the same obstruction that appears when a baseline hazard
    is wanted from a cumulative one.

    Smoothing costs the exact interpretation. The smoothed curve is
    not the nonparametric maximum-likelihood estimate any more, it
    need not be monotone at very small bandwidths, and it is biased
    at the boundaries where the kernel has support outside the data.
    ``monotone`` is checked rather than assumed.

    Parameters
    ----------
    time, event : array-like
        Times and 0/1 indicators.
    bandwidth : float, optional
        Kernel bandwidth; a spread-scaled ``n^{-1/5}`` otherwise.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``survival_smooth``, ``survival_km``,
        ``bandwidth``, ``monotone``, ``is_npmle`` (False),
        ``n_events``, ``n``, ``method``.
    """
    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    n = tv.size
    if n < 5:
        raise ValueError(f"need at least 5 observations, got {n}.")
    if ev.size != n:
        raise ValueError(f"event has {ev.size} entries for {n} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    uniq = np.unique(tv)
    km = np.ones(uniq.size)
    s = 1.0
    for i, v in enumerate(uniq):
        at = float(np.sum(tv >= v))
        d = float(np.sum((tv == v) & (ev == 1.0)))
        if at > 0 and d > 0:
            s *= 1.0 - d / at
        km[i] = s
    if bandwidth is None:
        # Scale by a ROBUST spread, not the full range. Survival times
        # are right-skewed, so max - min is set by the single largest
        # observation and a range-scaled bandwidth over-smooths badly:
        # measured on exponential(2) data at n = 200 it dropped the
        # correlation with Kaplan-Meier to 0.84.
        sd = float(np.std(tv, ddof=1))
        iqr = float(np.subtract(*np.percentile(tv, [75, 25])))
        scale = min(sd, iqr / 1.349) if iqr > 0 else sd
        h = 1.06 * (scale if scale > 0 else 1.0) * n ** -0.2
    else:
        h = float(bandwidth)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    g = np.linspace(uniq.min(), uniq.max(), 100) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    W = np.exp(-0.5 * ((g[:, None] - uniq[None, :]) / h) ** 2)
    den = W.sum(axis=1)
    sm = np.where(den > 0, (W @ km) / np.maximum(den, 1e-300), np.nan)
    return RichResult(payload={
        "grid": g, "survival_smooth": sm,
        "survival_km": np.interp(g, uniq, km),
        "bandwidth": h,
        "monotone": bool(np.all(np.diff(sm[np.isfinite(sm)]) <= 1e-12)),
        "is_npmle": False,
        "cost": "not the NPMLE any more, can lose monotonicity at small h, "
                "and is biased at the boundaries",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Kernel-smoothed survival; buys differentiability, costs the exact NPMLE interpretation"})


def cheatsheet():
    return "survlts: a step function has no hazard to read off -- smoothing buys that and costs the NPMLE"
