# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap SE for survival estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survival_bootstrap_se"]


def survival_bootstrap_se(time, event, t_grid=None, B=200, seed=0):
    r"""Bootstrap standard errors for the Kaplan-Meier estimator.

    Resamples subjects with replacement, recomputes the whole
    survival curve per replicate, and takes the pointwise standard
    deviation.

    Worth having beside Greenwood's formula rather than instead of
    it. Greenwood is a delta-method variance and is known to be poor
    in the tail, where few subjects remain at risk; the bootstrap
    makes no such approximation and is usually better there. Both
    are computed and returned, along with their ratio, because the
    place they disagree is precisely the place the answer matters.

    Resampling is at the SUBJECT level, which is what preserves the
    censoring pattern; resampling event times alone would destroy it.

    Parameters
    ----------
    time, event : array-like
        Times and 0/1 indicators.
    t_grid : array-like, optional
        Points at which to report; the distinct times otherwise.
    B : int
        Replicates.
    seed : int
        RNG seed.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``survival``, ``bootstrap_se``,
        ``greenwood_se``, ``se_ratio``, ``B``, ``resample_level``,
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
    Bn = int(B)
    if Bn < 20:
        raise ValueError(f"need at least 20 replicates, got {Bn}.")
    g = np.unique(tv) if t_grid is None else \
        np.atleast_1d(np.asarray(t_grid, dtype=float))

    def km(t, e, grid):
        s = 1.0
        out = np.empty(grid.size)
        var = np.zeros(grid.size)
        acc = 0.0
        for i, v in enumerate(grid):
            for u in np.unique(t[(t <= v) & (e == 1.0)]):
                pass
            at_all = t >= v
            s = 1.0
            acc = 0.0
            for u in np.unique(t[(e == 1.0) & (t <= v)]):
                at = float(np.sum(t >= u))
                d = float(np.sum((t == u) & (e == 1.0)))
                if at > 0:
                    s *= 1.0 - d / at
                    if at > d:
                        acc += d / (at * (at - d))
            out[i] = s
            var[i] = s * s * acc
        return out, np.sqrt(np.maximum(var, 0.0))

    surv, green = km(tv, ev, g)
    rng = np.random.default_rng(int(seed))
    reps = np.empty((Bn, g.size))
    for b in range(Bn):
        idx = rng.integers(0, n, n)          # SUBJECT-level resampling
        reps[b], _ = km(tv[idx], ev[idx], g)
    bse = reps.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(green > 0, bse / green, np.nan)
    return RichResult(payload={
        "t_grid": g, "survival": surv, "bootstrap_se": bse,
        "greenwood_se": green, "se_ratio": ratio, "B": Bn,
        "resample_level": "subject, which preserves the censoring pattern",
        "why_both": "Greenwood is a delta-method variance and is known to be "
                    "poor in the tail, where few remain at risk",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Bootstrap SE for Kaplan-Meier, reported beside Greenwood"})


def cheatsheet():
    return "survbs: resample SUBJECTS, not event times -- and compare with Greenwood in the tail"
