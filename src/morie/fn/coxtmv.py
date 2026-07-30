# morie.fn -- function file (rootcoder007/morie)
"""Cox model with time-varying coefficients."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["cox_time_varying"]


def cox_time_varying(time, event, X, n_intervals=3, ties="efron"):
    r"""Fit a Cox model allowing the coefficient itself to change over time.

    Follow-up is cut at quantiles of the event times and a separate
    :math:`\beta` is estimated in each interval, giving a piecewise-constant
    :math:`\beta(t)`. This is the direct remedy when
    :func:`~morie.fn.coxres.cox_schoenfeld_residuals` shows proportional
    hazards failing: rather than forcing one hazard ratio on the whole
    follow-up, report how it moves.

    A time-varying **coefficient** is not a time-varying **covariate**, and the
    two are routinely confused. This function handles the former -- the effect
    of a fixed covariate changing over time. A covariate whose *value* changes
    needs the counting-process (start, stop] data layout instead.

    Treatment effects that wane are the standard case: a hazard ratio of 0.5
    in the first year and 0.9 thereafter is a real and reportable finding,
    where a single pooled ratio of 0.7 describes neither period.

    Late intervals often carry few events, so their coefficients are
    imprecise; ``events_per_interval`` is returned so that is visible rather
    than inferred from wide confidence intervals.

    Parameters
    ----------
    time, event, X : array-like
        Survival data.
    n_intervals : int
        Number of time intervals, at least 1.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``beta`` ``(n_intervals, p)``, ``se``, ``hazard_ratio``,
        ``cutpoints``, ``events_per_interval``, ``constant_beta``,
        ``lr_vs_constant``, ``p_vs_constant``.

    References
    ----------
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    A treatment effect that wanes is recovered as two different coefficients,
    where a single Cox fit reports one intermediate value that fits neither
    period.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.integers(0, 2, 1200).astype(float)
    >>> early = rng.exponential(1 / np.exp(-1.2 * x))       # strong effect early
    >>> late = 1.0 + rng.exponential(1 / np.exp(-0.1 * x))  # weak effect later
    >>> T = np.where(early < 1.0, early, late)
    >>> C = rng.exponential(4.0, 1200)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_time_varying(t, e, x[:, None], n_intervals=2)
    >>> bool(r["beta"][0, 0] < r["beta"][1, 0])
    True

    The likelihood-ratio test against a constant coefficient detects it.

    >>> bool(r["p_vs_constant"] < 0.05)
    True

    Event counts per interval are reported, since late intervals are thin.

    >>> int(r["events_per_interval"].sum()) == int(e.sum())
    True

    >>> cox_time_varying(t, e, x[:, None], n_intervals=0)
    Traceback (most recent call last):
        ...
    ValueError: n_intervals must be at least 1
    """
    from scipy.stats import chi2, norm

    t, e, Xm = prepare(time, event, X)
    n_intervals = int(n_intervals)
    if n_intervals < 1:
        raise ValueError("n_intervals must be at least 1")
    p = Xm.shape[1]
    ev_times = np.sort(t[e == 1])
    if ev_times.size < n_intervals:
        raise ValueError(
            f"only {ev_times.size} events, too few for {n_intervals} intervals"
        )
    qs = np.linspace(0, 1, n_intervals + 1)[1:-1]
    cuts = np.unique(np.quantile(ev_times, qs)) if qs.size else np.array([])
    edges = np.r_[0.0, cuts, np.inf]

    betas = np.zeros((n_intervals, p))
    ses = np.zeros((n_intervals, p))
    counts = np.zeros(n_intervals, dtype=int)
    ll_tv = 0.0
    for j in range(n_intervals):
        lo, hi = edges[j], edges[j + 1]
        # Everyone at risk at `lo` contributes; events after `hi` are censored
        # at `hi`, which is the interval-specific risk set.
        keep = t > lo
        tj = np.minimum(t[keep], hi) - lo
        ej = np.where(t[keep] <= hi, e[keep], 0.0)
        counts[j] = int(ej.sum())
        if counts[j] == 0:
            ses[j] = np.nan
            continue
        b, ll_j, I, _, _, _ = cox_fit(tj, ej, Xm[keep], ties=ties)
        betas[j] = b
        ll_tv += ll_j
        try:
            ses[j] = np.sqrt(np.clip(np.diag(np.linalg.inv(I)), 0, None))
        except np.linalg.LinAlgError:
            ses[j] = np.nan

    b_const, ll_const, I_const, _, _, _ = cox_fit(t, e, Xm, ties=ties)
    lr = float(max(2.0 * (ll_tv - ll_const), 0.0))
    df = max(p * (n_intervals - 1), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = betas / ses
    return RichResult(
        title="Cox model with time-varying coefficients",
        summary_lines=[("intervals", n_intervals), ("events", int(e.sum())),
                       ("LR vs constant", lr)],
        warnings=(["late intervals often hold few events; read "
                   "events_per_interval before trusting a late coefficient"]),
        payload={
            "beta": betas, "se": ses, "z": z,
            "p_value": 2 * norm.sf(np.abs(z)),
            "hazard_ratio": np.exp(betas), "cutpoints": cuts,
            "events_per_interval": counts, "constant_beta": b_const,
            "loglik": ll_tv, "loglik_constant": ll_const,
            "lr_vs_constant": lr, "p_vs_constant": float(chi2.sf(lr, df)),
            "n_intervals": n_intervals, "n": int(t.size),
            "method": "cox_time_varying",
        },
    )


def cheatsheet():
    return "coxtmv: time-varying COEFFICIENT (not covariate); the fix when Schoenfeld shows PH failing"
