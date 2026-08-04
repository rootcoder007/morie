# morie.fn -- function file (rootcoder007/morie)
"""Inverse-probability-of-censoring weighted estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ipcw_estimator"]


def ipcw_estimator(time, event, tau=None, y=None):
    r"""Inverse-probability-of-censoring weighted (IPCW) estimator:

    .. math:: \hat\theta = \frac1n\sum_i
              \frac{\Delta_i\, Y_i}{\hat G(T_i-)},

    weighting each uncensored observation by the reciprocal of its
    probability of having escaped censoring that long.

    The idea is that censoring removes subjects non-randomly with
    respect to TIME -- late events are systematically more likely to
    be censored -- so unweighted complete-case analysis is biased
    even under completely random censoring. Weighting by
    :math:`1/\hat G` restores the distribution that would have been
    observed without it.

    The weights explode where :math:`\hat G` is small, which is to
    say late in follow-up, and that is the method's known failure
    mode rather than an implementation detail: a handful of
    long-surviving subjects can dominate the estimate.
    ``max_weight`` and ``effective_n`` are returned so the
    concentration is visible, and ``tau`` truncates follow-up, which
    is the standard remedy.

    Parameters
    ----------
    time : array-like
        Observed times.
    event : array-like of {0, 1}
        1 for an event, 0 for censoring.
    tau : float, optional
        Truncation time; weights past it are dropped.
    y : array-like, optional
        Quantity to average; the event indicator otherwise, which
        estimates :math:`P(T \le \tau)`.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``weights``, ``G``, ``max_weight``,
        ``effective_n``, ``tau``, ``n_used``, ``n``, ``method``.
    """
    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    n = tv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if ev.size != n:
        raise ValueError(f"event has {ev.size} entries for {n} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if np.any(tv < 0):
        raise ValueError("times must be non-negative.")
    # Kaplan-Meier for the CENSORING distribution: events and
    # censorings swap roles
    order = np.argsort(tv)
    ts, cs = tv[order], 1.0 - ev[order]
    G = np.ones(n)
    g = 1.0
    for i in range(n):
        at = float(np.sum(ts >= ts[i]))
        d = float(np.sum((ts == ts[i]) & (cs == 1.0)))
        if at > 0 and d > 0 and i == np.searchsorted(ts, ts[i]):
            g *= 1.0 - d / at
        G[i] = g
    Gt = np.empty(n)
    Gt[order] = G
    tt = float(np.max(tv)) if tau is None else float(tau)
    use = (ev == 1.0) & (tv <= tt) & (Gt > 0)
    if not np.any(use):
        raise ValueError("no uncensored observations with positive censoring "
                         "survival; nothing to weight.")
    yv = np.ones(n) if y is None else np.asarray(y, dtype=float).ravel()
    if yv.size != n:
        raise ValueError(f"y has {yv.size} entries for {n} observations.")
    w = np.zeros(n)
    w[use] = 1.0 / Gt[use]
    est = float(np.sum(w * yv) / n)
    ww = w[use]
    return RichResult(payload={
        "estimate": est, "weights": w, "G": Gt,
        "max_weight": float(ww.max()),
        "effective_n": float(ww.sum() ** 2 / np.sum(ww ** 2)),
        "tau": tt, "n_used": int(use.sum()), "n": int(n),
        "failure_mode": "weights explode where G is small, i.e. late in "
                        "follow-up; truncating at tau is the standard remedy",
        "method": "IPCW; reweights the uncensored to stand for those censored before them"})


def cheatsheet():
    return "survipw: the weights blow up late in follow-up -- check effective_n, truncate at tau"


# compact alias per ledger/NAMING.md
ipcwestimator = ipcw_estimator
