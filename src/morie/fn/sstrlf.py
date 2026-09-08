# morie.fn -- function file (rootcoder007/morie)
"""Left-truncated survival adjustment."""

from .lftrt import lftrt

from ._richresult import RichResult

__all__ = ["surv_truncation_left"]


def surv_truncation_left(entry, time, event):
    """Kaplan-Meier survival with the risk set conditional on entry < t.

    Under delayed entry a subject is not at risk before its entry time,
    so the risk set at ``t`` counts only those with ``entry < t <= time``.
    Ignoring the entry times inflates the early risk sets and biases
    survival upwards.  The delayed-entry estimator already exists in the
    tree, so this module is a thin alias for ``lftrt.lftrt`` rather than
    a second implementation.

    Formula: ``S(t) = prod_{t_j <= t} (1 - d_j / n_j)`` with
    ``n_j = #{i : entry_i < t_j <= time_i}``.

    Parameters
    ----------
    entry : array-like
        Left-truncation (entry) times.
    time : array-like
        Observed event or censoring times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.

    Returns
    -------
    RichResult
        ``estimate`` (survival at the last event time), ``times``,
        ``survival``, ``se``, ``ci_lower``, ``ci_upper``, ``n_obs``,
        ``n_events``, ``method``.

    References
    ----------
    Klein, J. P. & Moeschberger, M. L. (2003).  Survival Analysis:
    Techniques for Censored and Truncated Data, 2nd edition.  Springer,
    section 4.2.
    """
    r = lftrt(entry, time, event)
    s = list(r["survival"])
    return RichResult(payload={
        "estimate": float(s[-1]) if s else float("nan"),
        "times": r["times"], "survival": r["survival"], "se": r["se"],
        "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
        "n_obs": int(r["n_obs"]), "n_events": int(r["n_events"]),
        "method": "delayed-entry Kaplan-Meier [Klein & Moeschberger 2003]"})


# CANONICAL TEST
# >>> # no truncation, four distinct event times: S drops 1 -> .75 -> .5 -> .25
# >>> r = surv_truncation_left([0, 0, 0, 0], [5, 6, 7, 8], [1, 1, 1, 1])
# >>> assert abs(float(r["survival"][0]) - 0.75) < 1e-12
# >>> assert abs(r["estimate"]) < 1e-12


def cheatsheet():
    return "sstrlf(entry, time, event): delayed-entry KM (alias of lftrt)."
