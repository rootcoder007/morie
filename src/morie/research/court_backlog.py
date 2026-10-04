# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P16: court backlog, Little's law on a finite docket (``research/lean/P16Backlog.lean``; Little 1961).

* ``Research.P16.occupancy_integral`` / ``little`` / ``little_backlog`` / ``little_target``

R parity: ``rmorie`` ``R/court_backlog.R`` (``morie_court_backlog``).
"""

from __future__ import annotations

import math

__all__ = ["court_backlog", "backlog_censoring"]


def court_backlog(arrivals, dispositions, horizon=None, target_backlog=None) -> dict:
    """Little's law on a docket: time-average pending cases = filing rate x mean disposition time.

    Examples
    --------
    >>> r = court_backlog([0, 1, 2, 4, 5, 7], [3, 2.5, 6, 5, 9, 10], horizon=10, target_backlog=1)
    >>> (round(r["average_backlog"], 12), round(r["required_mean_time"], 12))
    (1.65, 1.666666666667)
    """
    a = [float(v) for v in arrivals]
    d = [float("nan") if v is None else float(v) for v in dispositions]
    n = len(a)
    if len(d) != n or n == 0:
        raise ValueError("arrivals and dispositions must have the same positive length")
    if any(v != v or v < 0 for v in a):
        raise ValueError("arrivals must be non-negative")
    if horizon is None:
        horizon = max(v for v in d if v == v)
    horizon = float(horizon)
    if horizon != horizon or horizon <= 0:
        raise ValueError("horizon must be a single positive number")
    if any(v > horizon for v in a):
        raise ValueError("every arrival must lie inside the horizon")
    censored = [(v != v) or v > horizon for v in d]
    if any((not c) and dv < av for c, dv, av in zip(censored, d, a)):
        raise ValueError("dispositions cannot precede arrivals")
    obs = [(av, dv) for c, av, dv in zip(censored, a, d) if not c]
    occupancy = sum(dv - av for av, dv in obs) + sum(horizon - av for c, av in zip(censored, a) if c)
    rate = n / horizon
    mean_wait = sum(dv - av for av, dv in obs) / len(obs) if obs else float("nan")
    out = {
        "n": n,
        "n_censored": sum(censored),
        "horizon": horizon,
        "filing_rate": rate,
        "mean_disposition_time": mean_wait,
        "average_backlog": occupancy / horizon,
        "occupancy_integral": occupancy,
        "pending_at_horizon": sum(censored),
        "little_identity_check": (occupancy / horizon - rate * mean_wait) if not any(censored) else float("nan"),
    }
    if target_backlog is not None:
        if float(target_backlog) < 0:
            raise ValueError("target_backlog must be non-negative")
        out["required_mean_time"] = float(target_backlog) / rate
    out["theorems"] = [
        "Research.P16.occupancy_integral",
        "Research.P16.little",
        "Research.P16.little_backlog",
        "Research.P16.little_target",
    ]
    return out


def _pending_count(a, d, t):
    return sum(1 for av, dv in zip(a, d) if av <= t < dv)


def occupancy_on_grid(arrivals, dispositions, horizon, step=0.001) -> float:
    """Riemann check of the occupancy integral (used by the tests)."""
    a = [float(v) for v in arrivals]
    d = [float(v) for v in dispositions]
    k = int(math.floor(horizon / step))
    return sum(_pending_count(a, d, i * step) for i in range(k + 1)) * step


def backlog_censoring(disposed, pending_ages) -> dict:
    """The disposed-cases mean as a bound: what the pending cases imply.

    ``Research.P16Censoring.true_mean_ge`` / ``lower_bound_sub`` / ``bias_lower`` / ``disposed_understates`` /
    ``no_upper_bound``.

    Examples
    --------
    >>> b = backlog_censoring([30, 45, 60, 90, 120], [100, 150, 200])
    >>> (b["disposed_mean"], b["lower_bound"], b["bias_lower"], b["understates"], b["upper_bound"])
    (69.0, 99.375, 30.375, True, inf)
    """
    try:
        t = [float(x) for x in disposed]
        a = [float(x) for x in pending_ages]
    except (TypeError, ValueError) as exc:
        raise ValueError("disposed and pending_ages must be numeric") from exc
    if any(math.isnan(x) for x in t + a):
        raise ValueError("no missing values allowed")
    if any(x < 0 for x in t + a):
        raise ValueError("durations and ages must be non-negative")
    n, m = len(t), len(a)
    if n < 1 or m < 1:
        raise ValueError("need at least one disposed and one pending case")
    tbar = math.fsum(t) / n
    abar = math.fsum(a) / m
    lower = (math.fsum(t) + math.fsum(a)) / (n + m)
    return {
        "n": n,
        "m": m,
        "disposed_mean": tbar,
        "pending_age": abar,
        "lower_bound": lower,
        "bias_lower": (m / (n + m)) * (abar - tbar),
        "understates": tbar <= abar,
        "upper_bound": math.inf,
        "theorems": [
            "Research.P16Censoring.true_mean_ge",
            "Research.P16Censoring.lower_bound_sub",
            "Research.P16Censoring.bias_lower",
            "Research.P16Censoring.disposed_understates",
            "Research.P16Censoring.no_upper_bound",
        ],
    }
