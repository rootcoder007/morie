# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P16: court backlog, Little's law on a finite docket (``research/lean/P16Backlog.lean``; Little 1961).

* ``Research.P16.occupancy_integral`` / ``little`` / ``little_backlog`` / ``little_target``

R parity: ``rmorie`` ``R/court_backlog.R`` (``morie_court_backlog``).
"""

from __future__ import annotations

import math

__all__ = ["court_backlog", "backlog_censoring", "backlog_published"]


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


def backlog_published(received, days, pending, mean_days, band_shares=None, band_floors=None) -> list:
    """Published court statistics read through Little's law (``Research.P16.little_target``,
    ``Research.P16Censoring.disposed_understates``, ``band_floor_le_mean``).

    R parity: ``morie_backlog_published``.

    Examples
    --------
    >>> r = backlog_published(21265, 30, 148904, 175, band_shares=[[0.808, 0.105, 0.034, 0.053]],
    ...                       band_floors=[0, 304, 456, 548])[0]
    >>> (round(r["implied_mean_days"], 4), round(r["ratio"], 6), round(r["band_floor_bound"], 3))
    (210.0691, 1.200395, 76.468)
    """

    def vec(x, name, pos=True):
        v = list(x) if isinstance(x, (list, tuple)) else [x]
        try:
            v = [float(e) for e in v]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite numbers") from exc
        if any(not math.isfinite(e) for e in v):
            raise ValueError(f"{name} must be finite numbers")
        if pos and any(e <= 0 for e in v):
            raise ValueError(f"{name} must be positive")
        if not pos and any(e < 0 for e in v):
            raise ValueError(f"{name} must be non-negative")
        return v

    cols = [
        vec(received, "received"),
        vec(days, "days"),
        vec(pending, "pending", pos=False),
        vec(mean_days, "mean_days"),
    ]
    n = max(len(c) for c in cols)
    for c, name in zip(cols, ["received", "days", "pending", "mean_days"]):
        if len(c) not in (1, n):
            raise ValueError(f"{name} must have length 1 or {n}")
    rec, dy, pend, md = ([c[0]] * n if len(c) == 1 else c for c in cols)
    shares = None
    if band_shares is not None:
        shares = [list(map(float, r)) for r in band_shares]
        if len(shares) == 1 and n > 1:
            shares = shares * n
        if len(shares) != n:
            raise ValueError(f"band_shares must have one row per window ({n})")
        if band_floors is None or len(band_floors) != len(shares[0]):
            raise ValueError("band_floors must give one lower edge per band")
        if any(e != e or e < 0 for r in shares for e in r):
            raise ValueError("band_shares must be non-negative")
        if any(abs(sum(r) - 1) > 0.01 for r in shares):
            raise ValueError("each row of band_shares must sum to 1 (within rounding)")
        fl = [float(f) for f in band_floors]
        if any(f != f or f < 0 for f in fl) or any(fl[i] > fl[i + 1] for i in range(len(fl) - 1)):
            raise ValueError("band_floors must be non-negative and increasing")
    out = []
    for i in range(n):
        rate = rec[i] / dy[i]
        row = {
            "filing_rate": rate,
            "implied_mean_days": pend[i] / rate,
            "published_mean_days": md[i],
            "ratio": (pend[i] / rate) / md[i],
            "little_backlog": rate * md[i],
            "pending": pend[i],
        }
        if shares is not None:
            row["band_floor_bound"] = math.fsum(s * f for s, f in zip(shares[i], fl)) / math.fsum(shares[i])
        out.append(row)
    return out
