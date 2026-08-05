# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Hodrick-Prescott filter.

Hodrick and Prescott (1997), "Postwar U.S. business cycles: an
empirical investigation", Journal of Money, Credit and Banking
29(1):1-16, doi:10.2307/2953682 (circulated 1980 as Carnegie-Mellon
working paper 451).  The trend minimises

    sum_t (y_t - tau_t)^2 + lambda sum_t [(tau_{t+1} - tau_t) - (tau_t - tau_{t-1})]^2,

whose normal equations are the banded system (I + lambda D'D) tau = y
with D the second-difference operator.  lambda = 0 returns the series
itself and lambda -> infinity returns the least-squares straight line;
both limits are exact and are what the tests check.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hodrick_prescott"]


def hodrick_prescott(y, lam=1600.0):
    """Trend and cycle of a series at smoothing parameter lam."""
    v = core.vec(y)
    n = len(v)
    if n < 3:
        raise ValueError("hodrick_prescott: need at least three observations")
    lv = float(lam)
    if lv < 0:
        raise ValueError("hodrick_prescott: lam must be non-negative")
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        K[i][i] = 1.0
    for r in range(n - 2):
        row = [0.0] * n
        row[r] = 1.0
        row[r + 1] = -2.0
        row[r + 2] = 1.0
        for i in range(n):
            if row[i] == 0.0:
                continue
            for j in range(n):
                if row[j] != 0.0:
                    K[i][j] += lv * row[i] * row[j]
    tau = core.cholsolve(K, v)
    cyc = [v[i] - tau[i] for i in range(n)]
    ss = 0.0
    for c in cyc:
        ss += c * c
    rough = 0.0
    for r in range(n - 2):
        dd = tau[r] - 2.0 * tau[r + 1] + tau[r + 2]
        rough += dd * dd
    return RichResult(
        title="Hodrick-Prescott filter",
        summary_lines=[("n", n), ("lambda", lv)],
        payload={
            "estimate": ss + lv * rough,
            "trend": tau,
            "cycle": cyc,
            "cycle_ss": ss,
            "roughness": rough,
            "lam": lv,
            "n": n,
            "method": "(I + lambda D'D) tau = y, Hodrick & Prescott (1997)",
        },
    )


def cheatsheet():
    return "hodprc: Hodrick-Prescott filter"
