# morie.fn -- function file (rootcoder007/morie)
"""Log-rank test for two-sample survival."""

from __future__ import annotations

from . import _stats_core as stats
from . import _t4core as T

from ._richresult import RichResult

__all__ = ["logrank_test"]


def logrank_test(time, event, group):
    """Mantel log-rank test comparing two survival curves.

    At each distinct death time with ``d`` deaths among ``r`` at risk,
    ``r1`` of them in group 1,

        ``E1 += d r1 / r``,
        ``V  += d (r1/r) (1 - r1/r) (r - d) / (r - 1)``

    and ``chi2 = (O1 - E1)^2 / V`` on one degree of freedom.  The
    ``(r-d)/(r-1)`` factor is the finite-population correction of the
    hypergeometric variance and vanishes when everyone still at risk
    dies, which is why a risk set of one contributes nothing.

    Ties in time are pooled into a single risk set rather than broken;
    ordering the loop from the largest time downwards makes the risk set
    a running count and keeps the two language arms bit-comparable.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    event : array-like
        1 for an observed death, 0 for a right-censored observation.
    group : array-like
        Two distinct labels; the lower one sorts to group 1.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``observed``, ``expected``,
        ``var``, ``n``, ``method``.

    References
    ----------
    Mantel (1966), Evaluation of survival data and two new rank order
    statistics arising in its consideration, Cancer Chemotherapy Reports
    50:163-170; Peto and Peto (1972), JRSS A 135:185-207.  The coded
    form was read from Therneau's ``survival`` package, src/survdiff2.c
    (fetched from github.com/therneau/survival), whose inner loop is
    ``exp[k] += wt*deaths*risk[k]/nrisk`` and
    ``tmp = wt*deaths*risk[j]*(nrisk-deaths)/(nrisk*(nrisk-1))`` with
    ``var[j][j] = tmp (1 - risk[j]/nrisk)`` -- the rho = 0 (log-rank)
    branch of the G-rho family.
    """
    time = T.vec(time)
    event = T.vec(event)
    g = T.vec(group)
    n = len(time)
    if len(event) != n or len(g) != n:
        raise ValueError("time, event and group must be the same length")
    labels = sorted(set(g))
    if len(labels) != 2:
        raise ValueError("logrank_test compares exactly two groups")
    lo = labels[0]
    order = sorted(range(n), key=lambda i: (time[i], i))
    tt = [time[i] for i in order]
    ee = [event[i] for i in order]
    gg = [1 if g[i] == lo else 2 for i in order]
    o1 = sum(ee[i] for i in range(n) if gg[i] == 1)
    e1 = 0.0
    v = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tt[j + 1] == tt[i]:
            j += 1
        r = n - i
        r1 = sum(1 for k in range(i, n) if gg[k] == 1)
        d = sum(ee[k] for k in range(i, j + 1))
        if d > 0:
            f = r1 / r
            e1 += d * f
            if r > 1:
                v += d * f * (1.0 - f) * (r - d) / (r - 1.0)
        i = j + 1
    chi2 = (o1 - e1) ** 2 / v if v > 0 else float("nan")
    p = 1.0 - stats.chi2.cdf(chi2, 1) if v > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(chi2),
            "p_value": float(p),
            "observed": float(o1),
            "expected": float(e1),
            "var": float(v),
            "n": int(n),
            "method": "Log-rank (Mantel) two-sample test",
        }
    )


def cheatsheet():
    return "logrank_test(time, event, group): (O-E)^2/V, chi2 on 1 df."


# compact alias per ledger/NAMING.md
logranktest = logrank_test
