# morie.fn -- function file (rootcoder007/morie)
"""Runs declustering of threshold exceedances."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_declustering_runs"]


def evt_declustering_runs(x, u, r):
    """
    Runs declustering of threshold exceedances

    Formula: cluster = consecutive exceedances within gap r

    A cluster ends once r consecutive observations fall below the
    threshold.  The cluster maxima are approximately independent, which
    is what makes a GPD fit to them legitimate; the extremal index is
    estimated by the runs estimator, number of clusters over number of
    exceedances.

    Parameters
    ----------
    x : array-like
        Time-ordered series.
    u : float
        Threshold.
    r : int
        Run length: the number of consecutive non-exceedances that
        separates two clusters.

    Returns
    -------
    result : dict
        Keys: cluster_max, cluster_id, n_clusters, theta, n_exceed,
        estimate (extremal index), n.

    References
    ----------
    Smith (1989), Statistical Science 4(4):367-377.
    """
    x = core.vec(x)
    n = len(x)
    if n == 0:
        raise ValueError("empty input: x has no observations")
    u = float(u)
    r = int(r)
    if r < 1:
        raise ValueError("r must be at least 1")
    cid = [0] * n
    cur = 0
    gap = r + 1
    for i in range(n):
        if x[i] > u:
            if gap > r:
                cur += 1
            gap = 0
            cid[i] = cur
        else:
            gap += 1
    cmax = []
    for c in range(1, cur + 1):
        vals = [x[i] for i in range(n) if cid[i] == c]
        cmax.append(max(vals))
    nex = sum(1 for v in cid if v > 0)
    theta = cur / float(nex) if nex else float("nan")
    return RichResult(payload={
        "cluster_max": cmax,
        "cluster_id": cid,
        "n_clusters": cur,
        "theta": theta,
        "n_exceed": nex,
        "estimate": theta,
        "n": n,
        "method": "runs declustering of threshold exceedances",
    })


def cheatsheet():
    return "evdec: runs declustering of threshold exceedances"


# compact alias per ledger/NAMING.md
evtdeclusteringruns = evt_declustering_runs
