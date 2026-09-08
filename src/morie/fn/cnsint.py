# morie.fn -- function file (rootcoder007/morie)
"""Concurrent calibration of two groups on common anchor items."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["concurrent_calibration"]


def rasch_jmle(X, iters=200, prior_var=4.0):
    """Joint MAP for the Rasch model: item difficulties b, ability theta.

    Difficulties are centred at zero each cycle, which is the
    identification constraint; without it b and theta drift together and
    the two groups can never be compared.  A N(0, prior_var) penalty is
    carried on both sets of parameters because joint ML alone DIVERGES
    on perfectly separated response patterns -- an examinee with a
    perfect score has no finite ML ability, and the estimates walk off
    to +/- 80 while the ordering still looks right.
    """
    n = len(X)
    k = len(X[0])
    b = [0.0] * k
    th = [0.0] * n
    for _ in range(int(iters)):
        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(k):
                p = core.sigmoid(th[i] - b[j])
                num += X[i][j] - p
                den += p * (1.0 - p)
            num -= th[i] / prior_var
            den += 1.0 / prior_var
            if den > 1e-12:
                step = num / den
                th[i] += max(min(step, 1.0), -1.0)
        for j in range(k):
            num = 0.0
            den = 0.0
            for i in range(n):
                p = core.sigmoid(th[i] - b[j])
                num += p - X[i][j]
                den += p * (1.0 - p)
            num -= b[j] / prior_var
            den += 1.0 / prior_var
            if den > 1e-12:
                step = num / den
                b[j] += max(min(step, 1.0), -1.0)
        m = sum(b) / k
        b = [v - m for v in b]
    return b, th


def concurrent_calibration(y, item=None, group=None, anchor=None, iters=200):
    """
    Concurrent calibration with anchor items

    Formula: jointly fit b_F, b_R on the combined sample with anchors

    Both groups are calibrated in ONE run, so the common anchor items
    put every parameter on a single scale with no separate linking
    transformation.  Any drift between the two groups' anchor
    difficulties is therefore an estimate of anchor instability rather
    than of scale: for two groups drawn from the same distribution the
    anchor drift is zero up to estimation error.

    Parameters
    ----------
    y : array-like
        n x k matrix of 0/1 responses, both groups stacked.
    item : array-like or None
        Ignored; the columns are the items.
    group : array-like or None
        Group label per examinee; None treats everyone as one group.
    anchor : array-like or None
        Indices of the anchor items; None uses every item.
    iters : int
        Joint ML cycles.

    Returns
    -------
    result : dict
        Keys: estimate (mean anchor drift), b, b_focal, b_reference,
        drift, theta_mean_focal, theta_mean_reference, n, k, n_anchor.

    References
    ----------
    Wingersky & Lord (1984), Applied Psychological Measurement
    8(3):347-364.
    Kolen & Brennan (2014), Test Equating, Scaling, and Linking, 3rd
    ed., Springer, ch. 6.
    """
    X = core.mat(y)
    n = len(X)
    if n == 0:
        raise ValueError("empty input: y has no rows")
    k = len(X[0])
    if k < 2:
        raise ValueError("need at least two items")
    for r in X:
        if any(v not in (0.0, 1.0) for v in r):
            raise ValueError("responses must be 0/1")
    g = [0] * n if group is None else list(group)
    if len(g) != n:
        raise ValueError("y and group must have the same length")
    anc = list(range(k)) if anchor is None else [int(v) for v in anchor]
    if any(v < 0 or v >= k for v in anc):
        raise ValueError("anchor indices out of range")
    if not anc:
        raise ValueError("at least one anchor item is required")
    b, th = rasch_jmle(X, iters)
    keys = []
    for v in g:
        if v not in keys:
            keys.append(v)
    if len(keys) == 1:
        bf = br = b
        tf = tr = sum(th) / n
        drift = [0.0] * len(anc)
    else:
        fi = [i for i in range(n) if g[i] == keys[0]]
        ri = [i for i in range(n) if g[i] != keys[0]]
        bf, thf = rasch_jmle([X[i] for i in fi], iters)
        br, thr = rasch_jmle([X[i] for i in ri], iters)
        tf = sum(thf) / len(thf)
        tr = sum(thr) / len(thr)
        drift = [bf[j] - br[j] for j in anc]
    return RichResult(payload={
        "estimate": sum(abs(v) for v in drift) / len(drift),
        "b": b,
        "b_focal": bf,
        "b_reference": br,
        "drift": drift,
        "theta_mean_focal": tf,
        "theta_mean_reference": tr,
        "n": n,
        "k": k,
        "n_anchor": len(anc),
        "method": "concurrent calibration with anchor items",
    })


def cheatsheet():
    return "cnsint: concurrent calibration with anchor items"


# compact alias per ledger/NAMING.md
concurrentcalibration = concurrent_calibration
