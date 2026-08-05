# morie.fn -- function file (rootcoder007/morie)
"""Raking ratio estimation by iterative proportional fitting."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["raking_ratio"]


def raking_ratio(y, weights, margins, tol=1e-12, max_iter=200):
    """Force the weighted margins onto known population totals.

    Each pass rescales within one margin, which breaks the previous
    margin, which is why this iterates rather than solving once.  It
    converges whenever the margins are consistent (they must share a
    common total) and the observed cross-classification has no empty
    cell that the targets require to be non-empty; both conditions are
    checked rather than left to diverge silently.

    Formula: repeat over margins A, B, ...:
    ``w_i <- w_i m^A_h / sum_{j in h} w_j`` for every level h of that
    margin, until every margin matches its target.

    Parameters
    ----------
    y : array-like
        Observed values (used only for the raked estimate).
    weights : array-like
        Starting weights, positive.
    margins : sequence of (labels, targets)
        One entry per margin: ``labels`` gives the level of that margin
        for each observation, ``targets`` is a dict level -> population
        total.

    Returns
    -------
    RichResult
        ``estimate`` (the raked weighted mean), ``weights``,
        ``iterations``, ``max_margin_error`` (the largest remaining
        absolute discrepancy), ``N``, ``n``.

    References
    ----------
    Deming, W. E. & Stephan, F. F. (1940).  On a least squares
    adjustment of a sampled frequency table when the expected marginal
    totals are known.  Annals of Mathematical Statistics 11(4):427-444.
    doi:10.1214/aoms/1177731829.
    """
    y = [float(v) for v in C.vec(y)]
    w = [float(v) for v in C.vec(weights)]
    if len(y) == 0:
        raise ValueError("raking_ratio: y is empty")
    if len(w) != len(y):
        raise ValueError("raking_ratio: weights must have one entry per observation")
    for v in w:
        if v <= 0.0:
            raise ValueError("raking_ratio: weights must be positive")
    marg = []
    for item in margins:
        labs, tgt = item[0], item[1]
        labs = [str(v) for v in (labs if isinstance(labs, (list, tuple)) else list(labs))]
        if len(labs) != len(y):
            raise ValueError("raking_ratio: margin labels must have one entry per observation")
        tt = {str(k): float(tgt[k]) for k in tgt}
        for k in labs:
            if k not in tt:
                raise ValueError("raking_ratio: no target for level " + k)
        for k in tt:
            if tt[k] < 0.0:
                raise ValueError("raking_ratio: margin targets must be non-negative")
        marg.append((labs, tt))
    if len(marg) == 0:
        raise ValueError("raking_ratio: at least one margin is required")
    tot0 = None
    for labs, tt in marg:
        t = 0.0
        for k in tt:
            t += tt[k]
        if tot0 is None:
            tot0 = t
        elif abs(t - tot0) > 1e-8 * max(1.0, abs(tot0)):
            raise ValueError("raking_ratio: margins have inconsistent totals")
    it = 0
    err = float("inf")
    for it in range(1, int(max_iter) + 1):
        for labs, tt in marg:
            for k in tt:
                cur = 0.0
                for i in range(len(y)):
                    if labs[i] == k:
                        cur += w[i]
                if cur <= 0.0:
                    if tt[k] > 0.0:
                        raise ValueError("raking_ratio: level " + k +
                                         " has no sampled unit but a positive target")
                    continue
                f = tt[k] / cur
                for i in range(len(y)):
                    if labs[i] == k:
                        w[i] *= f
        err = 0.0
        for labs, tt in marg:
            for k in tt:
                cur = 0.0
                for i in range(len(y)):
                    if labs[i] == k:
                        cur += w[i]
                e = abs(cur - tt[k])
                if e > err:
                    err = e
        if err <= float(tol):
            break
    sw = 0.0
    swy = 0.0
    for i in range(len(y)):
        sw += w[i]
        swy += w[i] * y[i]
    return RichResult(payload={
        "estimate": swy / sw, "weights": w, "iterations": it,
        "max_margin_error": err, "N": sw, "n": len(y),
        "method": "Raking ratio / iterative proportional fitting"})


def cheatsheet():
    return "raklng: Raking ratio post-stratification"
