# morie.fn -- function file (rootcoder007/morie)
"""ADWIN adaptive windowing for change detection."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["adwin"]


def adwin(x, delta=0.05):
    """Adaptive window over a stream, shrunk when the mean shifts.

    ADWIN keeps a window W of the most recent observations and, after
    every arrival, drops observations from the tail while some split
    W = W0 . W1 shows a mean difference larger than the Hoeffding-based
    threshold.  With n0, n1 the two part lengths and n = n0 + n1,

        m       = 1 / (1/n0 + 1/n1)      (harmonic mean of n0 and n1)
        delta'  = delta / n
        eps_cut = sqrt( (1 / (2 m)) * ln(4 / delta') )

    and the window shrinks while |mean(W0) - mean(W1)| >= eps_cut for some
    split, W1 holding the more recent items.

    Parameters
    ----------
    x : array-like
        Stream values, expected to lie in [0, 1] (rescale otherwise, as
        the bound is a Hoeffding bound on a bounded variable).
    delta : float
        Confidence parameter in (0, 1).

    Returns
    -------
    RichResult
        ``mean``, ``width``, ``window``, ``ndrops``, ``lastcut``,
        ``changepoints``, ``n``, ``delta``.

    References
    ----------
    Bifet, A. and Gavalda, R. (2007), "Learning from time-changing data
    with adaptive windowing", Proceedings of the 2007 SIAM International
    Conference on Data Mining, 443-448.  Section 3 defines m as the
    harmonic mean of n0 and n1, delta' = delta/n and eps_cut =
    sqrt(ln(4/delta') / (2m)); Figure 1 is the ADWIN loop reproduced here.
    Read from the authors' own PDF at www.cs.upc.edu/~gavalda.
    """
    x = C.vec(x)
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    W = []
    drops = 0
    cuts = []
    last = float("nan")
    for pos, v in enumerate(x):
        W.append(v)
        shrunk = True
        while shrunk and len(W) >= 2:
            shrunk = False
            n = len(W)
            pre = [0.0]
            for w in W:
                pre.append(pre[-1] + w)
            for n0 in range(1, n):
                n1 = n - n0
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                dp = delta / n
                cut = math.sqrt(math.log(4.0 / dp) / (2.0 * m))
                d = abs(pre[n0] / n0 - (pre[n] - pre[n0]) / n1)
                if d >= cut:
                    W = W[1:]
                    drops += 1
                    last = cut
                    cuts.append(pos)
                    shrunk = True
                    break
    n = len(W)
    return RichResult(payload={
        "mean": sum(W) / n if n else float("nan"), "width": n,
        "window": W, "ndrops": drops, "lastcut": last,
        "changepoints": cuts, "n": len(x), "delta": delta,
        "method": "ADWIN adaptive windowing (Bifet-Gavalda 2007 Sect. 3)"})


def cheatsheet():
    return "adwin: ADWIN adaptive windowing for change detection."
