# morie.fn -- function file (rootcoder007/morie)
"""Ties in the Wald-Wolfowitz runs test: the range of attainable R."""

import math

from ._richresult import RichResult

__all__ = ['wwties', 'gibbons_ww2_ties']


def wwties(x, y):
    """Smallest and largest R over all resolutions of the ties.

    Section 6.2.1 (book p. 233).  When an X ties a Y the pooled
    ordering is not determined, and the book's remedy is to break the
    tie at random -- which makes the statistic itself random.  The
    deterministic content of that advice is the pair of bounds: within
    each tied group the labels can be arranged to minimise or to
    maximise the run count, giving R_min (the least favourable, most
    conservative value) and R_max.  If R_min and R_max fall on the same
    side of the critical value the tie-breaking cannot change the
    decision, which is the only case in which the randomisation is
    safely ignorable.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``rmin``, ``rmax``, ``nties`` (tied groups containing both
        labels), ``ambiguous`` (1 if rmin != rmax), ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.2.1, p. 233.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    vals = sorted(set(xs + ys))
    groups = []
    for v in vals:
        a = sum(1 for t in xs if t == v)
        b = sum(1 for t in ys if t == v)
        groups.append((a, b))
    nties = sum(1 for a, b in groups if a > 0 and b > 0)

    def _runs(seq):
        r = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                r += 1
        return r

    # minimise: keep each group's labels blocked, ordered to continue the
    # previous run; maximise: alternate labels within each group.
    lo = []
    for a, b in groups:
        if a and b:
            if lo and lo[-1] == 0:
                lo.extend([0] * a + [1] * b)
            else:
                lo.extend([1] * b + [0] * a)
        else:
            lo.extend([0] * a + [1] * b)
    hi = []
    for a, b in groups:
        if a and b:
            i = j = 0
            start = 1 if (hi and hi[-1] == 0) else 0
            while i < a or j < b:
                if start == 0 and i < a:
                    hi.append(0)
                    i += 1
                elif j < b:
                    hi.append(1)
                    j += 1
                elif i < a:
                    hi.append(0)
                    i += 1
                start = 1 - start
        else:
            hi.extend([0] * a + [1] * b)
    rmin = _runs(lo)
    rmax = _runs(hi)
    return RichResult(
        payload={
            "rmin": int(rmin),
            "rmax": int(rmax),
            "nties": int(nties),
            "ambiguous": int(rmin != rmax),
            "m": m,
            "n": n,
            "method": "runs-test tie bounds over all tie resolutions",
        }
    )


gibbons_ww2_ties = wwties
