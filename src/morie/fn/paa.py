# morie.fn -- function file (rootcoder007/morie)
"""Piecewise aggregate approximation."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["paa"]


def paa(x, N):
    """Reduce a length-n series to N segment means.

    Formula (Keogh et al. 2001):

        xbar_i = (N / n) sum_{j = n(i-1)/N + 1}^{n i / N} x_j.

    When ``N`` divides ``n`` this is the mean of each of ``N`` equal
    blocks.  When it does not, the sum is over a FRACTIONAL window:
    the observation straddling a segment boundary contributes to both
    segments in proportion to the overlap.  Truncating instead of
    splitting is the usual implementation error and it breaks the one
    property the representation is supposed to have -- that the segment
    means average back to the mean of the series.  That identity is
    asserted as an anchor.

    Parameters
    ----------
    x : array-like, shape (n,)
        The series.
    N : int
        Number of segments, ``1 <= N <= n``.

    Returns
    -------
    RichResult
        ``paa`` (the N segment means), ``estimate`` (the first one),
        ``segment_width`` (``n / N``), ``N``, ``n``.

    References
    ----------
    Keogh, E., Chakrabarti, K., Pazzani, M. & Mehrotra, S. (2001).
    Dimensionality reduction for fast similarity search in large time
    series databases.  Knowledge and Information Systems, 3(3),
    263--286.  doi:10.1007/PL00011669
    """
    v = C.vec(x)
    n = len(v)
    k = int(N)
    if n == 0:
        raise ValueError("paa: x is empty")
    if k < 1 or k > n:
        raise ValueError("paa: N must satisfy 1 <= N <= n")
    w = n / float(k)
    out = []
    for i in range(k):
        lo = i * w
        hi = (i + 1) * w
        s = 0.0
        j = int(lo)
        while j < n and j < hi:
            a = lo if lo > j else float(j)
            b = hi if hi < j + 1 else float(j + 1)
            if b > a:
                s += (b - a) * v[j]
            j += 1
        out.append(s / w)
    return RichResult(payload={
        "paa": out, "estimate": out[0], "segment_width": w,
        "N": k, "n": n,
        "method": "Piecewise aggregate approximation"})


def cheatsheet():
    return "paa: Piecewise aggregate approximation"
