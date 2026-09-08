# morie.fn -- function file (rootcoder007/morie)
"""Hodges-Lehmann estimator and its identity with the Walsh-average counts."""

import math

from ._richresult import RichResult

__all__ = ['hlwsrlink', 'gibbons_hodges_wilcoxon']


def hlwsrlink(x, m0=0.0):
    """Median of the Walsh averages, and the T+ counting identity.

    Book p. 209-210.  The Hodges-Lehmann estimator of the median is
    the median of the N(N+1)/2 Walsh averages (X_i + X_k)/2, i <= k,
    and the book's counting identity is exact:

        T+ = #{Walsh averages > M0} + (1/2) #{Walsh averages = M0},
        T- = #{Walsh averages < M0} + (1/2) #{Walsh averages = M0}.

    The book's own illustration (N = 8, M0 = 4.5) gives 16 averages
    below, 2 equal and 18 above, hence T- = 17 and T+ = 19.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    m0 : float, optional
        Hypothesised median (default 0).

    Returns
    -------
    RichResult
        keys ``estimate`` (Hodges-Lehmann), ``tplus``, ``tminus``,
        ``nbelow``, ``nequal``, ``nabove``, ``nwalsh``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.7.5, pp. 209-210.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    m0 = float(m0)
    walsh = sorted(
        (xs[i] + xs[k]) / 2.0 for i in range(n) for k in range(i, n)
    )
    nw = len(walsh)
    below = sum(1 for w in walsh if w < m0)
    equal = sum(1 for w in walsh if w == m0)
    above = nw - below - equal
    mid = nw // 2
    est = walsh[mid] if nw % 2 else (walsh[mid - 1] + walsh[mid]) / 2.0
    return RichResult(
        payload={
            "estimate": float(est),
            "tplus": float(above + equal / 2.0),
            "tminus": float(below + equal / 2.0),
            "nbelow": int(below),
            "nequal": int(equal),
            "nabove": int(above),
            "nwalsh": int(nw),
            "n": n,
            "method": "Hodges-Lehmann median of Walsh averages",
        }
    )


gibbons_hodges_wilcoxon = hlwsrlink
