# morie.fn -- function file (rootcoder007/morie)
"""Two-sample Kolmogorov-Smirnov test D_{m,n}."""

import math

from ._richresult import RichResult

__all__ = ['ks2', 'gibbons_ks2']


def _ks2count(m, n, d, onesided=False):
    """Exact P(statistic >= d) by counting monotone lattice paths."""
    # Number of paths from (0,0) to (m,n) that never violate the bound.
    row = [0.0] * (n + 1)
    prev = [0.0] * (n + 1)
    lim = d - 1e-12
    for i in range(m + 1):
        for j in range(n + 1):
            dev = i / m - j / n
            if not onesided:
                dev = abs(dev)
            if dev >= lim:
                row[j] = 0.0
            elif i == 0 and j == 0:
                row[j] = 1.0
            else:
                v = prev[j] if i > 0 else 0.0
                v += row[j - 1] if j > 0 else 0.0
                row[j] = v
        prev = row[:]
    total = math.comb(m + n, m)
    inside = prev[n]
    return max(0.0, min(1.0, 1.0 - inside / total))


def ks2(x, y):
    """Two-sample KS statistic and its exact null tail probability.

    Section 6.3 (book p. 239):

    .. math:: D_{m,n} = \\sup_x |S_m(x) - S_n(x)|,

    distribution-free under H0: F_X = F_Y.  The exact tail is obtained
    by counting the monotone lattice paths from (0,0) to (m,n) that
    stay strictly inside the band |i/m - j/n| < d and dividing by
    C(m+n, m) -- the standard exact evaluation, with no simulation.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, sizes m and n.

    Returns
    -------
    RichResult
        keys ``statistic``, ``p_value``, ``dplus``, ``dminus``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.3, p. 239.
    """
    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    pts = sorted(set(xs + ys))
    dp = dm = 0.0
    for t in pts:
        sm = sum(1 for v in xs if v <= t) / m
        sn = sum(1 for v in ys if v <= t) / n
        dp = max(dp, sm - sn)
        dm = max(dm, sn - sm)
    d = max(dp, dm)
    return RichResult(
        payload={
            "statistic": float(d),
            "p_value": float(_ks2count(m, n, d, False)),
            "dplus": float(dp),
            "dminus": float(dm),
            "m": m,
            "n": n,
            "method": "two-sample KS test, exact lattice-path tail",
        }
    )


gibbons_ks2 = ks2
