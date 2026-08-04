# morie.fn -- function file (rootcoder007/morie)
"""Two-sided median test with both tails of the hypergeometric null."""

import math

from ._richresult import RichResult

__all__ = ['medtest2', 'gibbons_median_test_2sided']


def medtest2(x, y, alpha=0.05):
    """Median test against a two-sided alternative, with the exact region.

    Section 6.4 (book p. 247).  Large U says the X's sit above the
    combined median, small U that they sit below; the two-sided test
    combines both tails.  The exact critical values at level alpha are
    found from the hypergeometric null (never the nominal alpha, which
    a discrete statistic cannot attain), and the realised size is
    reported.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    alpha : float, optional
        Nominal two-sided level (default 0.05).

    Returns
    -------
    RichResult
        keys ``statistic``, ``p_value``, ``lower``, ``upper``
        (critical values), ``alpha_exact``, ``t``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.4, p. 247.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    alpha = float(alpha)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    pooled = sorted(xs + ys)
    nn = m + n
    med = (
        pooled[nn // 2]
        if nn % 2
        else (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    )
    t = sum(1 for v in pooled if v > med)
    u = sum(1 for v in xs if v > med)

    def _p(k):
        if k < 0 or k > m or t - k < 0 or t - k > n:
            return 0.0
        return math.comb(m, k) * math.comb(n, t - k) / math.comb(nn, t)

    lo = float("nan")
    al = 0.0
    acc = 0.0
    for k in range(0, m + 1):
        acc += _p(k)
        if acc <= alpha / 2.0:
            lo = float(k)
            al = acc
        else:
            break
    hi = float("nan")
    au = 0.0
    acc = 0.0
    for k in range(m, -1, -1):
        acc += _p(k)
        if acc <= alpha / 2.0:
            hi = float(k)
            au = acc
        else:
            break
    lower = sum(_p(k) for k in range(0, u + 1))
    upper = sum(_p(k) for k in range(u, m + 1))
    return RichResult(
        payload={
            "statistic": int(u),
            "p_value": float(min(1.0, 2.0 * min(lower, upper))),
            "lower": lo,
            "upper": hi,
            "alpha_exact": float(al + au),
            "t": int(t),
            "m": m,
            "n": n,
            "method": "two-sided median test, exact hypergeometric region",
        }
    )


gibbons_median_test_2sided = medtest2
