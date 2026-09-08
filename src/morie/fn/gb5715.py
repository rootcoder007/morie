# morie.fn -- function file (rootcoder007/morie)
"""Walsh-average confidence interval for the median."""

import math

from ._richresult import RichResult

__all__ = ['wsrci', 'gibbons_wsrt_ci']


def _wsrnull(n):
    """Exact null pmf of T+ over 0..N(N+1)/2 by subset-sum counting."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for t in range(total, r - 1, -1):
            counts[t] += counts[t - r]
    denom = float(2**n)
    return [c / denom for c in counts]


def wsrci(x, tcrit):
    """Confidence interval for M from the ordered Walsh averages.

    Book p. 207-209, eq. (5.7.16) and Table 5.7.5.  With the
    N(N+1)/2 Walsh averages (X_i + X_k)/2, i <= k, sorted, the
    endpoints are the (t_{alpha/2} + 1)-th values in from each end, and
    the exact confidence coefficient is 1 - 2 P(T+ <= t_{alpha/2}).

    The book's example: the eight values 1, 2, 3, 4, 5, 6, 9, 13 with
    t_{alpha/2} = 3 give (1.5, 9.0) at exact level 0.961.

    Parameters
    ----------
    x : sequence of float
        Sample, n >= 2.
    tcrit : int
        Left-tail critical value t_{alpha/2} from Table H.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``coverage``, ``tail``,
        ``nwalsh``, ``n``, ``tcrit``, ``estimate`` (median of the
        Walsh averages), ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.7.5, pp. 207-209.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    tcrit = int(tcrit)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if tcrit < 0:
        raise ValueError("tcrit must be non-negative.")
    walsh = sorted(
        (xs[i] + xs[k]) / 2.0 for i in range(n) for k in range(i, n)
    )
    nw = len(walsh)
    if tcrit + 1 > nw:
        raise ValueError("tcrit too large for this sample size.")
    pmf = _wsrnull(n)
    tail = sum(pmf[: tcrit + 1])
    mid = nw // 2
    est = walsh[mid] if nw % 2 else (walsh[mid - 1] + walsh[mid]) / 2.0
    return RichResult(
        payload={
            "lower": walsh[tcrit],
            "upper": walsh[nw - 1 - tcrit],
            "coverage": float(1.0 - 2.0 * tail),
            "tail": float(tail),
            "nwalsh": int(nw),
            "n": n,
            "tcrit": tcrit,
            "estimate": float(est),
            "method": "Walsh-average CI, (t+1)-th from each end (Sec. 5.7.5)",
        }
    )


gibbons_wsrt_ci = wsrci
