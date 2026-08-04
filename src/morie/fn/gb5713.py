# morie.fn -- function file (rootcoder007/morie)
"""Simulated power of the signed-rank test from caller-supplied samples."""

import math

from ._richresult import RichResult

__all__ = ['wsrsimpow', 'gibbons_wsrt_simpower']


def wsrsimpow(samples, m0, tcrit):
    """Monte-Carlo power of the signed-rank test over pre-drawn samples.

    Section 5.7.3 (book p. 204): the book's macro draws 1000 samples
    under H1, computes T+ for each, and reports the fraction exceeding
    the critical value from Table H.  The draws are an argument here so
    the estimate is identical in every language given the same input.

    Parameters
    ----------
    samples : sequence of sequence of float
        One row per simulated sample.
    m0 : float
        Hypothesised median.
    tcrit : float
        Rejection region is T+ >= tcrit.

    Returns
    -------
    RichResult
        keys ``power``, ``rejections``, ``nsim``, ``tmean``,
        ``tcrit``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.7.3, p. 204 (Table 5.7.3).
    """
    rows = [[float(v) for v in r] for r in samples]
    nsim = len(rows)
    if nsim < 1:
        raise ValueError("samples must be non-empty.")
    tcrit = float(tcrit)
    ts = []
    for row in rows:
        ds = [v - float(m0) for v in row if v != float(m0)]
        n = len(ds)
        a = [abs(v) for v in ds]
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = mid
            i = j + 1
        ts.append(sum(ranks[i] for i in range(n) if ds[i] > 0.0))
    rej = sum(1 for t in ts if t >= tcrit)
    return RichResult(
        payload={
            "power": rej / nsim,
            "rejections": int(rej),
            "nsim": int(nsim),
            "tmean": sum(ts) / nsim,
            "tcrit": tcrit,
            "method": "simulated signed-rank power over supplied samples",
        }
    )


gibbons_wsrt_simpower = wsrsimpow
