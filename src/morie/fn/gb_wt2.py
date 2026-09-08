# morie.fn -- function file (rootcoder007/morie)
"""Exact two-sample runs test with the full null distribution."""

import math

from ._richresult import RichResult

__all__ = ['wwexact', 'gibbons_ww_two_samp_runs']


def wwexact(x, y, tail="left"):
    """Wald-Wolfowitz runs test with an exact p-value from Theorem 3.2.2.

    Section 6.2 (book p. 231) with the null distribution of Section 3.2.
    Unlike the normal approximation this is exact for every m and n:
    the whole pmf of R is built from eq. (3.2.3) and summed over the
    rejection region.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    tail : str, optional
        ``"left"`` (clustering, the usual two-sample alternative),
        ``"right"`` or ``"two-sided"``.

    Returns
    -------
    RichResult
        keys ``statistic`` (R), ``p_value``, ``p_left``, ``p_right``,
        ``support``, ``pmf``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.2, p. 231; eq. (3.2.3), p. 79.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    if tail not in ("left", "right", "two-sided"):
        raise ValueError("tail must be left, right or two-sided.")
    tagged = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tagged.sort(key=lambda p: (p[0], p[1]))
    labels = [t for _, t in tagged]
    r = 1
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            r += 1
    nn = m + n
    den = math.comb(nn, m)
    support = list(range(2, nn + 1))
    pmf = []
    for rr in support:
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * math.comb(m - 1, k - 1) * math.comb(n - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                math.comb(m - 1, k - 1) * math.comb(n - 1, k)
                + math.comb(m - 1, k) * math.comb(n - 1, k - 1)
            )
        pmf.append(p / den)
    left = sum(pmf[i] for i, s in enumerate(support) if s <= r)
    right = sum(pmf[i] for i, s in enumerate(support) if s >= r)
    if tail == "left":
        pv = left
    elif tail == "right":
        pv = right
    else:
        pv = min(1.0, 2.0 * min(left, right))
    return RichResult(
        payload={
            "statistic": int(r),
            "p_value": float(min(1.0, pv)),
            "p_left": float(left),
            "p_right": float(right),
            "support": support,
            "pmf": pmf,
            "m": m,
            "n": n,
            "method": "exact Wald-Wolfowitz runs test (Sec. 6.2, eq. 3.2.3)",
        }
    )


gibbons_ww_two_samp_runs = wwexact
