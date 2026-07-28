# morie.fn -- function file (rootcoder007/morie)
"""Hodges-Lehmann two-sample shift estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_hodges_lehmann_2"]


def gibbons_hodges_lehmann_2(x, y, alpha=0.05):
    r"""Median of the :math:`mn` differences :math:`Y_j - X_i`.

    Gibbons and Chakraborti state it directly: "the median of the
    differences :math:`Y_j - X_i` is called the Hodges-Lehmann
    estimator of the shift parameter :math:`\theta` and this is an
    unbiased estimator of :math:`\theta` in the shift model", where
    :math:`\theta` is read as the difference of medians
    :math:`M_Y - M_X`.

    The shift model is the assumption doing the work. It says the two
    distributions have the SAME SHAPE and differ only by a translation;
    under it a single number describes the difference. When the shapes
    differ -- one arm more variable, or skewed the other way -- there
    is no single shift, and this estimator returns the median
    difference of a randomly paired observation, which is a different
    and usually less interesting quantity. ``shift_plausible`` compares
    the two spreads as a cheap check rather than leaving the assumption
    unexamined.

    The interval endpoints are order statistics of the same
    :math:`mn` differences, indexed by the Mann-Whitney null
    distribution, so coverage is exact and distribution-free.

    Parameters
    ----------
    x, y : array-like
        The two samples. The estimate is of ``y``'s location minus
        ``x``'s.
    alpha : float
        Nominal two-sided level.

    Returns
    -------
    RichResult
        ``estimate``, ``ci``, ``coverage``, ``n_differences``,
        ``median_difference``, ``shift_plausible``.

    References
    ----------
    Gibbons and Chakraborti (2011), *Nonparametric Statistical
    Inference*, 5th ed., section 6.6, p. 268.
    Hodges and Lehmann (1963), *Annals of Mathematical Statistics*
    34:598-611.

    Examples
    --------
    >>> gibbons_hodges_lehmann_2([1, 6, 7], [2, 4, 9, 10, 12])["estimate"]
    3.0
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    m, n = a.size, b.size
    if m < 1 or n < 1:
        raise ValueError(
            "both samples must be non-empty, got %d and %d." % (m, n)
        )
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("samples contain non-finite values.")
    diffs = np.sort((b[:, None] - a[None, :]).ravel())
    est = float(np.median(diffs))

    lo = hi = None
    cov = np.nan
    if m >= 3 and n >= 3:
        u = _mw_cut(m, n, alpha)
        if 1 <= u <= diffs.size // 2:
            lo, hi = float(diffs[u - 1]), float(diffs[diffs.size - u])
            cov = 1.0 - 2.0 * _mw_cdf(m, n, u - 1)

    sa, sb = float(np.std(a, ddof=1)) if m > 1 else 0.0, \
        float(np.std(b, ddof=1)) if n > 1 else 0.0
    ratio = (max(sa, sb) / min(sa, sb)) if min(sa, sb) > 0 else np.inf
    return RichResult(
        payload={
            "estimate": est,
            "hodges_lehmann": est,
            "median_difference": float(np.median(b) - np.median(a)),
            "mean_difference": float(np.mean(b) - np.mean(a)),
            "ci": (lo, hi),
            "ci_lower": lo,
            "ci_upper": hi,
            "coverage": float(cov),
            "coverage_note": (
                "attained coverage; the differences are discrete so it "
                "exceeds the nominal level"
            ),
            "alpha": float(alpha),
            "n_differences": int(diffs.size),
            "sd_ratio": float(ratio),
            "shift_plausible": bool(ratio < 2.0),
            "shift_note": (
                "the shift model assumes the two distributions differ only "
                "by a translation; when the spreads differ markedly there "
                "is no single shift to estimate and this number is the "
                "median difference of a random pairing instead"
            ),
            "n_x": int(m),
            "n_y": int(n),
            "n": int(m + n),
            "method": "Hodges-Lehmann two-sample shift estimator",
        }
    )


def _mw_counts(m, n):
    """Exact null distribution of the Mann-Whitney U statistic.

    Condition on the largest of the m + n observations: if it is an x it
    contributes n to U, if a y it contributes nothing. That gives

        f(m, n, u) = f(m-1, n, u-n) + f(m, n-1, u)

    with f(0, n, 0) = f(m, 0, 0) = 1. Getting the offset onto the wrong
    index produces a table that still sums to C(m+n, m) but is not
    symmetric about mn/2 -- which is the cheap check, and the one that
    catches it.
    """
    top = m * n
    table = np.zeros((m + 1, n + 1, top + 1))
    table[0, :, 0] = 1.0
    table[:, 0, 0] = 1.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            for u in range(top + 1):
                v = table[i, j - 1, u]
                if u - j >= 0:
                    v += table[i - 1, j, u - j]
                table[i, j, u] = v
    return table[m, n]


def _mw_cdf(m, n, u):
    if u < 0:
        return 0.0
    from math import comb
    cnt = _mw_counts(m, n)
    return float(cnt[: int(u) + 1].sum() / comb(m + n, m))


def _mw_cut(m, n, alpha):
    from math import comb
    if m * n > 400:
        mu = m * n / 2.0
        sd = np.sqrt(m * n * (m + n + 1) / 12.0)
        return max(int(np.floor(mu - 1.959963984540054 * sd)) + 1, 1)
    cnt = _mw_counts(m, n)
    cum = np.cumsum(cnt) / comb(m + n, m)
    idx = np.nonzero(cum <= alpha / 2.0)[0]
    return int(idx[-1]) + 1 if idx.size else 1


def cheatsheet():
    return (
        "gb_hg2: Hodges-Lehmann shift as the median of the mn differences, "
        "with an exact Mann-Whitney interval and a shift-model check"
    )
