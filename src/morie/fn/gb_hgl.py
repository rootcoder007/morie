# morie.fn -- function file (rootcoder007/morie)
"""Hodges-Lehmann one-sample location estimator from Walsh averages."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_hodges_lehmann"]


def gibbons_hodges_lehmann(x, alpha=0.05):
    r"""Median of the Walsh averages, with a distribution-free interval.

    The Walsh averages are the :math:`N(N+1)/2` pairwise means

    .. math:: W_{ik} = \frac{X_i + X_k}{2}, \qquad 1 \le i \le k \le N,

    and their median is the Hodges-Lehmann estimator of the population
    median. Including the :math:`i = k` terms is not a detail: those
    are the observations themselves, and dropping them changes the
    estimator.

    This is the location estimator that belongs with the Wilcoxon
    signed-rank test in the way the sample mean belongs with the
    :math:`t` test, and its efficiency relative to the mean is
    :math:`3/\pi \approx 0.955` at the normal -- the price of
    distribution-free validity is under five per cent, and it is
    unbounded better than the mean at heavy tails.

    The interval is the pair of order statistics of the Walsh averages
    determined by the signed-rank distribution, so its coverage is
    exact and free of any distributional assumption beyond symmetry.
    Because the Walsh averages are discrete, the attained coverage
    generally EXCEEDS the nominal level, and the attained value is
    returned rather than the requested one.

    Parameters
    ----------
    x : array-like, shape (n,)
        Sample, assumed drawn from a distribution symmetric about its
        median.
    alpha : float
        Nominal two-sided level for the interval.

    Returns
    -------
    RichResult
        ``estimate``, ``ci``, ``coverage`` (attained), ``n_walsh``,
        ``walsh_averages``, ``median`` (the ordinary sample median for
        comparison).

    References
    ----------
    Gibbons and Chakraborti (2011), *Nonparametric Statistical
    Inference*, 5th ed., section 5.7, pp. 194-198 (Walsh averages and
    the estimator of the median).
    Hodges and Lehmann (1963), *Annals of Mathematical Statistics*
    34:598-611.

    Examples
    --------
    >>> gibbons_hodges_lehmann([1, 2, 5, 6, 9, 13])["estimate"]
    5.5
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 1:
        raise ValueError("need at least 1 observation.")
    if np.any(~np.isfinite(v)):
        raise ValueError("x contains non-finite values.")
    i, k = np.triu_indices(n, k=0)          # i <= k, so i == k is kept
    walsh = np.sort((v[i] + v[k]) / 2.0)
    m = walsh.size
    est = float(np.median(walsh))

    # signed-rank based interval: find the largest u with
    # P(T+ <= u-1) <= alpha/2 under the null, by exact enumeration for
    # small n and the normal approximation beyond it
    lo = hi = None
    cov = np.nan
    if n >= 5:
        u = _signed_rank_cut(n, alpha)
        if 1 <= u <= m // 2:
            lo, hi = float(walsh[u - 1]), float(walsh[m - u])
            cov = 1.0 - 2.0 * _signed_rank_cdf(n, u - 1)
    return RichResult(
        payload={
            "estimate": est,
            "hodges_lehmann": est,
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "ci": (lo, hi),
            "ci_lower": lo,
            "ci_upper": hi,
            "coverage": float(cov),
            "coverage_note": (
                "attained coverage, which exceeds the nominal level because "
                "the Walsh averages are discrete"
            ),
            "alpha": float(alpha),
            "walsh_averages": walsh,
            "n_walsh": int(m),
            "walsh_note": (
                "the N(N+1)/2 averages include the i = k terms, which are "
                "the observations themselves; omitting them is a different "
                "estimator"
            ),
            "efficiency_note": (
                "asymptotic efficiency 3/pi = 0.955 relative to the mean at "
                "the normal, and unbounded better under heavy tails"
            ),
            "n": int(n),
            "method": "Hodges-Lehmann one-sample estimator (Walsh averages)",
        }
    )


def _signed_rank_counts(n):
    """Exact null distribution of the Wilcoxon signed-rank statistic."""
    total = n * (n + 1) // 2
    dp = np.zeros(total + 1)
    dp[0] = 1.0
    for r in range(1, n + 1):
        dp[r:] += dp[:-r] if r <= total else 0
    return dp


def _signed_rank_cdf(n, t):
    if t < 0:
        return 0.0
    dp = _signed_rank_counts(n)
    return float(dp[: int(t) + 1].sum() / 2.0 ** n)


def _signed_rank_cut(n, alpha):
    if n > 30:
        # normal approximation with continuity correction
        mu = n * (n + 1) / 4.0
        sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z(1 - alpha / 2)
        return max(int(np.floor(mu - z * sd)) + 1, 1)
    dp = _signed_rank_counts(n)
    cum = np.cumsum(dp) / 2.0 ** n
    idx = np.nonzero(cum <= alpha / 2.0)[0]
    return int(idx[-1]) + 1 if idx.size else 1


def _z(q):
    import math
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "gb_hgl: Hodges-Lehmann median as the median of the N(N+1)/2 Walsh "
        "averages, with an exact signed-rank interval"
    )
