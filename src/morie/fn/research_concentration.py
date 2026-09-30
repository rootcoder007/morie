# morie.fn -- function file (rootcoder007/morie)
"""Research P7: the law of crime concentration and crime-free places.

Python twin of ``R/concentration.R``; machine-checked in
``research/lean/P7Concentration.lean`` and its mixture and Polya files:

- ``Research.P7.gini_zero_decomposition``: G(all places) = z + (1 - z) G(places with any crime)
- ``Research.P7.poisson_zero_prob``: P(Poisson(mu) = 0) = exp(-mu)
- ``Research.P7.Mixture.*``: a Poisson mixture has variance >= mean and zero share >= exp(-mean)
- ``Research.P7.expectedDistinct_bounds``: Polya allocation grows the number of
  distinct places logarithmically
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = [
    "concentration_gini",
    "concentration_decompose",
    "concentration_dispersion",
    "concentration_distinct_growth",
]


def _counts(x, min_len=1, msg="x must be non-negative counts with a positive total"):
    try:
        v = [float(a) for a in x]
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if len(v) < min_len or any(math.isnan(a) or a < 0 for a in v) or math.fsum(v) <= 0:
        raise ValueError(msg)
    return v


def concentration_gini(x):
    """Gini coefficient of a count vector.

    Mean-absolute-difference form ``sum_i sum_j |x_i - x_j| / (2 n^2 mean(x))``,
    computed in O(n log n) from the sorted counts.

    Parameters
    ----------
    x : sequence of float
        Non-negative counts; at least one positive.

    Returns
    -------
    float
        The Gini coefficient in [0, 1).

    Examples
    --------
    >>> concentration_gini([0, 0, 0, 1, 9])
    0.76
    >>> round(concentration_gini([3, 1, 4, 1, 5, 9, 2, 6]), 12)
    0.366935483871
    """
    xs = sorted(_counts(x))
    n = len(xs)
    return 2 * math.fsum((i + 1) * v for i, v in enumerate(xs)) / (n * math.fsum(xs)) - (n + 1) / n


def concentration_decompose(x):
    """Decompose crime concentration into crime-free places and the rest.

    Splits the Gini of all places exactly into the zero share ``z`` and the
    Gini among places with any crime, ``G = z + (1 - z) G+``, and reports
    the zero share ``exp(-mu)`` that a uniform Poisson null would produce
    at the observed mean count.

    Parameters
    ----------
    x : sequence of float
        Non-negative counts per place; at least one positive.

    Returns
    -------
    RichResult
        ``n``, ``mean_count``, ``zero_share``, ``gini_all``,
        ``gini_positive``, ``identity_check`` (zero up to rounding),
        ``null_zero_share``, ``null_gini_same_positive``,
        ``excess_zero_share`` and ``theorems``.

    Examples
    --------
    >>> d = concentration_decompose([0, 0, 0, 1, 9, 0, 2, 0])
    >>> d.zero_share, round(d.gini_all, 12), round(d.gini_positive, 12)
    (0.625, 0.791666666667, 0.444444444444)
    >>> round(d.null_zero_share, 12), abs(d.identity_check) < 1e-15
    (0.223130160148, True)
    """
    v = _counts(x)
    n = len(v)
    z = sum(1 for a in v if a == 0) / n
    g_all = concentration_gini(v)
    pos = [a for a in v if a > 0]
    g_pos = concentration_gini(pos) if len(pos) > 1 else 0.0
    mu = math.fsum(v) / n
    z_null = math.exp(-mu)
    return RichResult(
        title="Crime concentration: zero share and Gini decomposition",
        payload={
            "n": n,
            "mean_count": mu,
            "zero_share": z,
            "gini_all": g_all,
            "gini_positive": g_pos,
            "identity_check": g_all - (z + (1 - z) * g_pos),
            "null_zero_share": z_null,
            "null_gini_same_positive": z_null + (1 - z_null) * g_pos,
            "excess_zero_share": z - z_null,
            "theorems": [
                "Research.P7.gini_zero_decomposition",
                "Research.P7.poisson_zero_prob",
                "Research.P7.Mixture.mixture_zero_ge_exp_neg_mean",
            ],
        },
    )


def concentration_dispersion(x):
    """Over-dispersion and excess zeros of place counts against a Poisson null.

    Under a Poisson mixture (places with heterogeneous intensities) the
    count variance is the mean plus the intensity variance, so the
    dispersion index var/mean is at least one with equality only for a
    common intensity, and the zero share is at least ``exp(-mean)``.

    Parameters
    ----------
    x : sequence of float
        At least two non-negative counts with a positive total.

    Returns
    -------
    RichResult
        ``mean_count``, ``variance`` (sample, divisor n - 1),
        ``dispersion_index``, ``implied_intensity_variance``,
        ``implied_intensity_sd``, ``zero_share``, ``null_zero_share``,
        ``zero_share_gap`` and ``theorems``.

    Examples
    --------
    >>> d = concentration_dispersion([0, 0, 0, 1, 9, 0, 2, 0])
    >>> d.mean_count, round(d.variance, 12), round(d.dispersion_index, 12)
    (1.5, 9.714285714286, 6.47619047619)
    """
    v = _counts(x, 2, "x must be at least two non-negative counts with a positive total")
    n = len(v)
    mu = math.fsum(v) / n
    var = math.fsum((a - mu) ** 2 for a in v) / (n - 1)
    z = sum(1 for a in v if a == 0) / n
    z_null = math.exp(-mu)
    return RichResult(
        title="Dispersion of place counts",
        payload={
            "mean_count": mu,
            "variance": var,
            "dispersion_index": var / mu,
            "implied_intensity_variance": max(var - mu, 0.0),
            "implied_intensity_sd": math.sqrt(max(var - mu, 0.0)),
            "zero_share": z,
            "null_zero_share": z_null,
            "zero_share_gap": z - z_null,
            "theorems": [
                "Research.P7.Mixture.variance_eq",
                "Research.P7.Mixture.mixture_var_ge_mean",
                "Research.P7.Mixture.mixture_var_eq_mean_iff",
                "Research.P7.Mixture.mixture_zero_ge_exp_neg_mean",
            ],
        },
    )


def _expected_distinct(M, n):
    return M * math.fsum(1.0 / (M + k) for k in range(n))


def _solve_m(n, K):
    """Root of M * sum_{k<n} 1/(M+k) = K on [1e-8, 1e8] by bisection to machine precision."""
    lo, hi = 1e-8, 1e8
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _expected_distinct(mid, n) < K:
            lo = mid
        else:
            hi = mid
    flo, fhi = _expected_distinct(lo, n) - K, _expected_distinct(hi, n) - K
    return lo if abs(flo) <= abs(fhi) else hi


def concentration_distinct_growth(place):
    """Growth of the number of distinct places under preferential allocation.

    Under Polya (Dirichlet-process) allocation with concentration ``M`` the
    expected number of distinct places after ``n`` events is
    ``M sum_{i<n} 1/(M+i)``, squeezed between ``M log((M+n)/M)`` and
    ``1 + M log((M+n-1)/M)``: logarithmic growth. Reports the observed
    growth curve, the concentration matching the final count, the proved
    envelope for it and the log-log slope over the second half of the
    series (near 0 for logarithmic growth, positive for a power law).

    Parameters
    ----------
    place : sequence
        Place identifiers in event order.

    Returns
    -------
    RichResult
        ``n``, ``distinct``, ``M_hat``, ``expected``, ``lower``, ``upper``,
        ``loglog_slope`` and ``theorems``.

    Examples
    --------
    >>> g = concentration_distinct_growth([1, 1, 2, 1, 3, 2, 1, 4, 1, 2, 2, 5])
    >>> g.distinct
    [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5]
    >>> round(g.M_hat, 9), round(g.expected[-1], 9), round(g.loglog_slope, 9)
    (2.684548138, 5.0, 0.657976742)
    """
    place = list(place)
    n = len(place)
    if n < 2:
        raise ValueError("need at least two events")
    seen = set()
    distinct = []
    for p in place:
        seen.add(p)
        distinct.append(len(seen))
    K = distinct[-1]
    if n <= K:
        M_hat = math.inf
    elif K <= 1:
        M_hat = 0.0
    else:
        M_hat = _solve_m(n, K)
    idx = range(1, n + 1)
    if math.isfinite(M_hat) and M_hat > 0:
        expected = []
        s = c = 0.0  # Neumaier running sum, as close to R's long-double cumsum as a float can get
        for k in range(n):
            t = 1.0 / (M_hat + k)
            u = s + t
            c += (s - u) + t if abs(s) >= abs(t) else (t - u) + s
            s = u
            expected.append(M_hat * (s + c))
        lower = [M_hat * math.log((M_hat + i) / M_hat) for i in idx]
        upper = [1 + M_hat * math.log((M_hat + i - 1) / M_hat) for i in idx]
    else:
        expected = lower = upper = [math.nan] * n
    half = [i for i in idx if i >= n / 2]
    if len(half) > 2:
        lx = [math.log(i) for i in half]
        ly = [math.log(distinct[i - 1]) for i in half]
        mx = math.fsum(lx) / len(lx)
        my = math.fsum(ly) / len(ly)
        sxx = math.fsum((a - mx) ** 2 for a in lx)
        slope = math.fsum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    else:
        slope = math.nan
    return RichResult(
        title="Distinct-place growth under Polya allocation",
        payload={
            "n": n,
            "distinct": distinct,
            "M_hat": M_hat,
            "expected": expected,
            "lower": lower,
            "upper": upper,
            "loglog_slope": slope,
            "theorems": ["Research.P7.log_le_S", "Research.P7.S_le_log", "Research.P7.expectedDistinct_bounds"],
        },
    )


def cheatsheet() -> str:
    return (
        "concentration_gini(x) -> Gini of place counts\n"
        "concentration_decompose(x) -> G = z + (1-z) G+, Poisson-null zero share exp(-mu) (Research P7)\n"
        "concentration_dispersion(x) -> var/mean, implied intensity variance, excess zeros\n"
        "concentration_distinct_growth(place) -> distinct-place curve, Polya M_hat, proved log envelope"
    )
