# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P7: the law of crime concentration and crime-free places
(``research/lean/P7Concentration.lean``, ``P7Mixture.lean``, ``P7Distinct.lean``).

* ``Research.P7.gini_zero_decomposition``: G = z + (1 - z) G_+
* ``Research.P7.poisson_zero_prob``: P(Poisson(mu) = 0) = exp(-mu)
* ``Research.P7.Mixture.variance_eq`` / ``mixture_var_ge_mean`` / ``mixture_var_eq_mean_iff`` / ``mixture_zero_ge_exp_neg_mean``
* ``Research.P7.expectedDistinct_bounds`` (``log_le_S``, ``S_le_log``)

R parity: ``rmorie`` ``R/concentration.R``.
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn._sci_core import brentq

__all__ = ["concentration_gini", "concentration_decompose", "concentration_dispersion", "concentration_distinct_growth"]


def _counts(x, min_len=1):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.shape[0] < min_len or np.any(np.isnan(x)) or np.any(x < 0) or float(x.sum()) <= 0:
        raise ValueError("x must be non-negative counts with a positive total")
    return x


def concentration_gini(x) -> float:
    """Gini coefficient of a count vector (mean-absolute-difference form).

    Examples
    --------
    >>> round(concentration_gini([0, 0, 0, 1, 9]), 12)
    0.76
    """
    x = _counts(x)
    n = x.shape[0]
    xs = np.sort(x)
    i = np.arange(1, n + 1, dtype=float)
    return float(2 * np.sum(i * xs) / (n * np.sum(xs)) - (n + 1) / n)


def concentration_decompose(x) -> dict:
    """Decompose crime concentration into crime-free places and the rest (``gini_zero_decomposition``).

    Examples
    --------
    >>> r = concentration_decompose([0, 0, 0, 1, 9])
    >>> (r["zero_share"], round(r["gini_positive"], 12), round(r["identity_check"], 15))
    (0.6, 0.4, 0.0)
    """
    x = _counts(x)
    n = x.shape[0]
    z = float(np.mean(x == 0))
    g_all = concentration_gini(x)
    pos = x[x > 0]
    g_pos = concentration_gini(pos) if pos.shape[0] > 1 else 0.0
    mu = float(np.mean(x))
    z_null = math.exp(-mu)
    return {
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
    }


def concentration_dispersion(x) -> dict:
    """Dispersion of place counts against the Poisson null and its mixtures.

    Examples
    --------
    >>> r = concentration_dispersion([0, 0, 1, 3, 0, 2])
    >>> (r["mean_count"], round(r["dispersion_index"], 12))
    (1.0, 1.6)
    """
    x = _counts(x, min_len=2)
    n = x.shape[0]
    mu = float(np.mean(x))
    v = float(np.sum((x - mu) ** 2) / (n - 1))
    z = float(np.mean(x == 0))
    z_null = math.exp(-mu)
    return {
        "mean_count": mu,
        "variance": v,
        "dispersion_index": v / mu,
        "implied_intensity_variance": max(v - mu, 0.0),
        "implied_intensity_sd": math.sqrt(max(v - mu, 0.0)),
        "zero_share": z,
        "null_zero_share": z_null,
        "zero_share_gap": z - z_null,
        "theorems": [
            "Research.P7.Mixture.variance_eq",
            "Research.P7.Mixture.mixture_var_ge_mean",
            "Research.P7.Mixture.mixture_var_eq_mean_iff",
            "Research.P7.Mixture.mixture_zero_ge_exp_neg_mean",
        ],
    }


def concentration_distinct_growth(place) -> dict:
    """Growth of the number of distinct places under Polya allocation (``expectedDistinct_bounds``).

    Examples
    --------
    >>> r = concentration_distinct_growth([1, 2, 1, 3, 2, 1, 4, 1])
    >>> (r["n"], r["distinct"][-1], round(r["M_hat"], 6))
    (8, 4, 2.500624)
    """
    place = [str(p) for p in place]
    n = len(place)
    if n < 2:
        raise ValueError("need at least two events")
    seen = set()
    distinct = []
    for p in place:
        seen.add(p)
        distinct.append(len(seen))
    K = distinct[-1]

    def expect_fun(M, n):
        return M * sum(1 / (M + i) for i in range(n))

    if n <= K:
        M_hat = float("inf")
    elif K <= 1:
        M_hat = 0.0
    else:
        M_hat = float(brentq(lambda M: expect_fun(M, n) - K, 1e-8, 1e8, xtol=1e-14, maxiter=500))
    finite = math.isfinite(M_hat) and M_hat > 0
    idx = list(range(1, n + 1))
    if finite:
        expected = []
        acc = 0.0
        for i in range(n):
            acc += 1 / (M_hat + i)
            expected.append(M_hat * acc)
        lower = [M_hat * math.log((M_hat + i) / M_hat) for i in idx]
        upper = [1 + M_hat * math.log((M_hat + i - 1) / M_hat) for i in idx]
    else:
        expected = lower = upper = [float("nan")] * n
    half = [i for i in idx if i >= n / 2]
    slope = float("nan")
    if len(half) > 2 and all(distinct[i - 1] > 0 for i in half):
        xs = [math.log(i) for i in half]
        ys = [math.log(distinct[i - 1]) for i in half]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((a - mx) ** 2 for a in xs)
        slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx if sxx > 0 else float("nan")
    return {
        "n": n,
        "distinct": distinct,
        "M_hat": M_hat,
        "expected": expected,
        "lower": lower,
        "upper": upper,
        "loglog_slope": slope,
        "theorems": ["Research.P7.log_le_S", "Research.P7.S_le_log", "Research.P7.expectedDistinct_bounds"],
    }
