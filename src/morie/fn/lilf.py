# morie.fn -- function file (rootcoder007/morie)
"""Lilliefors test (KS with estimated parameters)."""

from typing import Union

from . import _array_core as np
import scipy.stats as stats

from ._containers import TestResult


def _ks_stat_sorted(z, cdf_vals):
    n = z.size
    hi = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    return float(max(np.max(hi - cdf_vals), np.max(cdf_vals - lo)))


def lilliefors_test(
    x: Union[list, np.ndarray],
    n_mc: int = 2000,
    seed: int = 0,
) -> TestResult:
    """
    Lilliefors test for normality.

    A modification of the Kolmogorov-Smirnov test where the mean and
    standard deviation are estimated from the data rather than
    specified a priori. Fitting the parameters pulls the reference CDF
    toward the sample, so the observed D is systematically SMALLER
    than under a fully specified null; classical KS p-values applied
    to it are therefore too large and the test under-rejects. The
    Lilliefors null distribution fixes this, and it is parameter-free
    for a fitted location-scale family, so it can be simulated
    exactly: draw standard normal samples of the same n, refit, and
    recompute D.

    The p-value here is that Monte Carlo p-value (2000 replicates by
    default), not the classical KS approximation this module used to
    report.

    :param x: Sample data (1-D array-like, n >= 4).
    :param n_mc: Monte Carlo replicates for the null distribution.
    :param seed: Seed for the Monte Carlo.
    :return: TestResult with D statistic and p_value.
    :raises ValueError: If x has fewer than 4 observations.

    References
    ----------
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
        with mean and variance unknown. Journal of the American Statistical
        Association, 62(318), 399-402.
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 4:
        raise ValueError("Lilliefors test requires at least 4 observations.")

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

    if sigma == 0:
        return TestResult(
            test_name="Lilliefors",
            statistic=0.0,
            p_value=1.0,
            method="Lilliefors test (zero variance)",
            n=n,
        )

    z = np.sort(arr)
    D = _ks_stat_sorted(z, stats.norm.cdf(z, loc=mu, scale=sigma))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_mc)):
        s = np.sort(rng.standard_normal(n))
        d = _ks_stat_sorted(s, stats.norm.cdf(s, loc=s.mean(), scale=s.std(ddof=1)))
        if d >= D:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_mc)

    return TestResult(
        test_name="Lilliefors",
        statistic=float(D),
        p_value=float(p_value),
        method="Lilliefors test (Monte Carlo null, fitted parameters)",
        n=n,
        extra={"mu_hat": mu, "sigma_hat": sigma, "n_mc": int(n_mc)},
    )


lilf = lilliefors_test


def cheatsheet() -> str:
    return "lilliefors_test({}) -> Lilliefors test (KS with estimated parameters)."
