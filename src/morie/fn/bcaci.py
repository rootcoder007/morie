# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bias-corrected and accelerated (BCa) bootstrap confidence interval."""

from __future__ import annotations

from collections.abc import Callable

from . import _array_core as np
from . import _stats_core as _st

from ._containers import DescriptiveResult


def bca_ci(
    data: np.ndarray, stat_fn: Callable, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42, cdf=None
) -> DescriptiveResult:
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Adjusts percentile endpoints for bias and skewness of the bootstrap
    distribution.

    :param data: 1-D data array.
    :type data: numpy.ndarray
    :param stat_fn: Function that takes a 1-D array and returns a scalar.
    :type stat_fn: Callable
    :param n_boot: Number of bootstrap replicates. Default 2000.
    :type n_boot: int
    :param alpha: Significance level (two-sided). Default 0.05.
    :type alpha: float
    :param seed: Random seed. Default 42.
    :type seed: int
    :return: DescriptiveResult with CI bounds.
    :rtype: DescriptiveResult

    References
    ----------
    Efron B. (1987). Better bootstrap confidence intervals. *Journal of
    the American Statistical Association*, 82(397), 171-185.
    """
    data = np.asarray(data, dtype=float).ravel()
    n = len(data)
    rng = np.random.default_rng(seed)
    theta_hat = float(stat_fn(data))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)
    # ponytail: one BCa in the tree.  The bias correction (D&H 5.22), the
    # jackknife acceleration (D&H 5.27) and the endpoint transform (D&H 5.21)
    # live in btbca and are called from here rather than written twice -- a
    # second copy would agree with the first at 1e-9 forever and be
    # indistinguishable from correct work.
    from .btbca import boot_bca_ci

    _bca = boot_bca_ci(theta_hat, boot_stats, data, stat_fn, alpha)
    z0 = _bca["z0"]
    a = _bca["accel"]
    a1 = _bca["alpha_lo"]
    a2 = _bca["alpha_hi"]
    sorted_boot = np.sort(boot_stats)
    lo = float(sorted_boot[max(0, int(np.floor(a1 * n_boot)))])
    hi = float(sorted_boot[min(n_boot - 1, int(np.ceil(a2 * n_boot)) - 1)])
    return DescriptiveResult(
        name="bca_ci",
        value=theta_hat,
        extra={
            "lower": lo,
            "upper": hi,
            "estimate": theta_hat,
            "bias_correction": float(z0),
            "acceleration": a,
            "alpha": alpha,
        },
    )


bcaci = bca_ci


def cheatsheet() -> str:
    return "bca_ci({}) -> BCa bootstrap confidence interval."


# compact alias per ledger/NAMING.md
bcaci = bca_ci
