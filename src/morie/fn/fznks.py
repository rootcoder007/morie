# morie.fn -- function file (rootcoder007/morie)
"""Naive kernel-smoothed Kolmogorov-Smirnov statistic (Eq. 5.3)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kernks", "fauzi_naive_kernel_ks"]


def kernks(x, cdf, h=None, ngrid=2001, pad=4.0):
    r"""Naive kernel-smoothed Kolmogorov-Smirnov statistic (Eq. 5.3).

    Eq. (5.3):

    .. math:: \widehat{KS} = \sup_{x\in\mathbb R}|\hat F_X(x) - F(x)|,

    with :math:`\hat F_X` the NAIVE kernel distribution function
    estimator, :math:`n^{-1}\sum_iW((x-X_i)/h)`.

    Because :math:`\hat F_X` is continuous, the supremum is no longer
    attained at a jump, so unlike the empirical version it genuinely needs
    a grid. The grid is fixed and deterministic: ``ngrid`` equally spaced
    points spanning the sample extended by ``pad`` bandwidths on each
    side, since the smoothed estimator has not yet reached 0 and 1 at the
    extreme order statistics.

    This module previously carried a copy of the empirical KS body, which
    ignored the bandwidth entirely. It now smooths.

    Theorem 5.1 says :math:`|KS_n - \widehat{KS}| \to_p 0`, so the
    Kolmogorov critical values still apply; the p-value returned here uses
    them. What smoothing buys is not a different limit but better
    small-sample calibration, per Sec. 5.1.

    Parameters
    ----------
    x : array-like
        Sample.
    cdf : callable
        The null distribution ``F(t)``.
    h : float, optional
        Bandwidth; defaults to the distribution-function rule.
    ngrid : int, default 2001
        Grid size for the supremum; fixed, never adapted.
    pad : float, default 4.0
        How many bandwidths to extend the grid beyond the sample range.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``p_value``, ``argmax``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.3), Theorem 5.1.
    """
    from . import _stats_core as stats
    from ._fauzi import kdfe_bandwidth

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if not callable(cdf):
        raise ValueError("cdf must be a callable F(t).")
    if h is None:
        h = kdfe_bandwidth(xv)
    h = float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    lo = float(np.min(xv)) - float(pad) * h
    hi = float(np.max(xv)) + float(pad) * h
    grid = np.linspace(lo, hi, int(ngrid))
    khat = np.asarray(
        [float(np.mean(stats.norm.cdf((float(t) - xv) / h))) for t in grid], dtype=float
    )
    fv = np.asarray([float(cdf(float(t))) for t in grid], dtype=float)
    diff = np.abs(khat - fv)
    k = int(np.argmax(diff))
    stat = float(diff[k])
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * stat
    pval = 2.0 * float(
        np.sum([(-1) ** (j - 1) * np.exp(-2.0 * j * j * lam * lam) for j in range(1, 101)])
    )
    return RichResult(
        payload={
            "statistic": stat,
            "p_value": float(max(0.0, min(1.0, pval))),
            "argmax": float(grid[k]),
            "h": h,
            "n": int(n),
            "method": "naive kernel-smoothed Kolmogorov-Smirnov statistic (Eq. 5.3)",
        }
    )


fauzi_naive_kernel_ks = kernks


def cheatsheet():
    return "fznks: kernel-smoothed KS: continuous, so the sup needs a fixed grid, not the order statistics"


# CANONICAL TEST
# >>> r = kernks([0.1, 0.3, 0.5, 0.7, 0.9], cdf=lambda t: min(max(t, 0.0), 1.0))
# >>> 0.0 <= r['statistic'] <= 1.0
# True
