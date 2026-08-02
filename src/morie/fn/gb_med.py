# morie.fn -- function file (rootcoder007/morie)
"""Exact distribution of the sample median."""

from . import _array_core as np
from scipy import special

from ._richresult import RichResult

__all__ = ["gibbons_median_dist"]


def gibbons_median_dist(x, n, F=None):
    r"""Section 2.7.1: for odd n = 2m + 1, the sample median is the
    (m+1)th order statistic, so its CDF is the order-statistic form

    .. math:: F_{\mathrm{med}}(x) = I_{F(x)}(m + 1,\; m + 1),

    the regularised incomplete beta with both parameters m + 1 --
    symmetric in F(x) about 1/2, which is why the median of a sample
    from a symmetric law is itself symmetrically distributed.

    Parameters
    ----------
    x : float or array-like
        Evaluation point(s).
    n : int
        Sample size; must be odd for the exact single-order-statistic
        form (even n has the two-statistic average, which this
        function refuses rather than approximating silently).
    F : callable, optional
        Parent CDF; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``cdf``, ``m``, ``beta_params``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 2.7.1.
    """
    from . import _stats_core as stats

    n = int(n)
    if n < 1 or n % 2 == 0:
        raise ValueError(
            f"the single-order-statistic form needs odd n, got {n}; even n "
            "medians average two order statistics and need the joint law."
        )
    m = (n - 1) // 2
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    Fv = stats.norm.cdf(xv) if F is None else np.asarray([F(v) for v in xv], dtype=float)
    cdf = special.betainc(m + 1, m + 1, Fv)
    scalar = np.isscalar(x) or np.ndim(x) == 0
    return RichResult(
        payload={
            "cdf": float(cdf[0]) if scalar else cdf, "m": m,
            "beta_params": (m + 1, m + 1), "n": n,
            "method": "F_med(x) = I_F(x)(m+1, m+1), odd n (Gibbons Ch. 2.7.1)",
        }
    )


def cheatsheet():
    return "gb_med: median CDF = I_F(m+1, m+1); refuses even n"
