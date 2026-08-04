# morie.fn -- function file (rootcoder007/morie)
"""Cramer-von Mises statistic against a fully specified distribution."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["cvmstat", "fauzi_cvm_statistic"]


def cvmstat(x, cdf):
    r"""Cramer-von Mises statistic against a fully specified distribution.

    .. math:: CvM_n = n\!\int_{-\infty}^{\infty}[F_n(x)-F(x)]^2dF(x).

    Evaluated by its exact closed form, not by quadrature: substituting
    the empirical df and integrating gives

    .. math:: CvM_n = \frac1{12n} + \sum_{i=1}^n
              \Big(\frac{2i-1}{2n} - F(X_{(i)})\Big)^2,

    which is a finite sum with no discretisation error at all.

    This module previously carried a Kolmogorov-Smirnov body under the
    Cramer-von Mises name -- one of six modules in this shelf sharing a
    single copied KS implementation. It now computes what it says.

    Where KS uses a supremum and so responds to the single worst point,
    CvM integrates the squared discrepancy against ``dF`` and so responds
    to sustained departure. That is why Theorems 5.1 and 5.7 have to be
    proved separately: the two statistics are not functions of one
    another, and smoothing affects them differently.

    Parameters
    ----------
    x : array-like
        Sample.
    cdf : callable
        The fully specified null distribution ``F(t)``.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``p_value``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.2) and the display defining CvM_n in Sec. 5.1.
    """
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}.")
    if not callable(cdf):
        raise ValueError("cdf must be a callable F(t).")
    fv = np.asarray([float(cdf(float(t))) for t in xs], dtype=float)
    target = (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
    stat = float(1.0 / (12.0 * n) + np.sum((target - fv) ** 2))
    # Anderson-Darling-style asymptotic tail of the Cramer-von Mises law,
    # summed over a FIXED 100 terms so the value is reproducible.
    z = stat
    pval = 1.0
    if z > 0:
        acc = 0.0
        for k in range(100):
            acc += float(np.exp(-((4.0 * k + 1.0) ** 2) * np.pi ** 2 / (8.0 * z)))
        pval = max(0.0, min(1.0, 1.0 - acc * float(np.sqrt(2.0 / z))))
    return RichResult(
        payload={
            "statistic": stat,
            "p_value": float(pval),
            "n": int(n),
            "method": "Cramer-von Mises statistic against a specified F",
        }
    )


fauzi_cvm_statistic = cvmstat


def cheatsheet():
    return "fzcvms: CvM by its exact finite-sum form; previously this module held a copied KS body"


# CANONICAL TEST
# >>> r = cvmstat([0.1, 0.3, 0.5, 0.7, 0.9], cdf=lambda t: min(max(t, 0.0), 1.0))
# >>> abs(r['statistic'] - 1 / 60) < 1e-12
# True
