# morie.fn -- function file (rootcoder007/morie)
"""Probability integral transformation."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_pit"]


def gibbons_pit(X, F=None):
    r"""Theorem 2.5.1: if F_X is continuous, :math:`Y = F_X(X)` is
    Uniform(0, 1) -- the probability integral transformation, the
    single fact that makes distribution-free inference possible.

    Applies the transform and reports a K-S check of the transformed
    values against U(0, 1): if the supplied F is the true generator,
    the check should NOT reject.

    Parameters
    ----------
    X : array-like
        Sample.
    F : callable, optional
        The continuous CDF; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``Y`` (transformed values), ``ks_stat``, ``ks_p``
        (against U(0,1)), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.5.1.
    """
    X = np.asarray(X, dtype=float).ravel()
    if X.size < 1:
        raise ValueError("X must be non-empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite.")
    Y = stats.norm.cdf(X) if F is None else np.asarray([F(v) for v in X], dtype=float)
    if np.any((Y < 0) | (Y > 1)):
        raise ValueError("F must map into [0, 1]; check the supplied CDF.")
    ks = stats.kstest(Y, "uniform")
    return RichResult(
        payload={
            "Y": Y, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "n": int(X.size),
            "method": "Y = F(X) ~ U(0,1) for continuous F (Gibbons Theorem 2.5.1)",
        }
    )


def cheatsheet():
    return "gb251: F(X) ~ U(0,1); K-S against uniform as the check"
