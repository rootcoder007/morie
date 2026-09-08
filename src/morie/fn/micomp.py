# morie.fn -- function file (rootcoder007/morie)
"""Multiply-imputed Wald test."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["mitest", "mi_compare_models"]


def mitest(theta, U):
    """Multiply-imputed Wald test.

    Multiply-imputed Wald test with the Li et al. (1991) reference df.

    Between-imputation variance B, within-imputation variance Ubar, and
    total T = Ubar + (1 + 1/m) B.  The statistic is referred to an F on
    (k, v) with the small-m denominator degrees of freedom of Li et al.,
    which is what stops a handful of imputations from producing a
    wildly optimistic p-value.  ``theta[i]`` is the k-vector estimate
    from imputation i and ``U[i]`` its k x k covariance.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Multiply-imputed Wald test", payload=_c.mitest(theta=theta, U=U))


mi_compare_models = mitest


def cheatsheet():
    return "micomp: Multiply-imputed Wald test"
