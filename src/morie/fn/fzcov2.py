# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the second cumulative-survival estimator with the survival estimator (Eq. 4.22)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["srvcov2", "fauzi_cov_surv_est2"]


def srvcov2(n, surv, cdf=None):
    r"""Covariance of the second cumulative-survival estimator with the survival estimator (Eq. 4.22).

    Eq. (4.22):

    .. math:: \mathrm{Cov}[S_{X,2}(t), \tilde S_X(t)]
              = \tfrac1n S_X(t)F_X(t) + o\!\big(\tfrac hn\big),

    the same expression as (4.16) for :math:`S_{X,1}`.

    That the two covariances coincide, and Theorem 4.2's variance
    coincides with Theorem 4.1's, is why Theorem 4.3 can give ONE variance
    formula covering :math:`m_{X,1}` and :math:`m_{X,2}` together while
    giving them separate bias formulas. The estimators differ in bias
    only.

    Kept as its own entry point rather than aliased to
    :func:`morie.fn.fzcov1.srvcov1`, because the equality is a THEOREM --
    a result of the transformation argument in Sec. 4.1, not a definition
    -- and collapsing the two would hide that.

    Parameters
    ----------
    n : int
        Sample size.
    surv : float
        ``S_X(t)``.
    cdf : float, optional
        ``F_X(t)``; defaults to ``1 - surv``.

    Returns
    -------
    RichResult
        Keys ``covariance``, ``surv``, ``cdf``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (4.22).
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    s = float(surv)
    f = 1.0 - s if cdf is None else float(cdf)
    return RichResult(
        payload={
            "covariance": float(s * f / n),
            "surv": s,
            "cdf": f,
            "n": n,
            "method": "Cov[S_X,2, tilde S_X] (Eq. 4.22)",
        }
    )


fauzi_cov_surv_est2 = srvcov2


def cheatsheet():
    return "fzcov2: Cov = S F / n again -- equal to (4.16) by theorem, not by definition (4.22)"


# CANONICAL TEST
# >>> abs(srvcov2(n=100, surv=0.4)['covariance'] - 0.0024) < 1e-15
# True
