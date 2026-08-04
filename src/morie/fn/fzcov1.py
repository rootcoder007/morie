# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the first cumulative-survival estimator with the survival estimator (Eq. 4.16)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["srvcov1", "fauzi_cov_surv_est1"]


def srvcov1(n, surv, cdf=None):
    r"""Covariance of the first cumulative-survival estimator with the survival estimator (Eq. 4.16).

    Eq. (4.16):

    .. math:: \mathrm{Cov}[S_{X,1}(t), \tilde S_X(t)]
              = \tfrac1n S_X(t)F_X(t) + o\!\big(\tfrac hn\big).

    Small, but not negligible: it is exactly the term that survives when
    Theorem 4.3 forms the ratio :math:`m_{X,1} = S_{X,1}/\tilde S_X` and
    linearises it. Dropping it would leave the mean-residual-life variance
    wrong at order :math:`1/n`, which is its leading order.

    The leading term is the SAME as the empirical-df variance
    :math:`F(1-F)/n` written the other way round, since
    :math:`S_X = 1 - F_X`. That is not a coincidence: at leading order
    both estimators are the empirical df with the kernel smoothing only
    entering at :math:`O(h/n)`.

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
    Fauzi and Maesono (2023), Eq. (4.16).
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
            "method": "Cov[S_X,1, tilde S_X] (Eq. 4.16)",
        }
    )


fauzi_cov_surv_est1 = srvcov1


def cheatsheet():
    return "fzcov1: Cov = S F / n -- the term that keeps the MRL variance right at order 1/n (4.16)"


# CANONICAL TEST
# >>> abs(srvcov1(n=100, surv=0.4)['covariance'] - 0.0024) < 1e-15
# True
