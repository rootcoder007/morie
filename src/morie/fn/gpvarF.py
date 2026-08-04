"""GP posterior variance."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["gp_variance"]


def gp_variance(X, X_star, kernel=None, sigma2=0.0):
    """
    Predictive variance of a Gaussian process.

    Formula: V_* = k_** - K_* (K + sn^2 I)^{-1} K_*'

    Verified against Rasmussen & Williams (2006) eq. (2.24) p. 16 and
    eq. (2.26) p. 17 -- source consulted. As the book notes, this
    variance does not depend on the observed targets at all, only on
    the inputs.

    Parameters
    ----------
    X : nested sequence
        Training inputs, ``n x d``.
    X_star : nested sequence
        Test inputs, ``m x d``.
    kernel : sequence or callable, optional
        ``(sf, l)`` for the squared-exponential kernel, or a callable.
    sigma2 : float, optional
        Observation noise VARIANCE ``sn^2`` (not the standard
        deviation; the name follows the book).

    Returns
    -------
    RichResult
        Keys: estimate (predictive variances), prior (the prior
        variances ``k_**``), n, method.

    References
    ----------
    Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Eq. (2.24), (2.26).
    """
    A = _big2.mat(X)
    B = _big2.mat(X_star)
    n = len(A)
    kf = _big2.sekernel(kernel)
    s2 = float(sigma2)
    K = [[kf(A[i], A[j]) + (s2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    var = []
    prior = []
    for p in range(len(B)):
        ks = [kf(B[p], A[i]) for i in range(n)]
        w = _big2.solve(K, ks)
        prior.append(kf(B[p], B[p]))
        var.append(prior[-1] - sum(ks[i] * w[i] for i in range(n)))
    return RichResult(
        payload={
            "estimate": var,
            "prior": prior,
            "n": n,
            "method": "GP predictive variance -- Rasmussen & Williams (2006) eq. (2.24)",
        }
    )


def cheatsheet():
    return "gpvarF: GP posterior variance"


# compact alias per ledger/NAMING.md
gpvariance = gp_variance
