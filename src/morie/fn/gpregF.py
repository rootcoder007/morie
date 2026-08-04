"""Gaussian process regression."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["gp_regression"]


def gp_regression(X, y, X_star, kernel=None, noise=0.0):
    """
    Posterior mean of a Gaussian process at new inputs.

    Formula: fbar_* = K_* (K + sn^2 I)^{-1} y

    Verified against Rasmussen & Williams (2006) eq. (2.23) p. 16 and
    its single-test-point form eq. (2.25) p. 17 -- source consulted.
    This is the mean only; ``gp_variance`` gives eq. (2.24).

    Parameters
    ----------
    X : nested sequence
        Training inputs, ``n x d``.
    y : array-like
        Training targets.
    X_star : nested sequence
        Test inputs, ``m x d``.
    kernel : sequence or callable, optional
        ``(sf, l)`` for the squared-exponential kernel, or a callable.
    noise : float, optional
        Observation noise standard deviation.

    Returns
    -------
    RichResult
        Keys: estimate (posterior means), weights (the vector ``alpha``
        of eq. (2.25)), n, method.

    References
    ----------
    Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Eq. (2.23), (2.25).
    """
    A = _big2.mat(X)
    B = _big2.mat(X_star)
    yv = [float(t) for t in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(A)
    if len(yv) != n:
        raise ValueError("y must have one entry per row of X")
    kf = _big2.sekernel(kernel)
    sn2 = float(noise) ** 2
    K = [[kf(A[i], A[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    alpha = _big2.solve(K, yv)
    mean = [sum(kf(B[p], A[i]) * alpha[i] for i in range(n)) for p in range(len(B))]
    return RichResult(
        payload={
            "estimate": mean,
            "weights": alpha,
            "n": n,
            "method": "GP posterior mean K_*(K+sn^2 I)^-1 y -- Rasmussen & Williams (2006) eq. (2.25)",
        }
    )


def cheatsheet():
    return "gpregF: Gaussian process regression"


# compact alias per ledger/NAMING.md
gpregression = gp_regression
