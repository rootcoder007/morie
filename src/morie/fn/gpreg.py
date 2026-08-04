"""Gaussian process regression with squared-exponential kernel."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["gaussian_process_regression"]


def gaussian_process_regression(X, y, X_test, kernel=None, noise=0.0):
    """
    Gaussian-process regression with a squared-exponential kernel.

    Formula: f ~ GP(0, k); k(x,x') = sf^2 exp(-||x-x'||^2 / (2 l^2))

    Verified against Rasmussen & Williams (2006), *Gaussian Processes
    for Machine Learning*, eq. (2.23)-(2.24) p. 16 for the predictive
    mean and covariance, and p. 19 for the squared-exponential
    covariance ``k_y = sf^2 exp(-(1/2 l^2)(x_p - x_q)^2) + sn^2 d_pq``
    -- source consulted (the book's free PDF at gaussianprocess.org).

    Solved by a single linear solve against ``K + sn^2 I``; no
    iteration, so the result is deterministic.

    Parameters
    ----------
    X : nested sequence
        Training inputs, ``n x d``.
    y : array-like
        Training targets, length ``n``.
    X_test : nested sequence
        Test inputs, ``m x d``.
    kernel : sequence or callable, optional
        ``(sf, l)`` for the squared-exponential kernel (default
        ``(1, 1)``), or a callable ``k(x1, x2)`` taking two points.
    noise : float, optional
        Observation noise standard deviation ``sn`` (default 0).

    Returns
    -------
    RichResult
        Keys: estimate (predictive means), variance, loglik, n, method.
        ``loglik`` is the log marginal likelihood of eq. (2.30).

    References
    ----------
    Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Eq. (2.23)-(2.24), (2.30).
    """
    A = _big2.mat(X)
    B = _big2.mat(X_test)
    yv = [float(t) for t in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(A)
    if len(yv) != n:
        raise ValueError("y must have one entry per row of X")
    kf = _big2.sekernel(kernel)
    sn2 = float(noise) ** 2
    K = [[kf(A[i], A[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    Ks = [[kf(B[p], A[i]) for i in range(n)] for p in range(len(B))]
    alpha = _big2.solve(K, yv)
    mean = [sum(Ks[p][i] * alpha[i] for i in range(n)) for p in range(len(B))]
    var = []
    for p in range(len(B)):
        w = _big2.solve(K, Ks[p])
        var.append(kf(B[p], B[p]) - sum(Ks[p][i] * w[i] for i in range(n)))
    sgn, logdet = _big2.slogdet(K)
    ll = -0.5 * sum(yv[i] * alpha[i] for i in range(n)) - 0.5 * logdet - 0.5 * n * float(np.log(2.0 * np.pi))
    return RichResult(
        payload={
            "estimate": mean,
            "variance": var,
            "loglik": ll,
            "n": n,
            "method": "GP regression, SE kernel -- Rasmussen & Williams (2006) eq. (2.23)-(2.24)",
        }
    )


def cheatsheet():
    return "gpreg: Gaussian process regression with squared-exponential kernel"
