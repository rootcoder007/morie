"""GP residual modeling."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["gp_residual_kernel"]


def gp_residual_kernel(X, y, y_pred, kernel=None, noise=0.0):
    """
    Gaussian process fitted to the residuals of a parametric fit.

    Formula: y - yhat_param ~ GP(0, k)

    Verified against Rasmussen & Williams (2006), Section 2.7 (the
    explicit-basis / semi-parametric model, eq. (2.39)-(2.41)), where a
    fixed mean function is subtracted and the GP models what is left --
    source consulted. Here the parametric fit is supplied by the caller
    as ``y_pred`` rather than estimated, which is the form the residual
    kriging literature uses.

    Parameters
    ----------
    X : nested sequence
        Inputs, ``n x d``.
    y : array-like
        Observed targets.
    y_pred : array-like
        Parametric fitted values at the same inputs.
    kernel : sequence or callable, optional
        ``(sf, l)`` for the squared-exponential kernel, or a callable.
    noise : float, optional
        Observation noise standard deviation.

    Returns
    -------
    RichResult
        Keys: estimate (the smoothed residuals at the training inputs),
        residual, fitted (parametric + smoothed), loglik, n, method.

    References
    ----------
    Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Sec. 2.7, eq. (2.23).
    """
    A = _big2.mat(X)
    yv = [float(t) for t in np.atleast_1d(np.asarray(y, dtype=float))]
    pv = [float(t) for t in np.atleast_1d(np.asarray(y_pred, dtype=float))]
    n = len(A)
    if len(yv) != n or len(pv) != n:
        raise ValueError("y and y_pred must have one entry per row of X")
    r = [yv[i] - pv[i] for i in range(n)]
    kf = _big2.sekernel(kernel)
    sn2 = float(noise) ** 2
    K = [[kf(A[i], A[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    alpha = _big2.solve(K, r)
    smooth = [sum(kf(A[p], A[i]) * alpha[i] for i in range(n)) for p in range(n)]
    sgn, logdet = _big2.slogdet(K)
    ll = -0.5 * sum(r[i] * alpha[i] for i in range(n)) - 0.5 * logdet - 0.5 * n * float(np.log(2.0 * np.pi))
    return RichResult(
        payload={
            "estimate": smooth,
            "residual": r,
            "fitted": [pv[i] + smooth[i] for i in range(n)],
            "loglik": ll,
            "n": n,
            "method": "GP on parametric residuals -- Rasmussen & Williams (2006) Sec. 2.7",
        }
    )


def cheatsheet():
    return "gprsk: GP residual modeling"
