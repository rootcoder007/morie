"""Surface retrieval from sparse sensors."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["surface_retrieval"]


def surface_retrieval(coords, values, grid, method="gp", kernel=None, noise=0.0):
    """
    Interpolate a scattered surface onto a grid by GP regression.

    Formula: posterior mean of a GP conditioned on the observations

    Verified against Rasmussen & Williams (2006) eq. (2.23)-(2.24) --
    source consulted. Simple kriging with a known covariance and a zero
    mean is algebraically the same predictor as the GP posterior mean;
    the ``method`` argument selects the label reported, not a different
    computation, and that equivalence is exactly why both names appear
    in the literature.

    Parameters
    ----------
    coords : nested sequence
        Observation locations, ``n x d``.
    values : array-like
        Observed values.
    grid : nested sequence
        Target locations, ``m x d``.
    method : str, optional
        ``"gp"`` or ``"kriging"``; both compute the same predictor.
    kernel : sequence or callable, optional
        ``(sf, l)`` for the squared-exponential covariance.
    noise : float, optional
        Nugget standard deviation.

    Returns
    -------
    RichResult
        Keys: estimate (predicted surface), variance, method_used,
        n, method.

    References
    ----------
    Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Eq. (2.23)-(2.24).
    """
    if method not in ("gp", "kriging"):
        raise ValueError("method must be 'gp' or 'kriging'")
    A = _big2.mat(coords)
    B = _big2.mat(grid)
    zv = [float(t) for t in np.atleast_1d(np.asarray(values, dtype=float))]
    n = len(A)
    if len(zv) != n:
        raise ValueError("values must have one entry per row of coords")
    kf = _big2.sekernel(kernel)
    sn2 = float(noise) ** 2
    K = [[kf(A[i], A[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    alpha = _big2.solve(K, zv)
    pred = []
    var = []
    for p in range(len(B)):
        ks = [kf(B[p], A[i]) for i in range(n)]
        pred.append(sum(ks[i] * alpha[i] for i in range(n)))
        w = _big2.solve(K, ks)
        var.append(kf(B[p], B[p]) - sum(ks[i] * w[i] for i in range(n)))
    return RichResult(
        payload={
            "estimate": pred,
            "variance": var,
            "method_used": method,
            "n": n,
            "method": "Surface interpolation by GP posterior mean -- Rasmussen & Williams (2006) eq. (2.23)",
        }
    )


def cheatsheet():
    return "sfcret: Surface retrieval from sparse sensors"
