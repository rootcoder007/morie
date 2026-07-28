# morie.fn -- function file (rootcoder007/morie)
"""Taylor linearization variance for nonlinear estimators."""

import numpy as np

from ._richresult import RichResult

__all__ = ["taylor_linearization"]


def taylor_linearization(y, weights, grad, cov=None):
    r"""Taylor (delta-method) linearisation:

    .. math:: \operatorname{Var}\big(g(\hat\theta)\big)
              \approx (\nabla g)'\,\Sigma\,(\nabla g).

    Survey estimators are routinely nonlinear -- ratios, regression
    coefficients, medians -- and no closed-form design variance
    exists for them. Linearisation replaces the estimator by its
    first-order expansion, whose variance is a linear-combination
    variance the design DOES supply.

    Two limits are worth stating rather than discovering. The
    approximation is first order, so it misses the curvature that
    dominates in small samples; and it needs :math:`g` to be
    differentiable at the estimate, which rules out quantiles, where
    replication methods are used instead. ``valid_for`` records both.

    Parameters
    ----------
    y : array-like, shape (n, p)
        The estimating-function contributions.
    weights : array-like, shape (n,)
        Design weights.
    grad : array-like, shape (p,)
        Gradient of ``g`` at the estimate.
    cov : array-like, optional
        Covariance of the components; the weighted sandwich is used
        otherwise.

    Returns
    -------
    RichResult
        keys: ``variance``, ``se``, ``cov``, ``grad``,
        ``first_order_only`` (True), ``valid_for``, ``n``, ``p``,
        ``method``.
    """
    from ._survey import check_weights, linearise

    Y = np.atleast_2d(np.asarray(y, dtype=float))
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    n, p = Y.shape
    w = check_weights(weights, n)
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    if g.size != p:
        raise ValueError(f"grad has {g.size} entries for {p} components.")
    if cov is None:
        m = (w[:, None] * Y).sum(axis=0) / w.sum()
        C = ((w[:, None] * (Y - m)).T @ (Y - m)) / (w.sum() ** 2) * n
    else:
        C = np.atleast_2d(np.asarray(cov, dtype=float))
    var = linearise(g, C)
    return RichResult(payload={
        "variance": var, "se": float(np.sqrt(max(var, 0.0))),
        "cov": C, "grad": g, "first_order_only": True,
        "valid_for": "differentiable functionals only; quantiles need "
                     "replication methods, and the first-order approximation "
                     "misses curvature in small samples",
        "n": int(n), "p": int(p),
        "method": "Taylor linearisation; a linear stand-in whose design variance is available"})


def cheatsheet():
    return "taylor: first order only, and undefined for quantiles -- replication covers those"
