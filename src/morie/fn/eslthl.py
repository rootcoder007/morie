# morie.fn -- function file (rootcoder007/morie)
"""Thin-plate spline -- ESL Sec 5.7."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_thin_plate_spline"]


def esl_thin_plate_spline(X, y, lambda_=1.0, newdata=None):
    r"""Fit a two-dimensional thin-plate smoothing spline.

    Minimises the penalised criterion

    .. math::
        \sum_i (y_i - f(x_i))^2 + \lambda J(f), \qquad
        J(f) = \iint \left(f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2\right)\,dx\,dy,

    whose exact minimiser is finite-dimensional -- a radial basis expansion
    in :math:`\eta(r) = r^2 \log r` centred at the data points, plus a linear
    null-space term:

    .. math::
        f(x) = \beta_0 + \beta^\top x + \sum_j \delta_j\, \eta(\lVert x - x_j\rVert).

    That the infinite-dimensional problem has a finite-dimensional solution
    is the content of ESL Sec 5.7, and it is why this is solved exactly by
    linear algebra rather than by numerical optimisation.

    :math:`J` annihilates linear functions, so as :math:`\lambda \to \infty`
    the fit tends to the least-squares *plane*, not to a constant. At
    :math:`\lambda = 0` it interpolates.

    Parameters
    ----------
    X : array-like
        Locations ``(n, 2)``.
    y : array-like
        Observed values ``(n,)``.
    lambda_ : float
        Smoothing parameter, non-negative.
    newdata : array-like, optional
        Locations to predict at. Defaults to ``X``.

    Returns
    -------
    RichResult
        ``fitted`` at the evaluation points, ``delta`` (RBF weights),
        ``beta`` (the linear part), ``edf`` (effective degrees of freedom),
        ``residuals``, ``gcv``.

    References
    ----------
    Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in
        Sobolev spaces. In *Constructive Theory of Functions of Several
        Variables* (pp. 85-100). Springer.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    With no penalty the spline interpolates the observations exactly.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> P = rng.uniform(-1, 1, (25, 2))
    >>> z = np.sin(2 * P[:, 0]) + P[:, 1] ** 2
    >>> r = esl_thin_plate_spline(P, z, lambda_=0.0)
    >>> bool(np.max(np.abs(r["residuals"])) < 1e-6)
    True

    Heavy smoothing collapses onto the least-squares plane -- not onto a
    constant, because the penalty does not see linear functions.

    >>> big = esl_thin_plate_spline(P, z, lambda_=1e8)
    >>> A = np.column_stack([np.ones(25), P])
    >>> plane = A @ np.linalg.lstsq(A, z, rcond=None)[0]
    >>> bool(np.max(np.abs(big["fitted"] - plane)) < 1e-3)
    True

    More smoothing means fewer effective degrees of freedom.

    >>> bool(big["edf"] < r["edf"])
    True

    >>> esl_thin_plate_spline(P, z, lambda_=-1.0)
    Traceback (most recent call last):
        ...
    ValueError: lambda_ must be non-negative
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
    if X.shape[1] != 2:
        raise ValueError(f"thin-plate splines here are 2-D; X has {X.shape[1]} columns")
    if n < 3:
        raise ValueError("need at least 3 points to fit the linear null space")

    E = _tps_kernel(X, X)
    A = np.column_stack([np.ones(n), X])

    # Solve on an orthonormal basis of null(A') rather than forming the
    # saddle-point system directly. With a large lambda the bordered matrix has
    # condition number ~lambda, and lstsq's relative cutoff then truncates the
    # O(1) linear block outright -- the fit stops converging to the
    # least-squares plane and drifts instead.
    Q, _ = np.linalg.qr(A, mode="complete")
    Q2 = Q[:, 3:]                                   # basis for null(A'), n x (n-3)
    Elam = E + lambda_ * np.eye(n)
    gamma = np.linalg.solve(Q2.T @ Elam @ Q2, Q2.T @ y)
    delta = Q2 @ gamma
    beta = np.linalg.lstsq(A, y - Elam @ delta, rcond=None)[0]

    Z = X if newdata is None else np.atleast_2d(np.asarray(newdata, dtype=float))
    if Z.shape[1] != 2:
        raise ValueError("newdata must have 2 columns")
    fitted = _tps_kernel(Z, X) @ delta + np.column_stack([np.ones(Z.shape[0]), Z]) @ beta

    train = E @ delta + A @ beta
    resid = y - train
    S = np.linalg.lstsq(E + lambda_ * np.eye(n), E, rcond=None)[0]
    edf = float(np.trace(S)) + 3.0
    rss = float(np.sum(resid**2))
    gcv = n * rss / max((n - edf) ** 2, 1e-12)
    return RichResult(
        title="Thin-plate spline",
        summary_lines=[("n", n), ("lambda", float(lambda_)), ("edf", edf)],
        payload={
            "fitted": fitted, "delta": delta, "beta": beta,
            "residuals": resid, "edf": edf, "rss": rss, "gcv": float(gcv),
            "lambda_": float(lambda_),
            "method": "esl_thin_plate_spline",
        },
    )


def _tps_kernel(A, B):
    """eta(r) = r^2 log r, with eta(0) = 0 by continuity."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * d2 * np.log(d2)
    return np.where(d2 > 0, out, 0.0)


def cheatsheet():
    return "eslthl: 2-D thin-plate spline; lambda -> inf gives the least-squares PLANE, not a constant"
