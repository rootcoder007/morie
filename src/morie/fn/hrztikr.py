# morie.fn -- function file (rootcoder007/morie)
"""Tikhonov-regularised nonparametric IV."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_tikhonov_iv"]


def hrz_tikhonov_iv(T, Ey_w, alpha=None, alphas=None):
    r"""Tikhonov regularisation for nonparametric IV (Horowitz Ch. 6):

    .. math:: \hat g = \arg\min_{g \in H}
              \big[\|\widehat{E}(Y|W) - \hat T g\|^2
              + \alpha_n \|g\|_H^2\big].

    The operator T is compact, so its inverse is UNBOUNDED and the
    problem is ill-posed: without the penalty, arbitrarily small
    perturbations in the estimated right-hand side produce arbitrarily
    large changes in g. The regularisation parameter must vanish
    slowly enough to control that -- too fast and the solution
    explodes, too slow and it is biased. The solution norm across a
    grid of alpha is returned so the L-curve trade-off is visible
    rather than a single alpha being picked silently.

    Parameters
    ----------
    T : array-like, shape (m, k)
        Discretised operator.
    Ey_w : array-like, shape (m,)
        Estimated conditional mean.
    alpha : float, optional
        Regularisation parameter.
    alphas : sequence of float, optional
        Grid for the L-curve.

    Returns
    -------
    RichResult
        keys: ``g``, ``alpha``, ``residual_norm``, ``solution_norm``,
        ``l_curve`` (alpha, residual, norm), ``condition_number``,
        ``ill_posed`` (True), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 6 (nonparametric instrumental variables).
    """
    Tm = np.atleast_2d(np.asarray(T, dtype=float))
    b = np.asarray(Ey_w, dtype=float).ravel()
    if Tm.shape[0] != b.size:
        raise ValueError(f"T has {Tm.shape[0]} rows but Ey_w has {b.size}.")
    k = Tm.shape[1]
    TtT = Tm.T @ Tm
    Ttb = Tm.T @ b
    cond = float(np.linalg.cond(TtT)) if k else np.inf

    def solve(a):
        return np.linalg.solve(TtT + float(a) * np.eye(k), Ttb)

    grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if alphas is None else \
        [float(a) for a in alphas]
    if any(a <= 0 for a in grid):
        raise ValueError("alpha values must be positive.")
    curve = []
    for a in grid:
        ga = solve(a)
        curve.append((a, float(np.linalg.norm(Tm @ ga - b)),
                      float(np.linalg.norm(ga))))
    a_use = grid[len(grid) // 2] if alpha is None else float(alpha)
    if a_use <= 0:
        raise ValueError(f"alpha must be positive, got {a_use}.")
    g = solve(a_use)
    return RichResult(payload={"g": g, "alpha": a_use,
                               "residual_norm": float(np.linalg.norm(Tm @ g - b)),
                               "solution_norm": float(np.linalg.norm(g)),
                               "l_curve": curve, "condition_number": cond,
                               "ill_posed": True,
                               "method": "Tikhonov; T compact so T^{-1} is unbounded"})


def cheatsheet():
    return "hrztikr: ill-posed by construction -- the L-curve is returned, not hidden"
