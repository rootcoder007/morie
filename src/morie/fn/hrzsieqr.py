# morie.fn -- function file (rootcoder007/morie)
"""Series quantile regression."""

import numpy as np

from ._horowitz import sieve_basis
from ._richresult import RichResult

__all__ = ["hrz_series_quantile"]


from scipy import optimize


def hrz_series_quantile(x, y, tau=0.5, K=5, kind="poly", grid=None):
    r"""Series quantile regression (Horowitz Ch. 3):

    .. math:: \hat q_\tau(x) = \sum_k \hat a_k p_k(x), \qquad
              \hat a = \arg\min \frac1n \sum_i
              \rho_\tau\Big(Y_i - \sum_k a_k p_k(X_i)\Big).

    The check loss is convex but not differentiable, so the fit uses a
    subgradient-capable solver rather than a normal-equations
    shortcut. Unlike the mean version, a fitted quantile curve carries
    no guarantee of monotonicity in tau across separate fits --
    quantile crossing is a real artefact of estimating each tau
    independently.

    Parameters
    ----------
    x, y : array-like
        Regressor and response.
    tau : float in (0, 1), default 0.5
        Quantile level.
    K : int, default 5
        Sieve dimension.
    kind : {"poly", "fourier"}
        Basis.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``quantile``, ``coefficients``, ``tau``,
        ``K``, ``check_loss``, ``crossing_warning``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 3 (quantile regression; series estimators).
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must lie in (0, 1), got {tau}.")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    K = int(K)
    if K < 1 or K > x.size:
        raise ValueError(f"K must lie in 1..{x.size}, got {K}.")
    P = sieve_basis(x, K=K, kind=kind)

    def loss(a):
        r = y - P @ a
        return float(np.sum(r * (tau - (r < 0))))

    a0 = np.linalg.lstsq(P, y, rcond=None)[0]
    res = optimize.minimize(loss, a0, method="Powell",
                            options={"maxiter": 20000, "xtol": 1e-8})
    a = res.x
    g = x if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    Pg = sieve_basis(np.r_[x, g], K=K, kind=kind)[x.size:]
    return RichResult(payload={"grid": g, "quantile": Pg @ a,
                               "coefficients": a, "tau": float(tau), "K": K,
                               "check_loss": float(res.fun),
                               "crossing_warning":
                                   "separate tau fits may cross; not monotone by construction",
                               "converged": bool(res.success),
                               "method": "Series check-loss fit; convex but non-differentiable"})


def cheatsheet():
    return "hrzsieqr: separate tau fits can cross -- monotonicity is not built in"
