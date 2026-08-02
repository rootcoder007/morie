# morie.fn -- function file (rootcoder007/morie)
"""MM-estimator regression (Yohai 1987)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mm_regression_estimator"]


def mm_regression_estimator(X, y, n_subsets=200, seed=0):
    r"""Yohai's (1987) MM-estimator, the three-stage construction
    that gets 50% breakdown AND 95% normal efficiency at once:

    1. an initial high-breakdown fit (random p-subsets);
    2. the M-scale :math:`\hat s` of its residuals, biweight at
       :math:`c = 1.5476`, :math:`b = 1/2` -- this stage owns the
       breakdown point;
    3. an M-step for :math:`\beta` with the biweight at
       :math:`c = 4.685`, iterated from the stage-1 coefficients
       **with the scale held fixed at** :math:`\hat s`.

    Holding the scale fixed is the load-bearing detail. The
    efficiency stage's larger :math:`c` would, if allowed to
    re-estimate the scale, let gross outliers back into it -- and the
    breakdown would drop back toward the M-estimator's zero. Yohai's
    Theorem 2.1 gives consistency and asymptotic normality; the
    efficiency constant 4.685 solves the same 95% equation verified
    in ``morie.fn._robust``.

    Parameters
    ----------
    x, y : array-like
        Design (constant added when absent) and response.
    n_subsets : int, default 200
        Random p-subsets for the initial stage.
    seed : int, default 0
        Subset seed.

    Returns
    -------
    RichResult
        keys: ``beta``, ``scale``, ``beta_initial`` (the S-estimate),
        ``residuals``, ``se``, ``weights``, ``breakdown``,
        ``gaussian_efficiency``, ``scale_held_fixed`` (True),
        ``converged``, ``n``, ``p``, ``method``.

    References
    ----------
    Yohai, V. J. (1987), "High breakdown-point and high efficiency
    robust estimates for regression", *Annals of Statistics*
    15:642-656, Sec. 2 and Theorem 2.1.
    """
    from ._robust import (TUKEY_C_95, mm_regression, prepare_design,
                          tukey_weight)

    A, yv = prepare_design(X, y)
    beta, scale, beta_s, conv = mm_regression(A, yv, n_subsets=n_subsets,
                                              seed=seed)
    r = yv - A @ beta
    u = r / scale if scale > 0 else r
    w = tukey_weight(u, TUKEY_C_95)
    # asymptotic sandwich at the MM solution
    psi = u * w
    dpsi_mean = float(np.mean((1 - (u / TUKEY_C_95) ** 2)
                              * (1 - 5 * (u / TUKEY_C_95) ** 2)
                              * (np.abs(u) < TUKEY_C_95)))
    kappa = float(np.mean(psi ** 2)) / max(dpsi_mean ** 2, 1e-12)
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * kappa * scale ** 2, 0.0))
    return RichResult(payload={
        "beta": beta, "scale": float(scale), "beta_initial": beta_s,
        "residuals": r, "se": se, "weights": w,
        "breakdown": 0.5, "gaussian_efficiency": 0.95,
        "scale_held_fixed": True,
        "why_fixed": "re-estimating the scale in the efficiency stage would "
                     "let outliers back into it through the larger c, and "
                     "the breakdown would fall back toward zero",
        "converged": bool(conv),
        "n": int(A.shape[0]), "p": int(A.shape[1]),
        "method": "MM-estimator (Yohai 1987): S-scale at c = 1.5476, "
                  "M-step at c = 4.685, scale fixed"})


def cheatsheet():
    return "mmreg: the scale is FROZEN through the efficiency stage -- that is where the breakdown lives"
