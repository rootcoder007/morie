# morie.fn -- function file (rootcoder007/morie)
"""S-estimator regression."""

import numpy as np

from ._richresult import RichResult

__all__ = ["s_regression_estimator", "s_estimator_regression"]


def s_regression_estimator(X, y, n_subsets=200, seed=0):
    r"""The S-estimator of Rousseeuw and Yohai (1984): minimise, over
    :math:`\beta`, the M-scale of the residuals,

    .. math:: \hat\beta_S = \arg\min_\beta s(\beta), \qquad
              \frac1n \sum_i \rho\!\left(\frac{y_i - x_i'\beta}
              {s}\right) = b,

    with Tukey's biweight at :math:`c = 1.5476` and :math:`b = 1/2`
    -- the calibration :math:`E_\Phi[\rho] = b` at exactly
    :math:`b/\rho(\infty) = 1/2` is what makes the breakdown point
    50% (their Theorem 3.1 territory). The price is steep: normal
    efficiency 28.7%. The S-estimate is therefore a STARTING POINT
    -- its scale and coefficients seed the MM step
    (``morie.fn.mmreg``) which recovers 95% efficiency without giving
    the breakdown back -- and using it as a final answer wastes
    two-thirds of the data's information at the clean model.

    The objective is non-convex; computation is by random
    p-subsets (fit through p points exactly, keep the smallest
    residual M-scale) plus local IRLS refinement, the standard
    strategy since Rousseeuw (1984). ``n_subsets`` trades the chance
    of missing the global optimum against time, and the seed makes
    the answer reproducible.

    Parameters
    ----------
    x, y : array-like
        Design (constant added when absent) and response.
    n_subsets : int, default 200
        Random p-subsets to try.
    seed : int, default 0
        Subset seed.

    Returns
    -------
    RichResult
        keys: ``beta``, ``scale``, ``residuals``, ``breakdown``,
        ``gaussian_efficiency``, ``c``, ``b``, ``role``,
        ``n_subsets``, ``n``, ``p``, ``method``.

    References
    ----------
    Rousseeuw, P. J. and Yohai, V. J. (1984), "Robust regression by
    means of S-estimators", in *Robust and Nonlinear Time Series
    Analysis*, Lecture Notes in Statistics 26, Springer, 256-272.
    """
    from ._robust import TUKEY_C_BREAKDOWN, prepare_design, s_regression

    A, yv = prepare_design(X, y)
    beta, scale = s_regression(A, yv, n_subsets=n_subsets, seed=seed)
    return RichResult(payload={
        "beta": beta, "scale": scale, "residuals": yv - A @ beta,
        "breakdown": 0.5, "gaussian_efficiency": 0.287,
        "c": TUKEY_C_BREAKDOWN, "b": 0.5,
        "role": "a starting point: seed the MM step for 95% efficiency "
                "without giving the 50% breakdown back",
        "n_subsets": int(n_subsets),
        "n": int(A.shape[0]), "p": int(A.shape[1]),
        "method": "S-estimator: minimise the residual M-scale "
                  "(Rousseeuw-Yohai 1984), biweight c = 1.5476, b = 1/2"})


def cheatsheet():
    return "sestrg: 50% breakdown bought at 28.7% efficiency -- a starting point, not an endpoint"


#: Catalogue alias for :func:`s_regression_estimator`.
s_estimator_regression = s_regression_estimator
