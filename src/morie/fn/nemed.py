# morie.fn -- function file (rootcoder007/morie)
"""Nested counterfactual mediation effect."""

import numpy as np

from ._richresult import RichResult

__all__ = ["nested_counterfactual_mediation"]


def nested_counterfactual_mediation(x, m, y, c=None, x1=1.0, x0=0.0):
    r"""Natural direct and indirect effects with exposure-mediator interaction.

    Fits

    .. math::
        M &= \beta_0 + \beta_1 X + \beta_2' C, \\
        Y &= \theta_0 + \theta_1 X + \theta_2 M + \theta_3 X M
             + \theta_4' C,

    and evaluates VanderWeele's closed forms for the nested
    counterfactual :math:`Y_{x, M_{x'}}`:

    .. math::
        \mathrm{NDE} &= (\theta_1 + \theta_3(\beta_0 + \beta_1 x_0
                        + \beta_2' \bar c))(x_1 - x_0), \\
        \mathrm{NIE} &= (\theta_2 + \theta_3 x_1)\beta_1 (x_1 - x_0).

    With :math:`\theta_3 = 0` these collapse to the familiar
    :math:`\theta_1` and :math:`\theta_2 \beta_1`; the interaction term
    is exactly what makes NDE + NIE still equal the total effect while
    each piece depends on the reference level.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Exposure, mediator, outcome.
    c : array-like, optional
        Baseline covariates; the effects are evaluated at their means.
    x1, x0 : float, default 1 and 0
        Contrasted exposure levels.

    Returns
    -------
    RichResult
        keys: ``nde``, ``nie``, ``te``, ``interaction`` (theta3),
        ``coefficients``, ``n``, ``method``.

    References
    ----------
    VanderWeele, T. J. (2015). *Explanation in Causal Inference:
    Methods for Mediation and Interaction*. Oxford University Press.
    Ch. 2 (regression-based NDE/NIE with exposure-mediator
    interaction).
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n):
        raise ValueError("x, m, y must have equal length.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 6:
        raise ValueError("too few observations for the mediator and outcome regressions.")

    def ols(D, t):
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    one = np.ones(n)
    beta = ols(np.column_stack([one, x, C]), m)
    theta = ols(np.column_stack([one, x, m, x * m, C]), y)
    b0, b1 = float(beta[0]), float(beta[1])
    bc = beta[2:].astype(float)
    t1, t2, t3 = float(theta[1]), float(theta[2]), float(theta[3])
    cbar = C.mean(axis=0) if C.shape[1] else np.empty(0)

    m_at_x0 = b0 + b1 * x0 + (bc @ cbar if bc.size else 0.0)
    nde = (t1 + t3 * m_at_x0) * (x1 - x0)
    nie = (t2 + t3 * x1) * b1 * (x1 - x0)

    return RichResult(
        payload={
            "nde": float(nde),
            "nie": float(nie),
            "te": float(nde + nie),
            "interaction": t3,
            "coefficients": {"beta0": b0, "beta1": b1, "theta1": t1, "theta2": t2, "theta3": t3},
            "n": int(n),
            "method": "Nested counterfactual NDE/NIE with exposure-mediator interaction",
        }
    )


def cheatsheet():
    return "nemed: NDE = (t1 + t3*E[M|x0])(x1-x0); NIE = (t2 + t3*x1) b1 (x1-x0)"
