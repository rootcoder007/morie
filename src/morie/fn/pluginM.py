# morie.fn -- function file (rootcoder007/morie)
"""Plug-in g-computation NIE."""

import numpy as np

from ._richresult import RichResult

__all__ = ["plug_in_mediation"]


def plug_in_mediation(x, m, y, c=None):
    r"""Plug-in g-computation of natural direct and indirect effects.

    Fits linear models :math:`M = \alpha_0 + \alpha_1 X + \alpha_2' C`
    and :math:`Y = \beta_0 + \beta_1 X + \beta_2 M + \beta_3' C`, then
    imputes Y under the counterfactual mediator level:

    .. math:: \mathrm{NIE} = \beta_2\,\alpha_1, \qquad
              \mathrm{NDE} = \beta_1,

    the g-computation (regression-substitution) plug-in, which under
    linearity coincides with the product-of-coefficients method. Total
    effect = NDE + NIE by construction.

    Parameters
    ----------
    x : array-like, shape (n,)
        Exposure.
    m : array-like, shape (n,)
        Mediator.
    y : array-like, shape (n,)
        Outcome.
    c : array-like, shape (n,) or (n, p), optional
        Baseline covariates entering both models.

    Returns
    -------
    RichResult
        keys: ``nie``, ``nde``, ``te``, ``prop_mediated``, ``n``,
        ``method``.

    References
    ----------
    VanderWeele, T. J. (2015). *Explanation in Causal Inference:
    Methods for Mediation and Interaction*. Oxford University Press.
    Ch. 2 (regression-based NDE/NIE without exposure-mediator
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
    if n < C.shape[1] + 4:
        raise ValueError("too few observations for the two regressions.")

    Dm = np.column_stack([np.ones(n), x, C])
    am, *_ = np.linalg.lstsq(Dm, m, rcond=None)
    Dy = np.column_stack([np.ones(n), x, m, C])
    by, *_ = np.linalg.lstsq(Dy, y, rcond=None)

    nie = float(by[2] * am[1])
    nde = float(by[1])
    te = nde + nie
    prop = nie / te if te != 0 else float("nan")

    return RichResult(
        payload={
            "nie": nie,
            "nde": nde,
            "te": te,
            "prop_mediated": float(prop),
            "n": int(n),
            "method": "Plug-in g-computation NIE (linear models)",
        }
    )


def cheatsheet():
    return "pluginM: NIE = beta_M * alpha_X, NDE = beta_X (linear plug-in g-computation)"
