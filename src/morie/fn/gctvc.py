# morie.fn -- function file (rootcoder007/morie)
"""G-computation (parametric g-formula) for time-varying confounding."""

import numpy as np

from ._richresult import RichResult
from .gforml import robins_g_formula

__all__ = ["g_computation_time_varying"]


def g_computation_time_varying(y, treatment_history, covariate_history, n_mc=2000, seed=0):
    """Always-treat vs never-treat contrast by the parametric g-formula.

    Front-end to :func:`morie.fn.gforml.robins_g_formula`: runs the
    Monte Carlo g-formula under the two static regimes abar = 1 and
    abar = 0 and reports their difference -- the standard total-effect
    contrast under time-varying confounding, where L_t is both a
    confounder for A_t and affected by A_{t-1} and naive regression
    (adjusting for L) blocks part of the effect while failing to close
    the backdoor.

    Parameters
    ----------
    y, treatment_history, covariate_history :
        As in :func:`robins_g_formula`.
    n_mc, seed :
        Monte Carlo controls, shared by both regimes.

    Returns
    -------
    RichResult
        keys: ``estimate`` (E[Y(1bar)] - E[Y(0bar)]), ``EY_always``,
        ``EY_never``, ``n``, ``n_periods``, ``method``.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
    mortality studies with a sustained exposure period. *Mathematical
    Modelling*, 7, 1393-1512.
    """
    A = np.asarray(treatment_history, dtype=float)
    T = 1 if A.ndim == 1 else A.shape[1]
    hi = robins_g_formula(y, treatment_history, covariate_history, np.ones(T), n_mc=n_mc, seed=seed)
    lo = robins_g_formula(y, treatment_history, covariate_history, np.zeros(T), n_mc=n_mc, seed=seed)
    return RichResult(
        payload={
            "estimate": hi["estimate"] - lo["estimate"],
            "EY_always": hi["estimate"],
            "EY_never": lo["estimate"],
            "n": hi["n"],
            "n_periods": int(T),
            "method": "G-computation (parametric g-formula) always-vs-never contrast",
        }
    )


def cheatsheet():
    return "gctvc: g-formula contrast E[Y(1bar)] - E[Y(0bar)] via gforml"
