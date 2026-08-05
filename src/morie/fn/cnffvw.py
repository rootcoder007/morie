# morie.fn -- function file (rootcoder007/morie)
"""Cinelli-Hazlett sensitivity -- alias of :mod:`morie.fn.chzlt`.

`cnffvw` and `chzlt` document the SAME method: the Cinelli & Hazlett
(2020) omitted-variable-bias bound and robustness value.  Rather than
carry a second implementation -- which would agree with the first at
1e-9 forever while doubling the surface -- this module forwards to
`chzlt` with the argument names of its own stub signature.
"""

from ._richresult import RichResult
from .chzlt import cinelli_hazlett, ols_with_se, robustness_value

__all__ = ["cinelli_hazlett_robust"]


def cinelli_hazlett_robust(y, D, X=None, R2_Y=0.0, R2_D=0.0, q=1.0):
    """
    Cinelli-Hazlett sensitivity under a hypothesised confounder

    Formula: adjusted estimate under hypothesised confounder R2_Y * R2_D

    Same estimator as :func:`morie.fn.chzlt.cinelli_hazlett`; see there
    for the bias bound and the robustness value.

    Parameters
    ----------
    y : array-like
        Outcome.
    D : array-like
        Treatment.
    X : array-like or None
        Observed covariates.
    R2_Y, R2_D : float
        Hypothesised partial R2 of the confounder with outcome and
        treatment.
    q : float
        Fraction of the estimate the confounder would have to explain.

    Returns
    -------
    result : dict
        As :func:`morie.fn.chzlt.cinelli_hazlett`.

    References
    ----------
    Cinelli & Hazlett (2020), Making Sense of Sensitivity, JRSS B
    82(1):39-67.
    """
    return cinelli_hazlett(y, D, X, R2_Y, R2_D, q)


def cheatsheet():
    return "cnffvw: Cinelli-Hazlett sensitivity (alias of chzlt)"


# compact alias per ledger/NAMING.md
cinellihazlettrobust = cinelli_hazlett_robust
