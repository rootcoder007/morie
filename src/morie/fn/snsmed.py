# morie.fn -- function file (rootcoder007/morie)
"""Sensitivity analysis for unmeasured confounding in mediation."""

from ._richresult import RichResult
from .sensIM import imai_sensitivity_rho

__all__ = ["sensitivity_mediation"]


def sensitivity_mediation(x, m, y, rho=None, c=None):
    """ACME under a specified mediator-outcome error correlation.

    Front-end to :func:`morie.fn.sensIM.imai_sensitivity_rho`: pass a
    single ``rho`` (or a list) instead of the default grid and read the
    ACME back at exactly those values, together with the critical rho
    at which the effect vanishes.

    References
    ----------
    Imai, K., Keele, L. & Tingley, D. (2010). A general approach to
    causal mediation analysis. *Psychological Methods*, 15(4),
    309-334. Theorem 2, p. 316.
    """
    grid = [0.0] if rho is None else rho
    out = imai_sensitivity_rho(x, m, y, rho_grid=grid, c=c)
    payload = dict(out)
    payload["method"] = "Mediation sensitivity to unmeasured confounding at given rho"
    return RichResult(payload=payload)


def cheatsheet():
    return "snsmed: ACME at user-specified rho (sensIM front-end)"
