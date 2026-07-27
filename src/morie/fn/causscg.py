# morie.fn -- function file (rootcoder007/morie)
"""Generalised synthetic control via interactive fixed effects."""

from ._richresult import RichResult
from .gscmcl import generalized_synthetic_control

__all__ = ["causal_generalised_sc"]


def causal_generalised_sc(y_treated, y_controls, treat_time, r=2):
    """Front-end to :func:`morie.fn.gscmcl.generalized_synthetic_control`.

    Same latent-factor imputation of the treated unit's untreated
    trajectory (Xu 2017, *Political Analysis* 25(1), 57-76); kept as a
    separate entry point for the causal-inference namespace.
    """
    out = generalized_synthetic_control(y_treated, y_controls, treat_time, r=r)
    return RichResult(payload=dict(out))


def cheatsheet():
    return "causscg: front-end to gscmcl (Xu 2017 generalized synthetic control)"
