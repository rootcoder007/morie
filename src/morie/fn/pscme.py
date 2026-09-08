# morie.fn -- function file (rootcoder007/morie)
"""Path-specific causal effect for multiple mediators."""

from . import _array_core as np

from ._richresult import RichResult
from .medstg import sequential_mediation

__all__ = ["path_specific_causal_effect"]


def path_specific_causal_effect(x, m1, m2, y, c=None):
    r"""Enumerate the path-specific effects of a two-mediator SCM.

    For the causally ordered graph
    :math:`X \to M_1 \to M_2 \to Y` (with all shortcut edges present),
    the four directed paths from X to Y carry

    ============================  ==========================
    path                          effect
    ============================  ==========================
    X -> Y                        :math:`c'`
    X -> M1 -> Y                  :math:`a_1 b_1`
    X -> M2 -> Y                  :math:`a_2 b_2`
    X -> M1 -> M2 -> Y            :math:`a_1 d\, b_2`
    ============================  ==========================

    Under linearity a path-specific effect is the product of the edge
    coefficients along it, and the total effect is their sum -- the
    linear special case of Avin, Shpitser and Pearl's path-specific
    effects.

    Parameters
    ----------
    x, m1, m2, y : array-like, shape (n,)
        Treatment, ordered mediators, outcome.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``paths`` (dict path-string -> effect), ``total``,
        ``coefficients``, ``n``, ``method``.

    References
    ----------
    Avin, C., Shpitser, I. & Pearl, J. (2005). Identifiability of
    path-specific effects. *Proceedings of IJCAI-05*, 357-363.

    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Sec. 4.5 (path coefficients and effect decomposition in
    linear models).
    """
    s = sequential_mediation(x, m1, m2, y, c=c)
    paths = {
        "X->Y": s["direct"],
        "X->M1->Y": s["via_m1"],
        "X->M2->Y": s["via_m2"],
        "X->M1->M2->Y": s["serial"],
    }
    return RichResult(
        payload={
            "paths": paths,
            "total": float(sum(paths.values())),
            "coefficients": s["paths"],
            "n": s["n"],
            "method": "Path-specific effects (products of linear path coefficients)",
        }
    )


def cheatsheet():
    return "pscme: effect of each directed X->Y path = product of its edge coefficients"
