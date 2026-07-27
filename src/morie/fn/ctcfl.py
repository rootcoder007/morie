# morie.fn -- function file (rootcoder007/morie)
"""Counterfactual notation: Y_x -- outcome had X been set to x by intervention."""

from ._richresult import RichResult
from .scmdf import scm_definition

__all__ = ["counterfactual_notation"]


def counterfactual_notation(exogenous, equations, X, x_val, Y):
    r"""Evaluate the unit-level counterfactual :math:`Y_x(u)`.

    Pearl's definition: :math:`Y_x(u) = Y_{M_x}(u)`, the value of Y in
    the *modified* model :math:`M_x` where the equation for X is
    replaced by the constant :math:`X = x`, evaluated at the same
    exogenous setting u. With u fully specified the abduction step is
    trivial and the three-step recipe reduces to action + prediction:
    mutilate, then solve.

    Parameters
    ----------
    exogenous : dict
        The exogenous setting u.
    equations : dict
        endogenous name -> (parents, fn), as in
        :func:`morie.fn.scmdf.scm_definition`.
    X : hashable
        The intervened variable (must be endogenous).
    x_val :
        The value X is set to.
    Y : hashable
        The queried variable.

    Returns
    -------
    RichResult
        keys: ``counterfactual`` (Y_x(u)), ``factual`` (Y(u) in the
        unmutilated model), ``effect`` (difference when both are
        numeric), ``X``, ``x_val``, ``Y``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Def. 7.1.5 (potential response Y_x(u) via the submodel
    M_x), Thm 7.1.7 (the three-step procedure).
    """
    if X not in equations:
        raise ValueError(f"X = {X!r} must be an endogenous variable with an equation to replace.")
    if Y not in equations and Y not in exogenous:
        raise ValueError(f"Y = {Y!r} is not a variable of the model.")

    factual = scm_definition(exogenous, equations)["values"][Y]

    mutilated = dict(equations)
    mutilated[X] = ((), lambda: x_val)
    cf = scm_definition(exogenous, mutilated)["values"][Y]

    effect = None
    try:
        effect = cf - factual
    except TypeError:
        pass

    return RichResult(
        payload={
            "counterfactual": cf,
            "factual": factual,
            "effect": effect,
            "X": X,
            "x_val": x_val,
            "Y": Y,
            "method": "Y_x(u) = Y in the submodel M_x at the same exogenous u",
        }
    )


def cheatsheet():
    return "ctcfl: Y_x(u) by replacing X's equation with X = x and re-solving (Pearl Def 7.1.5)"
