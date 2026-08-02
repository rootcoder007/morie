# morie.fn -- function file (rootcoder007/morie)
"""Three-step counterfactual inference: abduction, action, prediction."""

from . import _array_core as np
from scipy import optimize

from ._richresult import RichResult
from .scmdf import scm_definition

__all__ = ["abduction_modification_prediction"]


def abduction_modification_prediction(evidence, equations, exogenous_names, do, query):
    r"""Pearl's three-step counterfactual algorithm.

    1. **Abduction** -- update the exogenous variables U to be
       consistent with the observed evidence: solve for the u that
       reproduces the observed endogenous values under the *original*
       model.
    2. **Action** -- replace the equation for each intervened variable
       with the constant it is set to, forming :math:`M_{do}`.
    3. **Prediction** -- solve :math:`M_{do}` at the abducted u and
       read off the query variable.

    The step order is what makes counterfactuals differ from plain
    interventions: the same unit's exogenous background is carried
    across, so "what would *this* unit have done" is answerable while
    :math:`P(y \mid do(x))` alone is not.

    Parameters
    ----------
    evidence : dict
        Observed values of endogenous (and any known exogenous)
        variables.
    equations : dict
        endogenous name -> (parents, fn), as in
        :func:`morie.fn.scmdf.scm_definition`.
    exogenous_names : sequence
        Names of the exogenous variables to abduct.
    do : dict
        Intervened variable -> forced value.
    query : hashable
        The variable to predict.

    Returns
    -------
    RichResult
        keys: ``counterfactual``, ``factual``, ``abducted`` (the
        recovered u), ``residual`` (how well the abduction reproduced
        the evidence), ``do``, ``query``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Theorem 7.1.7 (abduction-action-prediction).
    """
    if not isinstance(evidence, dict) or not isinstance(equations, dict):
        raise ValueError("evidence and equations must be dicts.")
    unames = list(exogenous_names)
    if not unames:
        raise ValueError("need at least one exogenous variable to abduct.")
    for v in do:
        if v not in equations:
            raise ValueError(f"cannot intervene on {v!r}: it has no structural equation.")
    if query not in equations and query not in unames:
        raise ValueError(f"query {query!r} is not a variable of the model.")

    observed = {k: float(v) for k, v in evidence.items() if k in equations}
    if not observed:
        raise ValueError("evidence must fix at least one endogenous variable.")

    def solve(u_vec, eqs):
        u = {name: float(val) for name, val in zip(unames, u_vec)}
        u.update({k: v for k, v in evidence.items() if k in unames})
        return scm_definition(u, eqs)["values"]

    def residuals(u_vec):
        vals = solve(u_vec, equations)
        return [vals[k] - observed[k] for k in observed]

    u0 = np.zeros(len(unames))
    if len(unames) == len(observed):
        sol = optimize.fsolve(residuals, u0, full_output=True)
        u_hat, info, flag = sol[0], sol[1], sol[2]
        resid = float(np.max(np.abs(info["fvec"])))
    else:  # over- or under-determined: least squares
        ls = optimize.least_squares(residuals, u0)
        u_hat = ls.x
        resid = float(np.max(np.abs(ls.fun)))

    factual = solve(u_hat, equations)[query]
    mutilated = dict(equations)
    for v, val in do.items():
        mutilated[v] = ((), (lambda val=val: val))
    cf = solve(u_hat, mutilated)[query]

    return RichResult(
        payload={
            "counterfactual": float(cf),
            "factual": float(factual),
            "abducted": {name: float(val) for name, val in zip(unames, u_hat)},
            "residual": resid,
            "do": dict(do),
            "query": query,
            "method": "Abduction-action-prediction (Pearl Thm 7.1.7)",
        }
    )


def cheatsheet():
    return "abdpd: solve u from evidence, mutilate for do(), re-solve at the same u"
