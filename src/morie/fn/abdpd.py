# morie.fn -- function file (rootcoder007/morie)
"""Three-step counterfactual inference: abduction, action, prediction."""

from . import _array_core as np
from ._sci_core import optimize

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

    This is the DETERMINISTIC form of the procedure: it recovers a point
    u and returns a single counterfactual value.  Pearl's Theorem 7.1.7
    states the probabilistic version, whose step 1 updates P(u) to
    P(u | e); that reduces to this one exactly when the evidence
    determines u uniquely, which is the case worked in Section 1.4.

    References
    ----------
    Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*
    (1st ed.). Cambridge University Press, Section 1.4 pp. 36-37 for the
    three-step procedure, Theorem 7.1.7 for the probabilistic statement.
    Verified against the copy held in the corpus.
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
    # The native fsolve drops **kw (including full_output) and returns the
    # solution vector alone, so unpacking scipy's 4-tuple raised
    # IndexError on every exactly-determined problem -- which is the case
    # Pearl actually describes.  The residual is recomputed here instead of
    # being read out of a solver info dict, which also makes the two
    # branches report the same quantity.
    if len(unames) == len(observed):
        u_hat = optimize.fsolve(residuals, u0)
    else:  # over- or under-determined: least squares
        u_hat = optimize.least_squares(residuals, u0).x
    resid = max(abs(r) for r in residuals(u_hat))

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
            "method": "Abduction-action-prediction (Pearl 2000, Sec. 1.4)",
        }
    )


def cheatsheet():
    return "abdpd: solve u from evidence, mutilate for do(), re-solve at the same u"
