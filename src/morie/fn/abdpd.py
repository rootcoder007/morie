# morie.fn -- function file (rootcoder007/morie)
"""Three-step counterfactual inference: abduction, action, prediction."""

from . import _array_core as np
from ._sci_core import optimize

from ._richresult import RichResult
from .scmdf import scm_definition

__all__ = ["abduction_modification_prediction"]


def abduction_modification_prediction(evidence, equations, exogenous_names, do, query,
                                      u_support=None):
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
    u_support : sequence, optional
        Candidate values for each exogenous variable.  When the
        continuous abduction cannot reproduce the evidence -- discrete
        models have zero gradient almost everywhere, so a Newton solver
        returns its starting point -- the abduction enumerates this
        support instead.  Defaults to ``(0.0, 1.0)``, the binary case of
        Pearl's Section 1.4.

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

    # Support enumeration runs FIRST.  With non-smooth equations the
    # evidence does not pin u uniquely in the continuum -- a step
    # function is satisfied by a whole interval -- so a solver can land
    # on an off-support u that reproduces the evidence yet yields the
    # wrong counterfactual.  Pearl's "compatible with only one
    # realization" is a statement WITHIN the support; grid solutions take
    # precedence, and the gradient path serves genuinely continuous
    # models where no grid point fits.
    support = tuple(float(v) for v in (u_support or (0.0, 1.0)))
    if u_support is not None and len(support) ** len(unames) > 200_000:
        raise ValueError(
            "discrete abduction over %d^%d candidates is too large; "
            "pass a smaller u_support" % (len(support), len(unames)))

    solutions = []
    if len(support) ** len(unames) <= 200_000:
        import itertools
        for cand in itertools.product(support, repeat=len(unames)):
            if max(abs(x) for x in residuals(list(cand))) < 1e-9:
                solutions.append(list(cand))
    if solutions:
        u_hat = solutions[0]
        resid = 0.0
        method = "discrete abduction over support %s" % (support,)
    else:
        u0 = np.zeros(len(unames))
        if len(unames) == len(observed):
            u_hat = optimize.fsolve(residuals, u0)
        else:
            u_hat = optimize.least_squares(residuals, u0).x
        u_hat = (list(u_hat._flat()) if hasattr(u_hat, "_flat")
                 else list(u_hat))
        resid = max(abs(r) for r in residuals(u_hat))
        solutions = [u_hat]
        method = "gradient abduction"

    mutilated = dict(equations)
    for v, val in do.items():
        mutilated[v] = ((), (lambda val=val: val))

    factuals = [solve(u, equations)[query] for u in solutions]
    cfs = [solve(u, mutilated)[query] for u in solutions]
    factual, cf = factuals[0], cfs[0]
    unique_cf = all(abs(c - cfs[0]) < 1e-12 for c in cfs)

    return RichResult(
        payload={
            "counterfactual": float(cf),
            "factual": float(factual),
            "abducted": {name: float(val) for name, val in zip(unames, u_hat)},
            "n_compatible_u": len(solutions),
            "counterfactual_unique": bool(unique_cf),
            "residual": resid,
            "do": dict(do),
            "query": query,
            "method": "Abduction-action-prediction (Pearl 2000, Sec. 1.4; %s)"
                      % method,
        }
    )


def cheatsheet():
    return "abdpd: solve u from evidence, mutilate for do(), re-solve at the same u"
