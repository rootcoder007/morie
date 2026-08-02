# morie.fn -- function file (rootcoder007/morie)
"""SEM-based mediation (path-analysis style)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sem_mediation"]


def sem_mediation(model_spec, data):
    r"""Path analysis for a recursive mediation model.

    ``model_spec`` maps each endogenous variable to its list of
    predictors, e.g. ``{"M": ["X"], "Y": ["X", "M"]}``. Each equation
    is estimated by OLS (recursive models with uncorrelated
    disturbances are identified equation by equation, so full-
    information estimation is unnecessary), and every directed path
    from a source to a sink is enumerated with its effect equal to the
    product of the edge coefficients along it -- Wright's rule for
    linear path models.

    Parameters
    ----------
    model_spec : dict
        endogenous variable -> list of predictor names.
    data : mapping
        variable name -> 1-D array, all the same length.

    Returns
    -------
    RichResult
        keys: ``coefficients`` (nested dict outcome -> predictor ->
        estimate), ``r_squared`` per equation, ``paths`` (dict of
        "A->B->C" -> effect for every directed path between the
        exogenous and the final endogenous variable), ``total_effects``
        (source -> summed path effect), ``n``, ``method``.

    References
    ----------
    Bollen, K. A. (1989). *Structural Equations with Latent
    Variables*. Wiley. Ch. 4 (path analysis; recursive models are
    estimable equation by equation).

    Wright, S. (1934). The method of path coefficients. *Annals of
    Mathematical Statistics*, 5(3), 161-215.
    """
    if not isinstance(model_spec, dict) or not model_spec:
        raise ValueError("model_spec must be a non-empty dict.")
    arrays = {k: np.asarray(v, dtype=float).ravel() for k, v in data.items()}
    if not arrays:
        raise ValueError("data is empty.")
    n = len(next(iter(arrays.values())))
    if any(v.size != n for v in arrays.values()):
        raise ValueError("all data columns must have the same length.")

    coefs, r2 = {}, {}
    for outcome, preds in model_spec.items():
        missing = [p for p in [outcome, *preds] if p not in arrays]
        if missing:
            raise ValueError(f"data missing variables: {missing}.")
        D = np.column_stack([np.ones(n)] + [arrays[p] for p in preds])
        if n < D.shape[1] + 2:
            raise ValueError(f"too few observations for the equation for {outcome!r}.")
        b, *_ = np.linalg.lstsq(D, arrays[outcome], rcond=None)
        resid = arrays[outcome] - D @ b
        tss = float(((arrays[outcome] - arrays[outcome].mean()) ** 2).sum())
        coefs[outcome] = {p: float(bi) for p, bi in zip(preds, b[1:])}
        r2[outcome] = float(1 - (resid**2).sum() / tss) if tss > 0 else float("nan")

    # enumerate directed paths source -> ... -> sink using the spec's edges
    children = {}
    for outcome, preds in model_spec.items():
        for p in preds:
            children.setdefault(p, []).append(outcome)
    endog = set(model_spec)
    sinks = [v for v in endog if v not in children]
    sources = [v for v in arrays if v not in endog]

    paths, totals = {}, {}
    for s in sources:
        total = 0.0
        stack = [(s, [s], 1.0)]
        while stack:
            cur, route, prod = stack.pop()
            if cur in sinks and len(route) > 1:
                paths["->".join(route)] = prod
                total += prod
                continue
            for nxt in children.get(cur, []):
                if nxt in route:
                    continue
                stack.append((nxt, route + [nxt], prod * coefs[nxt][cur]))
        totals[s] = total

    return RichResult(
        payload={
            "coefficients": coefs,
            "r_squared": r2,
            "paths": paths,
            "total_effects": totals,
            "n": int(n),
            "method": "Recursive path analysis (equation-by-equation OLS, Wright's rule)",
        }
    )


def cheatsheet():
    return "medSEM: OLS per equation; path effect = product of edge coefficients"
