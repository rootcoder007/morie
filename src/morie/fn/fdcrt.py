# morie.fn -- function file (rootcoder007/morie)
"""Front-door criterion checker."""

from ._richresult import RichResult

__all__ = ["frontdoor_criterion"]


def frontdoor_criterion(dag, X, Y, Z):
    r"""Check whether Z satisfies the front-door criterion for (X, Y).

    Pearl's three conditions (2009, Def. 3.3.3):

    1. Z intercepts every directed path from X to Y;
    2. there is no unblocked back-door path from X to Z;
    3. X blocks every back-door path from Z to Y.

    Reuses the d-separation machinery of
    :func:`morie.fn.bdcrt.backdoor_criterion`. When all three hold, the
    effect is identified by the front-door formula
    (:func:`morie.fn.fdadj.frontdoor_adjustment`) even with an
    unobserved X-Y confounder -- the case the back-door cannot handle.

    This replaces a placeholder that averaged the dag argument.

    Parameters
    ----------
    dag : dict or edge list
        As in ``backdoor_criterion``. Must be acyclic.
    X, Y : hashable
        Treatment and outcome nodes.
    Z : hashable or iterable
        Candidate mediator set.

    Returns
    -------
    RichResult
        keys: ``satisfied``, ``cond1``/``cond2``/``cond3``,
        ``unintercepted_paths``, ``reason``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality*, 2nd edn. Cambridge UP.
    Def. 3.3.3 and Thm. 3.3.4.
    """
    from .bdcrt import _has_cycle, _parse, _paths, backdoor_criterion

    Zs = {Z} if isinstance(Z, (str, int)) else set(Z)
    children, parents, nodes = _parse(dag)
    for name, node in (("X", X), ("Y", Y)):
        if node not in nodes:
            raise ValueError(f"{name} = {node!r} is not a node of the graph.")
    missing = Zs - nodes
    if missing:
        raise ValueError(f"Z contains nodes not in the graph: {sorted(map(str, missing))}.")
    if _has_cycle(children, nodes):
        raise ValueError("dag contains a cycle.")

    # Condition 1: every DIRECTED X -> Y path passes through Z.
    directed = [(p, d) for p, d in _paths(X, Y, children, parents) if all(step == "->" for step in d)]
    unintercepted = [" -> ".join(map(str, p)) for p, _ in directed if not (set(p[1:-1]) & Zs)]
    cond1 = len(directed) > 0 and not unintercepted

    # Condition 2: no unblocked back-door path X to Z (empty conditioning set).
    cond2 = all(backdoor_criterion(dag, X, z, ())["satisfied"] for z in Zs)

    # Condition 3: X blocks every back-door path from Z to Y.
    cond3 = all(backdoor_criterion(dag, z, Y, (X,))["satisfied"] for z in Zs)

    ok = cond1 and cond2 and cond3
    if ok:
        reason = "Z satisfies the front-door criterion; use the front-door formula."
    elif not cond1:
        reason = f"directed path(s) bypass Z: {unintercepted}" if unintercepted else "no directed X->Y path exists."
    elif not cond2:
        reason = "an unblocked back-door path runs from X to Z."
    else:
        reason = "X does not block every back-door path from Z to Y."
    return RichResult(
        payload={
            "satisfied": bool(ok),
            "cond1": bool(cond1),
            "cond2": bool(cond2),
            "cond3": bool(cond3),
            "unintercepted_paths": unintercepted,
            "reason": reason,
            "method": "Front-door criterion (Pearl 2009, Def. 3.3.3)",
        }
    )


def cheatsheet():
    return "fdcrt: front-door criterion check (Pearl Def. 3.3.3)"
