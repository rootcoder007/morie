# morie.fn -- function file (rootcoder007/morie)
"""Structural causal model (SCM) definition: (U, V, F) triple."""

from ._richresult import RichResult
from .bdcrt import _has_cycle, _parse

__all__ = ["scm_definition"]


def scm_definition(exogenous, equations):
    r"""Build and evaluate an SCM :math:`M = (U, V, F)`.

    ``equations`` maps each endogenous variable to
    ``(parents, fn)`` where ``fn(**values)`` computes
    :math:`V_i = f_i(\mathrm{pa}(V_i), U_i)`; parents may be exogenous
    or endogenous names. The induced graph is checked for acyclicity
    and the equations are solved once in topological order, giving the
    unique solution recursiveness guarantees.

    Parameters
    ----------
    exogenous : dict
        name -> value of every exogenous variable U.
    equations : dict
        endogenous name -> (iterable of parent names, callable). The
        callable receives every parent as a keyword argument.

    Returns
    -------
    RichResult
        keys: ``values`` (all variables, solved), ``order``
        (topological evaluation order), ``edges`` (parent, child)
        list, ``exogenous`` (names), ``endogenous`` (names),
        ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Def. 7.1.1 (causal model as a (U, V, F) triple) and
    Sec. 3.2 (recursiveness / acyclicity).
    """
    if not isinstance(exogenous, dict) or not isinstance(equations, dict):
        raise ValueError("exogenous and equations must be dicts.")
    overlap = set(exogenous) & set(equations)
    if overlap:
        raise ValueError(f"variables cannot be both exogenous and endogenous: {sorted(overlap)}.")

    edges = []
    for v, (pa, fn) in equations.items():
        if not callable(fn):
            raise ValueError(f"equation for {v!r} is not callable.")
        for p in pa:
            if p not in exogenous and p not in equations:
                raise ValueError(f"parent {p!r} of {v!r} is neither exogenous nor endogenous.")
            edges.append((p, v))

    children, parents, nodes = _parse(edges or {v: [] for v in list(exogenous) + list(equations)})
    if _has_cycle(children, nodes):
        raise ValueError("structural equations induce a directed cycle; SCM must be recursive.")

    values = dict(exogenous)
    order = []
    pending = dict(equations)
    while pending:
        ready = [v for v, (pa, _) in pending.items() if all(p in values for p in pa)]
        if not ready:  # unreachable given the acyclicity check, kept as a guard
            raise ValueError("could not order equations (unresolvable parents).")
        for v in sorted(ready):
            pa, fn = pending.pop(v)
            values[v] = fn(**{p: values[p] for p in pa})
            order.append(v)

    return RichResult(
        payload={
            "values": values,
            "order": order,
            "edges": edges,
            "exogenous": sorted(exogenous),
            "endogenous": sorted(equations),
            "method": "SCM (U, V, F): acyclicity checked, solved in topological order",
        }
    )


def cheatsheet():
    return "scmdf: build (U, V, F), check recursive, solve V_i = f_i(pa, U) in topo order"
