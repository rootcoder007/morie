# morie.fn -- function file (rootcoder007/morie)
"""Three conditions for a valid instrument Z for the causal effect of X on Y."""

from ._richresult import RichResult
from ._dsep import d_separated
from .bdcrt import _parse

__all__ = ["iv_conditions"]


def iv_conditions(dag, Z, X, Y):
    r"""Graphical instrument check.

    Z is an instrument for the effect of X on Y when

    1. **relevance** -- Z is associated with X: Z and X are *not*
       d-separated in the graph;
    2. **exclusion + independence** -- every association between Z and
       Y travels through X's causal effect: Z and Y are d-separated in
       the mutilated graph :math:`G_{\underline X}` with all edges
       *out of* X removed. One check covers both textbook clauses --
       a direct Z -> Y edge and a Z <- U -> Y confounding path both
       survive edge removal and are caught.

    Parameters
    ----------
    dag : dict or edge list
        The causal graph including unobserved nodes if any.
    Z, X, Y : hashable
        Instrument, treatment, outcome.

    Returns
    -------
    RichResult
        keys: ``relevance``, ``exclusion_independence``, ``valid``,
        ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Sec. 7.4.1 (instrumental variables, the graphical
    definition via :math:`G_{\underline X}`).
    """
    children, parents, nodes = _parse(dag)
    for n in (Z, X, Y):
        if n not in nodes:
            raise ValueError(f"node {n!r} not in the graph.")

    relevance = not d_separated(dag, Z, X)

    # G_underline-X: drop X's outgoing edges, keep every node present
    g = {n: [] for n in nodes}
    for u in children:
        if u != X:
            g[u].extend(children[u])
    excl = d_separated(g, Z, Y)

    return RichResult(
        payload={
            "relevance": bool(relevance),
            "exclusion_independence": bool(excl),
            "valid": bool(relevance and excl),
            "method": "IV conditions: Z ~ X and Z _||_ Y in G with X's outgoing edges removed",
        }
    )


def cheatsheet():
    return "ivcrt: relevance + d-sep(Z, Y) in G_underline-X (Pearl Sec 7.4.1)"
