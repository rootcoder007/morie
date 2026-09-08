# morie.fn -- function file (rootcoder007/morie)
"""Exchangeability (unconfoundedness/ignorability) as a graph check."""

from ._richresult import RichResult
from .bdcrt import backdoor_criterion

__all__ = ["exchangeability_assumption"]


def exchangeability_assumption(dag, T, Y, X=()):
    r"""Does conditioning on X give exchangeability in this DAG?

    Exchangeability, :math:`Y(t) \perp T \mid X`, is untestable from
    the joint distribution of (Y, T, X) alone; what *can* be checked
    is whether it holds in a posited causal graph. Graphically it is
    exactly Pearl's back-door criterion: X blocks every back-door path
    from T to Y and contains no descendant of T. This function
    delegates to that check and reports the verdict with the offending
    open paths.

    Parameters
    ----------
    dag : dict or edge list
        The causal graph, including unobserved common causes if any.
    T, Y : hashable
        Treatment and outcome nodes.
    X : iterable, optional
        The proposed adjustment set.

    Returns
    -------
    RichResult
        keys: ``holds``, ``open_backdoor_paths``,
        ``descendant_violations``, ``adjustment_set``, ``method``.

    References
    ----------
    Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
    Chapman & Hall/CRC. Ch. 2 (exchangeability), Ch. 7 (its graphical
    expression as the back-door criterion).

    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Def. 3.3.1 (back-door criterion).
    """
    bd = backdoor_criterion(dag, T, Y, Z=tuple(X))
    return RichResult(
        payload={
            "holds": bd["satisfied"],
            "open_backdoor_paths": bd.get("open_paths", []),
            "descendant_violations": bd.get("descendant_violations", []),
            "adjustment_set": tuple(X),
            "method": "Exchangeability given X == back-door criterion in the posited DAG",
        }
    )


def cheatsheet():
    return "exchg: Y(t) _||_ T | X holds in the DAG iff X satisfies the back-door criterion"
