# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Graphical model joint (delegates to the undirected core)."""

from ._richresult import RichResult
from .wsmund import wasserman_undirected_graph

__all__ = ["wasserman_graphical_model"]


def wasserman_graphical_model(graph, psi):
    """
    Normalised clique-potential model p(x) = prod_C psi_C(x_C) / Z.

    Delegates the enumeration and partition function to wsmund
    (single source of truth) and re-reports with the most probable
    configuration decoded, which is what Ch 17's examples read off.

    Parameters
    ----------
    graph : tuple (n_nodes, cliques)
        As in wsmund.
    psi : sequence of callables
        Clique potentials.

    Returns
    -------
    result : dict
        Keys: estimate (probability of the mode), mode (bit list),
        partition_function, probabilities, n_nodes, method.

    References
    ----------
    Wasserman (2004), Ch 17.

    Examples
    --------
    >>> agree = lambda t: 2.0 if t[0] == t[1] else 1.0
    >>> out = wasserman_graphical_model((2, [(0, 1)]), [agree])
    >>> out["mode"]
    [0, 0]
    >>> round(out["estimate"], 12)
    0.333333333333
    >>> out["partition_function"]
    6.0
    """
    core = wasserman_undirected_graph(graph, psi)
    probs = core["probabilities"]
    n = core["n_nodes"]
    best = max(range(len(probs)), key=lambda i: probs[i])
    mode = [(best >> (n - 1 - j)) & 1 for j in range(n)]
    return RichResult(payload={
        "estimate": float(probs[best]), "mode": mode,
        "partition_function": core["estimate"],
        "probabilities": probs, "n_nodes": n,
        "method": "clique-potential joint via wsmund; mode decoded (ties -> lowest index)"})


def cheatsheet():
    return "wsmgrp: delegates to wsmund; adds argmax config decode"
