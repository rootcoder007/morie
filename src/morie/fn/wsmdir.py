# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Directed graphical model factorisation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_directed_graph"]


def wasserman_directed_graph(dag, x):
    """
    Joint probability of a binary DAG model by the Markov
    factorisation.

    Formula: p(x) = prod_i p(x_i | pa(x_i)). ``dag`` is a list of
    nodes in topological order, each a dict with keys ``parents``
    (list of earlier node indices) and ``cpt`` — the conditional
    probability P(X_i = 1 | parent values), keyed by the parents'
    bit-tuple (a plain dict; () for roots). ``x`` is the binary
    configuration. Cycles are rejected by the topological-order
    requirement itself (a parent index >= its child raises).

    Parameters
    ----------
    dag : sequence of dict
        Nodes with 'parents' and 'cpt' as above.
    x : sequence of int
        Binary configuration, same length.

    Returns
    -------
    result : dict
        Keys: estimate (joint probability), log_joint, factors (per
        node), n_nodes, method.

    References
    ----------
    Wasserman (2004), Ch 17 (directed graphs).

    Examples
    --------
    Chain X -> Y with P(X=1) = 0.3, P(Y=1|X=1) = 0.9, P(Y=1|X=0) = 0.2:

    >>> dag = [{"parents": [], "cpt": {(): 0.3}},
    ...        {"parents": [0], "cpt": {(0,): 0.2, (1,): 0.9}}]
    >>> out = wasserman_directed_graph(dag, [1, 1])
    >>> round(out["estimate"], 12)
    0.27
    >>> round(wasserman_directed_graph(dag, [0, 0])["estimate"], 12)
    0.56
    >>> s = sum(wasserman_directed_graph(dag, [a, b])["estimate"]
    ...         for a in (0, 1) for b in (0, 1))
    >>> round(s, 12)
    1.0
    >>> bad = [{"parents": [1], "cpt": {(0,): 0.5, (1,): 0.5}},
    ...        {"parents": [], "cpt": {(): 0.5}}]
    >>> wasserman_directed_graph(bad, [0, 0])
    Traceback (most recent call last):
        ...
    ValueError: node 0 has parent 1 not earlier in the ordering; supply a topological order.
    """
    x = [int(v) for v in x]
    if len(dag) != len(x):
        raise ValueError(f"dag has {len(dag)} nodes but x has {len(x)} entries.")
    if not all(v in (0, 1) for v in x):
        raise ValueError("configurations must be binary 0/1.")
    factors = []
    for i, node in enumerate(dag):
        parents = list(node["parents"])
        if any(p >= i for p in parents):
            bad = [p for p in parents if p >= i][0]
            raise ValueError(f"node {i} has parent {bad} not earlier in the ordering; "
                             "supply a topological order.")
        key = tuple(x[p] for p in parents)
        if key not in node["cpt"]:
            raise ValueError(f"node {i}'s CPT lacks the parent configuration {key}.")
        p1 = float(node["cpt"][key])
        if not 0.0 <= p1 <= 1.0:
            raise ValueError(f"node {i}'s CPT value {p1} is not a probability.")
        factors.append(p1 if x[i] == 1 else 1.0 - p1)
    joint = float(np.prod(factors))
    with np.errstate(divide="ignore"):
        lj = float(np.sum(np.log(factors))) if joint > 0 else float("-inf")
    return RichResult(payload={
        "estimate": joint, "log_joint": lj,
        "factors": [float(v) for v in factors],
        "n_nodes": len(dag),
        "method": "DAG factorisation prod p(x_i | pa_i), binary CPTs"})


def cheatsheet():
    return "wsmdir: prod over nodes of CPT[parent bits]; topological order enforced"
