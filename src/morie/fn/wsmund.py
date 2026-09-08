# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Undirected graphical model (clique potentials)."""

import itertools

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_undirected_graph"]


def wasserman_undirected_graph(graph, psi):
    """
    Unnormalised undirected model made exact by brute-force Z.

    Formula: p(x) = (1/Z) prod_C psi_C(x_C). ``graph`` is
    (n_nodes, cliques) with cliques a list of node-index tuples;
    ``psi`` is a same-length list of callables mapping the clique's
    bit-tuple to a positive potential. All 2^n binary configurations
    are enumerated for the exact partition function (n <= 20
    enforced — this is the didactic exact version, not an inference
    engine). The full normalised table is returned.

    Parameters
    ----------
    graph : tuple (n_nodes, cliques)
        Node count and clique index tuples.
    psi : sequence of callables
        Clique potentials, one per clique, > 0.

    Returns
    -------
    result : dict
        Keys: estimate (Z), probabilities (2^n, config index =
        binary number, x_0 the most significant bit), n_nodes,
        n_cliques, method.

    References
    ----------
    Wasserman (2004), Ch 17 (undirected graphs).

    Examples
    --------
    Ising pair with attraction 2 on the edge, unit fields:

    >>> agree = lambda t: 2.0 if t[0] == t[1] else 1.0
    >>> out = wasserman_undirected_graph((2, [(0, 1)]), [agree])
    >>> out["estimate"]
    6.0
    >>> [round(p, 12) for p in out["probabilities"]]
    [0.333333333333, 0.166666666667, 0.166666666667, 0.333333333333]
    >>> round(sum(out["probabilities"]), 12)
    1.0
    """
    n, cliques = graph
    n = int(n)
    if n < 1 or n > 20:
        raise ValueError(f"the exact version handles 1 <= n <= 20 nodes; got {n}.")
    cliques = [tuple(int(v) for v in c) for c in cliques]
    if len(cliques) != len(psi):
        raise ValueError(f"{len(cliques)} cliques but {len(psi)} potentials.")
    for c in cliques:
        if any(not 0 <= v < n for v in c):
            raise ValueError(f"clique {c} references a node outside 0..{n-1}.")
    weights = []
    for config in itertools.product((0, 1), repeat=n):
        w = 1.0
        for c, f in zip(cliques, psi):
            val = float(f(tuple(config[v] for v in c)))
            if val <= 0:
                raise ValueError("clique potentials must be strictly positive.")
            w *= val
        weights.append(w)
    Z = float(np.sum(weights))
    return RichResult(payload={
        "estimate": Z,
        "probabilities": [float(w / Z) for w in weights],
        "n_nodes": n, "n_cliques": len(cliques),
        "method": "exact undirected model; brute-force Z over 2^n configs"})


def cheatsheet():
    return "wsmund: Z = sum over 2^n of prod psi_C; config index binary, x0 MSB"
