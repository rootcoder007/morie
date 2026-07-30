# morie.fn -- function file (rootcoder007/morie)
"""Markov random field -- ESL Sec 17.1-17.3."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_markov_rf"]


def esl_markov_rf(graph, psi=None, states=2, normalize=True):
    r"""Build a pairwise Markov random field and compute its distribution.

    .. math::
        p(x) = \frac{1}{Z}\prod_{C} \psi_C(x_C), \qquad
        Z = \sum_x \prod_C \psi_C(x_C).

    By Hammersley-Clifford a strictly positive distribution factorises over
    the cliques of its graph exactly when it satisfies the graph's Markov
    properties -- so the graph *is* the conditional-independence statement.
    This computes :math:`Z` and the exact marginals by enumeration, which
    costs :math:`O(s^{|V|})` and is therefore capped: past a few dozen states
    the exact answer is unavailable and the function says so rather than
    running for hours.

    Potentials are not probabilities. They need not be normalised, need not
    be less than one, and an individual :math:`\psi_C` has no marginal
    interpretation -- only the product, after dividing by :math:`Z`, does.

    Parameters
    ----------
    graph : array-like or sequence of pairs
        Adjacency matrix ``(V, V)``, or a list of ``(i, j)`` edges.
    psi : dict, optional
        Maps ``(i, j)`` to an ``(states, states)`` potential. Missing edges
        get the Ising-style attractive potential ``exp(+1)`` on agreement.
    states : int
        Number of states per node.
    normalize : bool
        Compute ``Z`` and the normalised marginals.

    Returns
    -------
    RichResult
        ``log_Z``, ``marginals`` ``(V, states)``, ``configurations``,
        ``probabilities``, ``mode``, ``n_edges``.

    References
    ----------
    Hammersley, J. M., & Clifford, P. (1971). Markov fields on finite graphs
        and lattices. Unpublished manuscript.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    A 3-node chain with attractive potentials: by symmetry every node is
    equally likely to be in either state.

    >>> import numpy as np
    >>> r = esl_markov_rf([(0, 1), (1, 2)])
    >>> bool(np.allclose(r["marginals"], 0.5))
    True

    The all-agreeing configurations are the modes, as attraction implies.

    >>> tuple(int(v) for v in r["mode"])
    (0, 0, 0)

    Attraction really does favour agreement over disagreement.

    >>> cfg = [tuple(c) for c in r["configurations"]]
    >>> p = r["probabilities"]
    >>> bool(p[cfg.index((0, 0, 0))] > p[cfg.index((0, 1, 0))])
    True

    Exact enumeration is refused rather than attempted past the cap.

    >>> esl_markov_rf([(i, i + 1) for i in range(30)])
    Traceback (most recent call last):
        ...
    ValueError: exact enumeration needs 2^31 configurations; the cap is 2^22
    """
    G = np.asarray(graph)
    if G.ndim == 2 and G.shape[0] == G.shape[1] and G.shape[0] > 1 and G.dtype != object:
        if set(np.unique(G).tolist()) <= {0, 1} and np.array_equal(G, G.T):
            edges = [(int(i), int(j)) for i in range(G.shape[0])
                     for j in range(i + 1, G.shape[0]) if G[i, j]]
            V = G.shape[0]
        else:
            edges = [(int(a), int(b)) for a, b in G]
            V = int(max(max(e) for e in edges)) + 1
    else:
        edges = [(int(a), int(b)) for a, b in np.asarray(graph).reshape(-1, 2)]
        V = int(max(max(e) for e in edges)) + 1

    s = int(states)
    if s < 2:
        raise ValueError("states must be at least 2")
    total = s**V
    if total > 2**22:
        raise ValueError(
            f"exact enumeration needs {s}^{V} configurations; the cap is 2^22"
        )

    psi = dict(psi or {})
    default = np.where(np.eye(s, dtype=bool), np.exp(1.0), np.exp(-1.0))
    for e in edges:
        if e not in psi and (e[1], e[0]) not in psi:
            psi[e] = default
    for e, P in psi.items():
        if np.shape(P) != (s, s):
            raise ValueError(f"potential for edge {e} has shape {np.shape(P)}, expected ({s}, {s})")

    cfgs = np.array(np.meshgrid(*[np.arange(s)] * V, indexing="ij")).reshape(V, -1).T
    logw = np.zeros(cfgs.shape[0])
    for (i, j), P in psi.items():
        logw += np.log(np.asarray(P, dtype=float)[cfgs[:, i], cfgs[:, j]] + 1e-300)

    mx = logw.max()
    logZ = float(mx + np.log(np.exp(logw - mx).sum()))
    prob = np.exp(logw - logZ) if normalize else np.exp(logw - mx)

    marg = np.zeros((V, s))
    for a in range(s):
        marg[:, a] = ((cfgs == a) * prob[:, None]).sum(axis=0)

    return RichResult(
        title="Markov random field",
        summary_lines=[("nodes", V), ("edges", len(edges)),
                       ("states", s), ("log Z", logZ)],
        payload={
            "log_Z": logZ, "marginals": marg,
            "configurations": cfgs, "probabilities": prob,
            "mode": cfgs[int(np.argmax(prob))],
            "edges": edges, "n_edges": len(edges), "n_nodes": V,
            "method": "esl_markov_rf",
        },
    )


def cheatsheet():
    return "eslmrf: exact Z by enumeration (capped at 2^22); potentials are NOT probabilities"
