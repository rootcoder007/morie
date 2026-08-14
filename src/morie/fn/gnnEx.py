# morie.fn -- function file (rootcoder007/morie)
r"""GNNExplainer: which subgraph and which features drove this
prediction.

A trained GNN's prediction for a node depends on its computation graph
-- the :math:`L`-hop neighbourhood -- and on the node features within
it. GNNExplainer asks which *small* part of each actually mattered, and
formulates that as an optimisation rather than a heuristic:

.. math:: \max_{G_S}\ MI\big(Y, (G_S, X_S)\big)
          = H(Y) - H(Y \mid G = G_S, X = X_S).

:math:`H(Y)` is fixed once the model is trained, so maximising the
mutual information is minimising the conditional entropy: find the
subgraph under which the model is *least uncertain* about the
prediction it actually made.

**Why a mask, and why it is continuous.** Searching subgraphs directly
is combinatorial. A mean-field variational relaxation replaces the
discrete choice with a real-valued **graph mask** on the edges and a
**feature mask** on the dimensions, both learned by gradient descent
and squashed through a sigmoid. Size and entropy penalties keep the
explanation small and near-binary -- without them the mask stays diffuse
and explains nothing, which the anchor demonstrates rather than
assumes.

**Two masks, because two things can matter.** An explanation naming
only edges cannot say that the prediction hinged on one feature
dimension; one naming only features cannot say which neighbours
carried it. Both are learned simultaneously.

**Multi-instance explanations.** The same machinery explains a whole
*class* of nodes rather than one, by aligning their computation graphs
to a prototype -- so the output is a motif, not an anecdote.

The evaluation uses synthetic graphs with **planted motifs** that
determine node labels, so there is a ground truth to recover; the
anchor follows the same design.

References
----------
Ying, R., Bourgeois, D., You, J., Zitnik, M. & Leskovec, J. (2019)
"GNNExplainer: Generating Explanations for Graph Neural Networks",
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*,
9240-9251, arXiv:1903.03894. The abstract and Sec. 1 (an explanation
is a small subgraph plus a small subset of node features; the
formulation as maximising mutual information between the prediction
and the distribution of possible subgraph structures; consistent
explanations for an entire class of instances; evaluation on synthetic
graphs with planted motifs). Sec. 4.1 (the MI objective, H(Y) being
constant for a trained GNN so the problem reduces to minimising
conditional entropy, and the mean-field variational approximation
learning a real-valued graph mask). Sec. 4.2 (the feature mask).

Kipf, T. N. & Welling, M. (2017) "Semi-Supervised Classification with
Graph Convolutional Networks", *ICLR 2017*, arXiv:1609.02907. The
model class being explained.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["computation_graph", "mask_objective", "explain_node",
           "conditional_entropy"]

_EPS = 1e-12


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def computation_graph(adj, v, L):
    r"""The :math:`L`-hop neighbourhood -- everything the prediction
    could depend on."""
    seen = {int(v)}
    frontier = {int(v)}
    for _ in range(int(L)):
        nxt = set()
        for u in frontier:
            nxt |= set(adj.get(u, ()))
        frontier = nxt - seen
        seen |= nxt
    edges = [(a, b) for a in sorted(seen)
             for b in sorted(adj.get(a, ())) if b in seen and a < b]
    return {"nodes": sorted(seen), "edges": edges,
            "hops": int(L), "size": len(edges)}


def conditional_entropy(probs):
    r""":math:`H(Y \mid \cdot)` for a predicted distribution."""
    p = [float(v) for v in k.vec(probs)]
    s = sum(p)
    if s <= _EPS:
        raise ValueError("gnnEx: the prediction has no mass")
    p = [v / s for v in p]
    return -sum(v * math.log(max(v, _EPS)) for v in p)


def mask_objective(predict, edges, edge_logits, feature_logits, y,
                   size_coef=0.005, entropy_coef=1.0):
    r"""Minimise :math:`-\log p_\theta(y)` plus size and entropy
    penalties.

    The penalties are what force a *small*, near-binary explanation;
    without them the mask stays diffuse.
    """
    em = [_sig(v) for v in edge_logits]
    fm = [_sig(v) for v in feature_logits]
    p = predict(edges, em, fm)
    loss = -math.log(max(float(p[int(y)]), _EPS))
    size = size_coef * (sum(em) + sum(fm))
    ent = entropy_coef * (
        sum(-(v * math.log(max(v, _EPS))
              + (1 - v) * math.log(max(1 - v, _EPS))) for v in em)
        / max(len(em), 1))
    return {"loss": loss + size + ent, "fit": loss, "size": size,
            "entropy": ent, "edge_mask": em, "feature_mask": fm,
            "prediction": p}


def explain_node(predict, adj, v, y, n_features, L=2, iters=300,
                 lr=0.1, size_coef=0.005, entropy_coef=1.0, seed=0,
                 penalize=True):
    r"""Learn the edge and feature masks by gradient descent.

    ``penalize=False`` drops the size and entropy terms, which leaves
    a diffuse mask -- the comparison the anchor makes.
    """
    cg = computation_graph(adj, v, L)
    edges = cg["edges"]
    if not edges:
        raise ValueError("gnnEx: node %r has an empty computation "
                         "graph" % (v,))
    rng = np.random.default_rng(seed)
    el = [(float(rng.uniform()) - 0.5) * 0.1 for _ in edges]
    fl = [(float(rng.uniform()) - 0.5) * 0.1
          for _ in range(int(n_features))]
    sc = size_coef if penalize else 0.0
    ec = entropy_coef if penalize else 0.0
    h = 1e-4
    hist = []
    for _ in range(int(iters)):
        base = mask_objective(predict, edges, el, fl, y, sc, ec)
        hist.append(base["loss"])
        # every partial derivative is taken against the SAME point:
        # updating in place mid-sweep would mix a stale baseline with
        # already-moved coordinates and descend in the wrong
        # direction.
        ge = []
        for i in range(len(el)):
            up = list(el)
            up[i] += h
            ge.append((mask_objective(predict, edges, up, fl, y, sc,
                                      ec)["loss"] - base["loss"]) / h)
        gf = []
        for i in range(len(fl)):
            up = list(fl)
            up[i] += h
            gf.append((mask_objective(predict, edges, el, up, y, sc,
                                      ec)["loss"] - base["loss"]) / h)
        for i in range(len(el)):
            el[i] -= lr * ge[i]
        for i in range(len(fl)):
            fl[i] -= lr * gf[i]
    final = mask_objective(predict, edges, el, fl, y, sc, ec)
    order = sorted(range(len(edges)),
                   key=lambda i: -final["edge_mask"][i])
    return RichResult(payload={
        "estimate": [edges[i] for i in order],
        "edges_ranked": [(edges[i], final["edge_mask"][i])
                         for i in order],
        "edge_mask": final["edge_mask"],
        "feature_mask": final["feature_mask"],
        "loss_history": hist, "final": final,
        "computation_graph": cg, "penalized": bool(penalize),
        "method": "GNNExplainer; Ying et al. (2019) Sec. 4",
        "note": "maximising MI(Y, (G_S, X_S)) is minimising the "
                "conditional entropy, since H(Y) is fixed once the "
                "model is trained",
    })


def cheatsheet():
    return ("gnnEx: explanation = a SMALL SUBGRAPH plus a SMALL "
            "FEATURE SUBSET, chosen by maximising MI(Y, (G_S, X_S)). "
            "Since H(Y) is fixed for a trained model, that is "
            "MINIMISING CONDITIONAL ENTROPY -- find the subgraph under "
            "which the model is least uncertain. Combinatorial search "
            "is replaced by a mean-field relaxation: continuous edge "
            "and feature masks learned by gradient descent, with size "
            "and entropy penalties WITHOUT WHICH the mask stays "
            "diffuse. Both masks matter: edges alone cannot name a "
            "feature, features alone cannot name a neighbour.")


# compact alias per ledger/NAMING.md
gnnexplainer = explain_node
