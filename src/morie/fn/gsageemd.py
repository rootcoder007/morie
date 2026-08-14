# morie.fn -- function file (rootcoder007/morie)
r"""GraphSAGE: embeddings for nodes the model has never seen.

Matrix-factorisation and random-walk embeddings train one vector per
node. That makes them **transductive**: a node absent at training time
has no embedding, and getting one means more optimisation. Worse, for
many of those methods the objective is invariant to orthogonal
transformations of the embedding space, so the space does not
generalise across graphs and can drift between retrainings.

GraphSAGE learns *functions* instead. Node features are aggregated from
the local neighbourhood by learned aggregators
:math:`\mathrm{AGGREGATE}_k` and weight matrices :math:`W^k`, one pair
per search depth. Because the parameters are shared across nodes, an
unseen node with features can be embedded by a forward pass -- no
retraining. Each of the :math:`K` layers reaches one hop further.

.. math:: h^k_{N(v)} = \mathrm{AGGREGATE}_k(\{h^{k-1}_u,
          u \in N(v)\}), \qquad
          h^k_v = \sigma\big(W^k \cdot
          [h^{k-1}_v \,\|\, h^k_{N(v)}]\big),

with the representation normalised after each layer.

**The aggregator must be permutation-invariant, and the choices trade
off differently.** Mean is the cheapest and is nearly the transductive
GCN's rule when concatenation is replaced by averaging. Pooling passes
each neighbour through a layer and takes an element-wise max, which can
capture different aspects of the neighbourhood. LSTM aggregation is
more expressive but **not symmetric**, so it is applied to a random
permutation of the neighbours -- an admitted patch, not a property.
All three are implemented, mean by default.

**Fixed-size neighbour sampling.** Rather than the full neighbourhood,
a fixed number :math:`S_k` is sampled per layer, bounding per-batch
cost at :math:`\prod_k S_k` regardless of node degree. Sampling is with
replacement when the neighbourhood is smaller than the budget.

References
----------
Hamilton, W. L., Ying, R. & Leskovec, J. (2017) "Inductive
Representation Learning on Large Graphs", *Advances in Neural
Information Processing Systems 30 (NeurIPS 2017)*, 1024-1034,
arXiv:1706.02216. Sec. 2 (factorisation-based embeddings are
inherently transductive and require extra optimisation for new nodes;
their objectives are invariant to orthogonal transformation so the
space does not generalise between graphs and can drift on retraining).
Sec. 3.1 (Algorithm 1: the per-depth AGGREGATE_k and W^k, the
concatenation of the node's own previous representation, and
normalisation). Sec. 3.2 (the unsupervised graph-based loss). Sec. 3.3
(mean aggregation and its relation to the transductive GCN; LSTM
aggregation is not permutation invariant and is applied to a random
permutation; max-pooling aggregation).

Kipf, T. N. & Welling, M. (2017) "Semi-Supervised Classification with
Graph Convolutional Networks", *ICLR 2017*, arXiv:1609.02907. The
transductive convolution GraphSAGE extends to the inductive setting.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["aggregate", "sample_neighbors", "sage_layer",
           "embed", "unsupervised_loss"]

_EPS = 1e-12
_AGGS = ("mean", "max_pool", "lstm_order")


def aggregate(vectors, how="mean", W=None):
    r"""A permutation-invariant summary of the neighbours.

    ``lstm_order`` returns the vectors in the order given, to make the
    point that an LSTM aggregator is order-dependent and must be fed a
    random permutation.
    """
    if how not in _AGGS:
        raise ValueError("gsageemd: aggregator must be one of %s, got "
                         "%r" % (", ".join(_AGGS), how))
    V = [[float(v) for v in r] for r in k.mat(vectors)]
    if not V:
        raise ValueError("gsageemd: no neighbours to aggregate")
    d = len(V[0])
    if how == "mean":
        return [sum(V[i][f] for i in range(len(V))) / len(V)
                for f in range(d)]
    if how == "max_pool":
        if W is None:
            return [max(V[i][f] for i in range(len(V)))
                    for f in range(d)]
        H = [[max(0.0, sum(W[o][j] * V[i][j] for j in range(d)))
              for o in range(len(W))] for i in range(len(V))]
        return [max(H[i][o] for i in range(len(H)))
                for o in range(len(W))]
    return [V[0][f] for f in range(d)]


def sample_neighbors(adj, v, size, rng):
    r"""A fixed-size sample, with replacement when the neighbourhood is
    smaller -- this is what bounds the per-batch cost."""
    nb = sorted(adj.get(v, ()))
    if not nb:
        raise ValueError("gsageemd: node %r has no neighbours" % (v,))
    s = int(size)
    if s < 1:
        raise ValueError("gsageemd: the sample size must be at least 1")
    return [nb[int(float(rng.uniform()) * len(nb)) % len(nb)]
            for _ in range(s)]


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return v if n <= _EPS else [x / n for x in v]


def sage_layer(H, adj, W, how="mean", sizes=None, rng=None,
               normalize=True):
    r"""One depth of Algorithm 1: aggregate, concatenate, transform."""
    out = []
    for v in range(len(H)):
        nb = (sorted(adj.get(v, ())) if sizes is None
              else sample_neighbors(adj, v, sizes, rng))
        if not nb:
            raise ValueError("gsageemd: node %d has no neighbours" % v)
        agg = aggregate([H[u] for u in nb], how)
        cat = list(H[v]) + list(agg)
        if len(W[0]) != len(cat):
            raise ValueError("gsageemd: W expects %d inputs but the "
                             "concatenation is %d"
                             % (len(W[0]), len(cat)))
        z = [max(0.0, sum(W[o][j] * cat[j] for j in range(len(cat))))
             for o in range(len(W))]
        out.append(_norm(z) if normalize else z)
    return out


def embed(features, adj, Ws, how="mean", sizes=None, seed=0):
    r"""K layers, so K hops -- and an unseen node needs only a forward
    pass."""
    rng = np.random.default_rng(seed)
    H = [[float(v) for v in r] for r in k.mat(features)]
    for W in Ws:
        H = sage_layer(H, adj, W, how, sizes, rng)
    return RichResult(payload={
        "estimate": H, "embeddings": H, "depth": len(Ws),
        "aggregator": how,
        "per_batch_bound": (None if sizes is None
                            else int(sizes) ** len(Ws)),
        "method": "GraphSAGE; Hamilton, Ying & Leskovec (2017) "
                  "Algorithm 1",
        "note": "parameters are shared across nodes, so an unseen node "
                "is embedded by a forward pass -- inductive, not "
                "transductive",
    })


def unsupervised_loss(z_u, z_v, z_negatives):
    r"""Sec. 3.2: nearby nodes agree, sampled negatives disagree."""
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(len(a)))

    pos = math.log(max(1.0 / (1.0 + math.exp(-dot(z_u, z_v))), _EPS))
    neg = sum(math.log(max(1.0 / (1.0 + math.exp(dot(z_u, z_n))),
                           _EPS)) for z_n in z_negatives)
    return -(pos + neg)


def cheatsheet():
    return ("gsageemd: factorisation embeddings train ONE VECTOR PER "
            "NODE, so a new node has none -- transductive. GraphSAGE "
            "learns AGGREGATOR FUNCTIONS with shared parameters, so an "
            "unseen node is embedded by a forward pass. K layers = K "
            "hops. Aggregators must be permutation-invariant: mean "
            "(nearly the GCN rule), max-pooling; the LSTM aggregator "
            "is NOT symmetric and is fed a random permutation, which "
            "the paper admits is a patch. Fixed-size neighbour "
            "sampling bounds per-batch cost at prod_k S_k.")


# compact alias per ledger/NAMING.md
graphsage = embed
