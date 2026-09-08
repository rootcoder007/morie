# morie.fn -- function file (rootcoder007/morie)
r"""Neural graph collaborative filtering: the signal is in the graph.

Standard collaborative filtering learns an embedding per user and item
and then scores by an interaction function. The collaborative signal
-- the behavioural similarity encoded in *who consumed what* -- never
enters the embedding itself; it is only present in the objective. NGCF
puts it into the embedding by propagating over the user-item bipartite
graph.

**Message construction, with a term that is not standard.** For a
connected pair,

.. math:: m_{u \leftarrow i} = \frac{1}{\sqrt{|N_u||N_i|}}
          \big(W_1 e_i + W_2 (e_i \odot e_u)\big).

The first term is ordinary graph convolution. The second,
:math:`e_i \odot e_u`, makes the message depend on the **affinity**
between the two embeddings, so more is passed from items similar to
the user. That element-wise term is NGCF's addition, and dropping it
degrades the model to a plain GCN -- the anchor separates the two
rather than trusting the description.

**The Laplacian coefficient is a decay, not a normaliser.**
:math:`p_{ui} = 1/\sqrt{|N_u||N_i|}` can be read two ways: as how much
a historical item contributes to the user's preference, or as a
discount reflecting that messages should weaken with path length.

**Aggregation keeps the node's own signal.**

.. math:: e_u^{(1)} = \mathrm{LeakyReLU}\Big(m_{u\leftarrow u}
          + \sum_{i \in N_u} m_{u \leftarrow i}\Big),

with the self-message retaining the original features.

**Stacking layers is stacking orders of connectivity.** Two layers
capture :math:`u_1 \leftarrow i_2 \leftarrow u_2` -- behavioural
similarity between users. Three capture
:math:`u_1 \leftarrow i_2 \leftarrow u_2 \leftarrow i_4` -- a
recommendation path. The final representation concatenates the
embeddings from all layers, so each order contributes explicitly, and
the trainable weights between layers determine the strength of that
flow.

References
----------
Wang, X., He, X., Wang, M., Feng, F. & Chua, T.-S. (2019) "Neural
Graph Collaborative Filtering", *Proceedings of the 42nd
International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR '19)*, 165-174,
doi:10.1145/3331184.3331267. Sec. 1 (the collaborative signal is
absent from the embedding in conventional CF; the interpretation of
two- and three-layer propagation as behavioural similarity and
potential recommendations). Sec. 2.2.1 (message construction eq. (3)
including the e_i (*) e_u affinity term, the Laplacian coefficient
p_ui = 1/sqrt(|N_u||N_i|) read both as contribution and as a
path-length discount, and message aggregation eq. (4) with the
self-connection). Sec. 2.3 (concatenating the per-layer embeddings).

Kipf, T. N. & Welling, M. (2017) "Semi-Supervised Classification with
Graph Convolutional Networks", *ICLR 2017*, arXiv:1609.02907. The
graph convolution being extended.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X. & Chua, T.-S. (2017)
"Neural Collaborative Filtering", *WWW '17*, 173-182,
doi:10.1145/3038912.3052569. The framework NGCF is measured against;
implemented in :mod:`ncfRS`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["laplacian_coefficient", "message", "propagate",
           "stack_layers", "score"]

_EPS = 1e-12


def _leaky(x, slope=0.2):
    return x if x >= 0.0 else float(slope) * x


def laplacian_coefficient(n_u, n_i):
    r""":math:`p_{ui} = 1/\sqrt{|N_u||N_i|}`."""
    a, b = int(n_u), int(n_i)
    if a < 1 or b < 1:
        raise ValueError("ngcf: both nodes need at least one "
                         "neighbour, got (%d, %d)" % (a, b))
    return 1.0 / math.sqrt(float(a * b))


def message(e_i, e_u, W1, W2, p_ui, affinity=True):
    r"""Eq. (3). ``affinity=False`` drops the :math:`e_i \odot e_u`
    term, leaving a plain graph convolution."""
    ei = [float(v) for v in k.vec(e_i)]
    eu = [float(v) for v in k.vec(e_u)]
    d = len(W1)
    out = []
    for o in range(d):
        s = sum(W1[o][j] * ei[j] for j in range(len(ei)))
        if affinity:
            s += sum(W2[o][j] * ei[j] * eu[j] for j in range(len(ei)))
        out.append(float(p_ui) * s)
    return out


def propagate(E, adjacency, W1, W2, affinity=True, slope=0.2):
    r"""Eq. (4): one embedding propagation layer over the bipartite
    graph."""
    n = len(E)
    deg = [len(adjacency.get(v, [])) for v in range(n)]
    out = []
    for v in range(n):
        nb = adjacency.get(v, [])
        if not nb:
            raise ValueError("ngcf: node %d has no neighbours" % v)
        acc = message(E[v], E[v], W1, W2,
                      laplacian_coefficient(deg[v], deg[v]),
                      affinity)
        for w in nb:
            m = message(E[w], E[v], W1, W2,
                        laplacian_coefficient(deg[v], deg[w]),
                        affinity)
            acc = [acc[o] + m[o] for o in range(len(acc))]
        out.append([_leaky(v_, slope) for v_ in acc])
    return out


def stack_layers(E0, adjacency, Ws, affinity=True, slope=0.2):
    r"""Stack :math:`L` layers and concatenate every order.

    Layer :math:`l` reaches :math:`l`-hop connectivity, so the
    concatenation makes each order's contribution explicit.
    """
    E = [[float(v) for v in r] for r in k.mat(E0)]
    layers = [E]
    for (W1, W2) in Ws:
        E = propagate(E, adjacency, W1, W2, affinity, slope)
        layers.append(E)
    final = [sum((layers[l][v] for l in range(len(layers))), [])
             for v in range(len(E))]
    return RichResult(payload={
        "estimate": final, "final": final, "layers": layers,
        "n_layers": len(Ws), "affinity": bool(affinity),
        "method": "embedding propagation; Wang et al. (2019) eqs. "
                  "(3)-(4) with per-layer concatenation",
        "note": "2 layers reach user-user behavioural similarity, 3 "
                "reach a recommendation path",
    })


def score(final, u, i):
    r"""Inner product of the concatenated representations."""
    a, b = final[int(u)], final[int(i)]
    if len(a) != len(b):
        raise ValueError("ngcf: representations differ in length")
    return sum(a[f] * b[f] for f in range(len(a)))


def cheatsheet():
    return ("ngcf: conventional CF never puts the COLLABORATIVE "
            "SIGNAL into the embedding -- only into the objective. "
            "NGCF propagates over the user-item graph: "
            "m_{u<-i} = p_ui (W1 e_i + W2 (e_i * e_u)), where the "
            "elementwise AFFINITY term is NGCF's addition and dropping "
            "it leaves a plain GCN. p_ui = 1/sqrt(|N_u||N_i|) doubles "
            "as a path-length discount. Two layers reach user-user "
            "similarity, three reach a recommendation path; all "
            "layers are concatenated.")


# compact alias per ledger/NAMING.md
neuralgraphcf = stack_layers

# public names resolved by fn/_lazy_map.json
ngcf = stack_layers
