# morie.fn -- function file (rootcoder007/morie)
r"""Graphormer: making a standard Transformer work on graphs.

Transformers dominate language and vision and had *not* been
competitive on graph-level prediction leaderboards. The paper's answer
is that nothing is wrong with the architecture -- what was missing is
**structural information encoded into the model**. Three encodings
supply it, and each fixes a distinct blindness.

**Centrality encoding, because attention only sees semantics.**
Self-attention computes similarities from node *features*, so a hub
and a leaf with identical features are indistinguishable to it, even
though a celebrity is not interchangeable with a typical user. A
learnable vector indexed by **degree** is added to the node features
at the input layer. Simple, and the paper reports it is effective.

**Spatial encoding, because a graph has no canonical grid.** Language
has positions and images have coordinates; nodes lie in a
non-Euclidean space linked by edges. For each *pair* a learnable
scalar indexed by the **shortest-path distance** is added as a bias
inside the softmax:

.. math:: A_{ij} = \frac{(h_iW_Q)(h_jW_K)^\top}{\sqrt{d}} + b_{\phi(i,j)}.

Because it is a bias rather than a mask, distant nodes are *reachable*
but discouraged -- which is what preserves the Transformer's global
receptive field while still telling it what is near.

**Edge encoding, for information that lives on edges.** Bond type
between two atoms is not a property of either atom. The edge features
along the shortest path between :math:`i` and :math:`j` are averaged
against learnable weights and added to the same attention bias.

**Disconnected pairs need their own token.** A pair with no path has
no shortest-path distance; it is assigned a special value rather than
infinity, and the anchor checks that a disconnected graph does not
produce a NaN.

References
----------
Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y. &
Liu, T.-Y. (2021) "Do Transformers Really Perform Bad for Graph
Representation?", *Advances in Neural Information Processing Systems
34 (NeurIPS 2021)*, 28877-28888, arXiv:2106.05234. The abstract and
Sec. 3: the Transformer has not achieved competitive performance on
graph-level prediction leaderboards compared with mainstream GNN
variants; the key insight is the necessity of effectively encoding
STRUCTURAL information into the model; the Centrality Encoding using
degree centrality, a learnable vector per degree added to the node
features at the input layer, because self-attention computes
similarities mainly from node semantic features and does not reflect
node importance; the Spatial Encoding, motivated by the absence of a
canonical grid for graphs, assigning a learnable embedding per node
pair based on the shortest-path distance and encoding it as a bias
term in the softmax attention; and the edge encoding carrying edge
features such as bond type into the Transformer layers.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You
Need", *NIPS 2017*, 5998-6008, arXiv:1706.03762.

Dwivedi, V. P. & Bresson, X. (2020) "A Generalization of Transformer
Networks to Graphs", arXiv:2012.09699. The alternative that restricts
attention to neighbours; implemented in :mod:`gtrf`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["centrality_encoding", "shortest_path_matrix",
           "spatial_bias", "edge_encoding", "graphormer_attention"]

_EPS = 1e-12
UNREACHABLE = -1


def centrality_encoding(adj, n, z_in, z_out=None, directed=False):
    r"""A learnable vector per DEGREE, added to the node features.

    Attention sees semantics only; without this a hub and a leaf with
    the same features are identical to it.
    """
    N = int(n)
    deg_in = [0] * N
    deg_out = [0] * N
    for v in adj:
        for w in adj[v]:
            if v == w:
                continue
            deg_out[int(v)] += 1
            deg_in[int(w)] += 1
    if not directed:
        deg_in = [len(set(adj.get(v, ())) - {v}) for v in range(N)]
        deg_out = list(deg_in)
    out = []
    for v in range(N):
        d = min(deg_in[v], len(z_in) - 1)
        vec = list(z_in[d])
        if directed and z_out is not None:
            o = min(deg_out[v], len(z_out) - 1)
            vec = [vec[i] + z_out[o][i] for i in range(len(vec))]
        out.append(vec)
    return {"encoding": out, "degrees": deg_in,
            "note": "indexed by degree, added at the INPUT layer"}


def shortest_path_matrix(adj, n):
    r"""All-pairs shortest path lengths by BFS.

    A disconnected pair gets ``UNREACHABLE`` rather than infinity, so
    it can be given its own learnable bias instead of a NaN.
    """
    N = int(n)
    D = [[UNREACHABLE] * N for _ in range(N)]
    for s in range(N):
        D[s][s] = 0
        frontier = [s]
        d = 0
        seen = {s}
        while frontier:
            d += 1
            nxt = []
            for v in frontier:
                for w in sorted(set(adj.get(v, ())) - {v}):
                    if w not in seen:
                        seen.add(w)
                        D[s][w] = d
                        nxt.append(w)
            frontier = nxt
    return {"distance": D, "unreachable": UNREACHABLE,
            "n_unreachable": sum(1 for r in D for v in r
                                 if v == UNREACHABLE)}


def spatial_bias(distance, b_table, unreachable_bias=None):
    r"""Turn the distance matrix into the attention bias
    :math:`b_{\phi(i,j)}`.

    A BIAS, not a mask: distant nodes stay reachable, which is what
    keeps the global receptive field a Transformer is for.
    """
    D = [[int(v) for v in r] for r in k.mat(distance)]
    ub = float(unreachable_bias) if unreachable_bias is not None \
        else -10.0
    out = []
    for i in range(len(D)):
        row = []
        for j in range(len(D[0])):
            if D[i][j] == UNREACHABLE:
                row.append(ub)
            else:
                row.append(float(b_table[min(D[i][j],
                                             len(b_table) - 1)]))
        out.append(row)
    return {"bias": out, "unreachable_bias": ub,
            "note": "a bias inside the softmax keeps distant nodes "
                    "reachable but discouraged"}


def edge_encoding(paths, edge_features, w_table):
    r"""Average the edge features along the shortest path.

    Bond type is a property of neither endpoint, so it cannot enter
    through the node features at all.
    """
    out = {}
    for (i, j), path in paths.items():
        if not path:
            out[(i, j)] = 0.0
            continue
        acc = 0.0
        for step, e in enumerate(path):
            f = edge_features.get(e, edge_features.get((e[1], e[0])))
            if f is None:
                raise ValueError("grphmr: no features for edge %r"
                                 % (e,))
            w = w_table[min(step, len(w_table) - 1)]
            fv = [float(q) for q in k.vec(f)]
            wv = [float(q) for q in k.vec(w)]
            acc += sum(fv[a] * wv[a] for a in range(len(fv)))
        out[(i, j)] = acc / len(path)
    return {"edge_bias": out,
            "note": "edge information cannot reach the model through "
                    "node features"}


def graphormer_attention(H, WQ, WK, WV, bias, edge_bias=None):
    r"""Full attention with the structural biases added to the
    logits."""
    X = [[float(v) for v in r] for r in k.mat(H)]
    n, dk = len(X), len(WQ)

    def proj(W, x):
        return [sum(W[o][j] * x[j] for j in range(len(x)))
                for o in range(len(W))]

    out, weights = [], []
    for i in range(n):
        q = proj(WQ, X[i])
        sc = []
        for j in range(n):
            kk = proj(WK, X[j])
            s = sum(q[a] * kk[a] for a in range(dk)) / math.sqrt(dk)
            s += float(bias[i][j])
            if edge_bias is not None:
                s += float(edge_bias.get((i, j), 0.0))
            sc.append(s)
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        weights.append(w)
        vs = [proj(WV, X[j]) for j in range(n)]
        out.append([sum(w[j] * vs[j][a] for j in range(n))
                    for a in range(len(vs[0]))])
    return RichResult(payload={
        "estimate": out, "output": out, "weights": weights,
        "method": "Graphormer attention with centrality, spatial and "
                  "edge encodings; Ying et al. (2021)",
        "note": "the architecture is a STANDARD Transformer; the "
                "structural encodings are what was missing",
    })


def cheatsheet():
    return ("grphmr: a standard Transformer was NOT competitive on "
            "graph leaderboards, and the missing piece is STRUCTURAL "
            "ENCODING, not architecture. Three of them: CENTRALITY "
            "(a learnable vector per degree added to node features, "
            "because attention sees only semantics and cannot tell a "
            "hub from a leaf); SPATIAL (a learnable bias per "
            "shortest-path distance inside the softmax, since a graph "
            "has no canonical grid -- a BIAS, so distant nodes stay "
            "reachable); and EDGE (features along the path, since bond "
            "type belongs to neither endpoint). Disconnected pairs get "
            "their own token, not infinity.")


# compact alias per ledger/NAMING.md
graphormer = graphormer_attention
