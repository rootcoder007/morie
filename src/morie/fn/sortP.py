# morie.fn -- function file (rootcoder007/morie)
r"""SortPooling: reading a graph in a consistent order.

Graph classification faces two problems. Extracting useful features
from a graph is the first. The second is stated less often and is
what this layer solves: **how to read the vertices in a meaningful and
consistent order**, so that an ordinary neural network -- which
expects a sequence -- can be trained on graphs at all.

**Why summing is the usual answer, and what it costs.** Summing or
averaging vertex features is permutation-invariant and throws away
which vertex contributed what. SortPooling instead *arranges* the
vertices in a consistent order and outputs a fixed-size sorted
representation, so a traditional convolutional network can read them
in that order and be trained end-to-end on the original graphs, with no
prior conversion to vectors.

**The order comes from the graph, not from the input file.** Vertices
are sorted by the *last* channel of their graph-convolution features,
which act as continuous WL colours: two vertices with the same
structural role get similar values, so the ordering is a function of
the graph rather than of how it happened to be listed. That is exactly
the property the anchor checks -- relabel the vertices and the sorted
output must not move.

**Truncation is part of the design.** The output keeps :math:`k`
vertices: graphs with more are truncated (the lowest-ranked dropped),
graphs with fewer are zero-padded. So every graph yields the same
shape, and :math:`k` is chosen so that a stated percentage of graphs
are not truncated -- a coverage choice rather than a hyperparameter to
tune blindly, and ``choose_k`` computes it.

References
----------
Zhang, M., Cui, Z., Neumann, M. & Chen, Y. (2018) "An End-to-End Deep
Learning Architecture for Graph Classification", *Proceedings of the
Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)*,
4438-4445, doi:10.1609/aaai.v32i1.11782. The two stated challenges --
extracting useful features characterising the rich information encoded
in a graph, and how to sequentially read a graph in a meaningful and
CONSISTENT order; the localized graph convolution and its connection
with two graph kernels; and the SortPooling layer, which instead of
summing up the vertex features arranges them in a consistent order and
outputs a sorted graph representation of fixed size, so that
traditional convolutional networks can read vertices in a consistent
order and be trained end-to-end on original graphs without first
transforming them into vectors.

Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K. &
Borgwardt, K. M. (2011) "Weisfeiler-Lehman Graph Kernels", *Journal of
Machine Learning Research* 12, 2539-2561. The colour refinement the
convolution's channels behave like.

Kipf, T. N. & Welling, M. (2017) "Semi-Supervised Classification with
Graph Convolutional Networks", *ICLR 2017*, arXiv:1609.02907.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sort_pooling", "choose_k", "wl_colours",
           "order_is_graph_determined"]

_EPS = 1e-12


def wl_colours(adj, n, rounds=2, initial=None):
    r"""Continuous colour refinement, the role the last channel plays.

    Each round replaces a vertex's colour by a function of its own and
    its neighbours' -- so vertices with the same structural role
    converge, which is what makes a sort by that value meaningful.
    """
    c = [1.0] * int(n) if initial is None else \
        [float(v) for v in k.vec(initial)]
    if len(c) != int(n):
        raise ValueError("sortP: %d initial colours for %d vertices"
                         % (len(c), n))
    for _ in range(int(rounds)):
        nc = []
        for v in range(int(n)):
            nb = sorted(set(adj.get(v, ())) - {v})
            # SUM, not mean: averaging discards the degree, and two
            # vertices of different degree would then share a colour
            # however many rounds are run.
            nc.append(c[v] + sum(c[u] for u in nb))
        m = sum(nc) / len(nc)
        c = [v / m for v in nc] if m > _EPS else nc
    return c


def sort_pooling(features, k_keep, sort_channel=-1):
    r"""Sort the vertices by one channel, then truncate or pad to
    :math:`k`.

    Sorting on the LAST channel is the paper's choice: it carries the
    most refined colour, so the order reflects structural role.
    """
    X = [[float(v) for v in r] for r in k.mat(features)]
    n, d = len(X), len(X[0])
    kk = int(k_keep)
    if kk < 1:
        raise ValueError("sortP: k must be at least 1")
    ch = int(sort_channel) % d
    order = sorted(range(n), key=lambda i: (-X[i][ch], i))
    kept = order[:kk]
    out = [list(X[i]) for i in kept]
    pad = 0
    while len(out) < kk:
        out.append([0.0] * d)
        pad += 1
    return {"pooled": out, "order": kept, "n_truncated":
            max(0, n - kk), "n_padded": pad, "k": kk,
            "sort_channel": ch,
            "note": "a fixed-size, ordered representation, so an "
                    "ordinary CNN can read it"}


def choose_k(graph_sizes, coverage=0.6):
    r"""Pick :math:`k` so a stated fraction of graphs are not
    truncated.

    A coverage decision, not a blind hyperparameter: :math:`k` is the
    corresponding quantile of the graph-size distribution.
    """
    s = sorted(int(v) for v in graph_sizes)
    c = float(coverage)
    if not 0.0 < c <= 1.0:
        raise ValueError("sortP: the coverage must lie in (0,1], got "
                         "%r" % (coverage,))
    if not s:
        raise ValueError("sortP: no graph sizes given")
    idx = min(len(s) - 1, int(math.ceil(c * len(s))) - 1)
    kk = s[max(idx, 0)]
    return {"k": kk, "coverage": c,
            "fraction_untruncated": sum(1 for v in s if v <= kk)
            / float(len(s)),
            "note": "k is the coverage quantile of the size "
                    "distribution"}


def order_is_graph_determined(features, adj, perm, k_keep,
                              tol=1e-9):
    r"""Relabel the vertices; the sorted output must not move.

    If it does, the ordering came from the input file rather than
    from the graph, and the representation is not well defined.
    """
    X = [[float(v) for v in r] for r in k.mat(features)]
    n = len(X)
    base = sort_pooling(X, k_keep)["pooled"]
    inv = [0] * n
    for i in range(n):
        inv[perm[i]] = i
    Xp = [X[inv[i]] for i in range(n)]
    other = sort_pooling(Xp, k_keep)["pooled"]
    dev = max(abs(base[i][j] - other[i][j])
              for i in range(len(base)) for j in range(len(base[0])))
    return {"max_deviation": dev, "invariant": dev < float(tol),
            "note": "the sort key must be a function of the GRAPH, "
                    "not of the vertex listing"}


def cheatsheet():
    return ("sortP: the under-stated problem in graph classification is "
            "how to read vertices in a MEANINGFUL AND CONSISTENT "
            "order so an ordinary network can be trained on graphs. "
            "Summing is invariant and forgets who contributed what; "
            "SortPooling ARRANGES vertices instead, sorting by the "
            "last convolution channel -- a continuous WL colour, so "
            "the order comes from the GRAPH, not the input file -- "
            "then truncates or pads to a fixed k. Relabel the vertices "
            "and the output must not move. k is chosen for coverage of "
            "the size distribution.")


# compact alias per ledger/NAMING.md
sortpooling = sort_pooling

# public names resolved by fn/_lazy_map.json
sortpool = sort_pooling
