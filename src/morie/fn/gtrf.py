# morie.fn -- function file (rootcoder007/morie)
r"""Graph transformer: attention that respects the graph.

The original transformer attends over *all* pairs, which is right for a
sentence -- a line graph where every token may relate to every other --
and wrong for an arbitrary graph, where the edges are the information.
Four changes close that gap.

**Attention is restricted to the neighbourhood.** Each node attends
only to its neighbours, so the attention pattern is a function of
connectivity rather than of position in an arbitrary ordering. Dense
attention would discard the graph entirely.

**Positional encoding by Laplacian eigenvectors.** A graph has no
canonical node order, so the sinusoidal encodings of NLP do not apply.
The eigenvectors of the graph Laplacian
:math:`L = I - D^{-1/2}AD^{-1/2}` generalise them: on a path graph the
Laplacian eigenvectors *are* sinusoids, so the NLP encoding is the
special case. They are added to the input embeddings before the first
layer.

**One caveat that is intrinsic, not a bug.** Eigenvectors are defined
up to sign (and up to rotation within a degenerate eigenspace), so the
encoding is not unique; the sign is randomly flipped during training so
the model cannot rely on it.

**Batch normalisation instead of layer normalisation**, reported as
training faster and generalising better.

**Edge features get their own pipeline.** Bond type or relation type
explicitly modifies the pairwise attention score and is maintained
layer by layer, rather than being folded into the nodes.

References
----------
Dwivedi, V. P. & Bresson, X. (2020) "A Generalization of Transformer
Networks to Graphs", *AAAI Workshop on Deep Learning on Graphs*,
arXiv:2012.09699. The abstract and Sec. 1 (attention as a function of
each node's neighbourhood connectivity; positional encoding by
Laplacian eigenvectors, which naturally generalise the sinusoidal
encodings used in NLP; batch normalisation replacing layer
normalisation for faster training and better generalisation; the
extension to edge features, critical for chemistry bond types and
knowledge-graph relations; and the framing that the original
transformer handles the limited case of line graphs). Sec. 3 (the
LapPE added to the input embeddings, the random sign flipping, and the
attention and feed-forward blocks).

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You
Need", *NIPS 2017*, 5998-6008, arXiv:1706.03762. The architecture
being generalised.

Belkin, M. & Niyogi, M. (2003) "Laplacian Eigenmaps for
Dimensionality Reduction and Data Representation", *Neural
Computation* 15(6), 1373-1396, doi:10.1162/089976603321780317. The
Laplacian eigenvectors used as the encoding.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["laplacian", "laplacian_positional_encoding",
           "sparse_attention", "graph_transformer_layer",
           "random_sign_flip"]

_EPS = 1e-12


def laplacian(adj, n, normalized=True):
    r""":math:`L = I - D^{-1/2}AD^{-1/2}`, or :math:`D - A`."""
    A = [[0.0] * int(n) for _ in range(int(n))]
    for v in adj:
        for w in adj[v]:
            A[int(v)][int(w)] = 1.0
            A[int(w)][int(v)] = 1.0
    d = [sum(r) for r in A]
    L = []
    for i in range(int(n)):
        row = []
        for j in range(int(n)):
            if normalized:
                if d[i] <= _EPS or d[j] <= _EPS:
                    row.append(1.0 if i == j else 0.0)
                else:
                    row.append((1.0 if i == j else 0.0)
                               - A[i][j] / math.sqrt(d[i] * d[j]))
            else:
                row.append((d[i] if i == j else 0.0) - A[i][j])
        L.append(row)
    return L


def laplacian_positional_encoding(adj, n, dim=2, normalized=True):
    r"""The :math:`k` smallest non-trivial Laplacian eigenvectors.

    On a path graph these are sinusoids, which is the sense in which
    the NLP encoding is a special case.
    """
    L = laplacian(adj, n, normalized)
    vals, vecs = np.linalg.eigh(L)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    take = order[1:1 + int(dim)]
    if len(take) < int(dim):
        raise ValueError("gtrf: the graph has only %d non-trivial "
                         "eigenvectors, %d were asked for"
                         % (len(take), int(dim)))
    pe = [[vecs[i][j] for j in take] for i in range(int(n))]
    return {"encoding": pe, "eigenvalues": [vals[j] for j in take],
            "caveat": "eigenvectors are defined up to SIGN, so the "
                      "encoding is not unique -- the sign is flipped "
                      "at random during training"}


def random_sign_flip(pe, rng):
    r"""Flip each eigenvector's sign at random, per the paper."""
    d = len(pe[0])
    s = [1.0 if float(rng.uniform()) < 0.5 else -1.0
         for _ in range(d)]
    return [[pe[i][j] * s[j] for j in range(d)]
            for i in range(len(pe))]


def sparse_attention(H, adj, WQ, WK, WV, edge_bias=None):
    r"""Softmax attention restricted to each node's neighbours.

    Dense attention would throw the graph away; the restriction is
    what makes this a graph model.
    """
    rows = [[float(v) for v in r] for r in k.mat(H)]
    dk = len(WQ)

    def proj(W, x):
        return [sum(W[o][j] * x[j] for j in range(len(x)))
                for o in range(len(W))]

    out = []
    for i in range(len(rows)):
        nb = sorted(adj.get(i, ()))
        if not nb:
            raise ValueError("gtrf: node %d has no neighbours" % i)
        q = proj(WQ, rows[i])
        sc = []
        for j in nb:
            kk = proj(WK, rows[j])
            s = sum(q[a] * kk[a] for a in range(dk)) / math.sqrt(dk)
            if edge_bias is not None:
                s += float(edge_bias.get((i, j),
                                         edge_bias.get((j, i), 0.0)))
            sc.append(s)
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        vs = [proj(WV, rows[j]) for j in nb]
        out.append([sum(w[t] * vs[t][a] for t in range(len(nb)))
                    for a in range(len(vs[0]))])
    return {"output": out, "note": "attention is a function of the "
                                   "NEIGHBOURHOOD, not of an arbitrary "
                                   "node ordering"}


def graph_transformer_layer(H, adj, WQ, WK, WV, W1, W2,
                            edge_bias=None, norm="batch"):
    r"""Attention, residual, feed-forward -- with batch normalisation.

    ``norm="layer"`` is offered for comparison; the paper reports
    batch normalisation trains faster and generalises better.
    """
    if norm not in ("batch", "layer", "none"):
        raise ValueError("gtrf: norm must be batch, layer or none, "
                         "got %r" % (norm,))
    att = sparse_attention(H, adj, WQ, WK, WV, edge_bias)["output"]
    res = [[H[i][f] + att[i][f] for f in range(len(att[0]))]
           for i in range(len(H))]
    res = _normalize(res, norm)
    ff = []
    for i in range(len(res)):
        h1 = [max(0.0, sum(W1[o][j] * res[i][j]
                           for j in range(len(res[i]))))
              for o in range(len(W1))]
        ff.append([sum(W2[o][j] * h1[j] for j in range(len(h1)))
                   for o in range(len(W2))])
    out = [[res[i][f] + ff[i][f] for f in range(len(ff[0]))]
           for i in range(len(res))]
    return _normalize(out, norm)


def _normalize(X, how):
    if how == "none":
        return X
    n, d = len(X), len(X[0])
    if how == "batch":
        out = []
        mu = [sum(X[i][f] for i in range(n)) / n for f in range(d)]
        sd = [math.sqrt(sum((X[i][f] - mu[f]) ** 2
                            for i in range(n)) / n + 1e-5)
              for f in range(d)]
        for i in range(n):
            out.append([(X[i][f] - mu[f]) / sd[f] for f in range(d)])
        return out
    out = []
    for i in range(n):
        mu = sum(X[i]) / d
        sd = math.sqrt(sum((v - mu) ** 2 for v in X[i]) / d + 1e-5)
        out.append([(v - mu) / sd for v in X[i]])
    return out


def cheatsheet():
    return ("gtrf: the transformer attends over ALL pairs, which suits "
            "a sentence (a line graph) and throws away an arbitrary "
            "graph. Four changes: attention restricted to the "
            "NEIGHBOURHOOD; positional encoding by LAPLACIAN "
            "EIGENVECTORS, which are sinusoids on a path graph so NLP "
            "encodings are the special case; batch norm instead of "
            "layer norm; and an explicit edge-feature pipeline that "
            "modifies attention scores. Eigenvectors are defined up to "
            "SIGN, so the sign is randomly flipped in training.")


# compact alias per ledger/NAMING.md
graphtransformer = graph_transformer_layer
