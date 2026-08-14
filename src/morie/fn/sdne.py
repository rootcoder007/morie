# morie.fn -- function file (rootcoder007/morie)
r"""SDNE: first- and second-order proximity, jointly.

Network embedding had used shallow models -- IsoMap, Laplacian
Eigenmaps, LINE -- whose representational capacity cannot capture a
highly non-linear network structure. SDNE is a deep model, and its
argument is about **which** structure to preserve.

**Two proximities, doing different jobs.**

* **First-order** proximity is the pairwise similarity between
  vertices that are *linked*. It is local, and in a real network it is
  desperately sparse: many legitimate links are simply missing, so it
  cannot describe the structure on its own.
* **Second-order** proximity is the similarity of two vertices'
  *neighbourhoods*. Two vertices need not be linked to be similar --
  which is exactly what rescues the sparse case.

They enter through different halves of a semi-supervised architecture:
an unsupervised autoencoder reconstructs the adjacency row (second
order, global), and a supervised Laplacian-eigenmaps term pulls linked
vertices together (first order, local).

**The reconstruction penalty must be re-weighted, and this is the part
that is easy to get wrong.** The adjacency row is mostly zeros, so a
plain squared error is minimised by predicting zero everywhere -- a
perfect score for a useless embedding. SDNE imposes **more penalty on
the non-zero entries**: :math:`\|(\hat X - X)\odot B\|^2` with
:math:`b_{ij} = \beta > 1` where an edge exists and 1 where it does
not. ``second_order_loss`` computes both, and the anchor shows the
all-zero reconstruction winning at :math:`\beta = 1` and losing at
:math:`\beta > 1`.

References
----------
Wang, D., Cui, P. & Zhu, W. (2016) "Structural Deep Network
Embedding", *Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD '16)*,
1225-1234, doi:10.1145/2939672.2939753. Sec. 1 and 3: that shallow
models such as IsoMap, Laplacian Eigenmaps and LINE have limited
representation ability and cannot capture the highly non-linear
network structure; the semi-supervised deep model exploiting
first-order and second-order proximity JOINTLY, with the unsupervised
component reconstructing the second-order proximity to preserve the
GLOBAL structure and the supervised component using first-order
proximity to preserve the LOCAL structure; that first-order proximity
is the local pairwise similarity only between linked vertices and is
insufficient because network sparsity means many legitimate links are
missing; and the re-weighted reconstruction imposing more penalty on
the reconstruction error of non-zero elements than on zero elements.

Belkin, M. & Niyogi, P. (2003) "Laplacian Eigenmaps for
Dimensionality Reduction and Data Representation", *Neural
Computation* 15(6), 1373-1396, doi:10.1162/089976603321780317. The
first-order term.

Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J. & Mei, Q. (2015)
"LINE: Large-scale Information Network Embedding", *WWW 2015*,
1067-1077, doi:10.1145/2736277.2741093, arXiv:1503.03578. The
shallow model whose two proximities this deepens.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["penalty_matrix", "second_order_loss",
           "first_order_loss", "proximity_counts", "sdne_loss"]

_EPS = 1e-12


def penalty_matrix(adjacency, beta=5.0):
    r""":math:`b_{ij} = \beta` on edges, 1 elsewhere.

    Without :math:`\beta > 1` the all-zero reconstruction wins.
    """
    A = [[float(v) for v in r] for r in k.mat(adjacency)]
    b = float(beta)
    if b < 1.0:
        raise ValueError("sdne: beta must be at least 1; below it the "
                         "zeros would be weighted MORE than the "
                         "edges")
    return {"B": [[b if v != 0.0 else 1.0 for v in r] for r in A],
            "beta": b,
            "n_nonzero": sum(1 for r in A for v in r if v != 0.0),
            "n_zero": sum(1 for r in A for v in r if v == 0.0)}


def second_order_loss(adjacency, reconstruction, beta=5.0):
    r""":math:`\|(\hat X - X)\odot B\|_F^2` -- the GLOBAL structure.

    The neighbourhood row is what is reconstructed, so two unlinked
    vertices with similar neighbours end up close.
    """
    X = [[float(v) for v in r] for r in k.mat(adjacency)]
    H = [[float(v) for v in r] for r in k.mat(reconstruction)]
    if len(X) != len(H) or len(X[0]) != len(H[0]):
        raise ValueError("sdne: the adjacency is %dx%d but the "
                         "reconstruction is %dx%d"
                         % (len(X), len(X[0]), len(H), len(H[0])))
    B = penalty_matrix(X, beta)["B"]
    weighted = sum(((H[i][j] - X[i][j]) * B[i][j]) ** 2
                   for i in range(len(X)) for j in range(len(X[0])))
    plain = sum((H[i][j] - X[i][j]) ** 2
                for i in range(len(X)) for j in range(len(X[0])))
    return {"loss": weighted, "unweighted": plain, "beta": float(beta),
            "note": "the row is mostly zeros, so the unweighted loss "
                    "rewards predicting nothing"}


def first_order_loss(adjacency, embeddings):
    r""":math:`\sum_{ij} s_{ij}\|y_i - y_j\|^2` -- the LOCAL structure.

    The Laplacian-eigenmaps term: linked vertices are pulled together,
    and it is exactly zero when every linked pair coincides.
    """
    S = [[float(v) for v in r] for r in k.mat(adjacency)]
    Y = [[float(v) for v in r] for r in k.mat(embeddings)]
    n = len(S)
    if len(Y) != n:
        raise ValueError("sdne: %d vertices but %d embeddings"
                         % (n, len(Y)))
    tot, pairs = 0.0, 0
    for i in range(n):
        for j in range(n):
            if S[i][j] == 0.0 or i == j:
                continue
            pairs += 1
            tot += S[i][j] * sum((Y[i][a] - Y[j][a]) ** 2
                                 for a in range(len(Y[0])))
    return {"loss": tot, "linked_pairs": pairs,
            "note": "zero iff every LINKED pair shares an embedding"}


def proximity_counts(adjacency):
    r"""How many pairs each proximity actually covers.

    The paper's motivation, measurable: second-order pairs vastly
    outnumber first-order ones in a sparse network, which is why the
    first order alone cannot describe the structure.
    """
    A = [[float(v) for v in r] for r in k.mat(adjacency)]
    n = len(A)
    first = second = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != 0.0:
                first += 1
            shared = sum(1 for t in range(n)
                         if A[i][t] != 0.0 and A[j][t] != 0.0)
            if shared > 0:
                second += 1
    total = n * (n - 1) // 2
    return {"first_order_pairs": first, "second_order_pairs": second,
            "total_pairs": total,
            "density": first / float(total) if total else 0.0,
            "ratio": second / float(first) if first else float("inf"),
            "note": "many legitimate links are missing, so the "
                    "first-order set is far the smaller"}


def sdne_loss(adjacency, reconstruction, embeddings, beta=5.0,
              alpha=0.1, nu=0.0, parameters=None):
    r"""The joint semi-supervised objective."""
    s2 = second_order_loss(adjacency, reconstruction, beta)
    s1 = first_order_loss(adjacency, embeddings)
    reg = 0.0
    if parameters is not None:
        reg = float(nu) * sum(float(v) ** 2
                              for r in k.mat(parameters) for v in r)
    total = s2["loss"] + float(alpha) * s1["loss"] + reg
    return RichResult(payload={
        "estimate": total, "loss": total,
        "second_order": s2["loss"], "first_order": s1["loss"],
        "regulariser": reg, "alpha": float(alpha),
        "beta": float(beta),
        "method": "SDNE joint objective; Wang, Cui & Zhu (2016)",
        "note": "unsupervised autoencoder for the GLOBAL structure, "
                "supervised Laplacian term for the LOCAL one",
    })


def cheatsheet():
    return ("sdne: shallow embeddings (IsoMap, Laplacian Eigenmaps, "
            "LINE) cannot capture a highly NON-LINEAR network, so go "
            "deep -- and preserve TWO proximities jointly. FIRST-order "
            "is the local similarity between LINKED vertices, and in a "
            "sparse network most legitimate links are missing, so it "
            "is not enough. SECOND-order is the similarity of "
            "NEIGHBOURHOODS, which needs no edge between the pair. An "
            "autoencoder reconstructs the adjacency row (global) and a "
            "Laplacian term pulls linked vertices together (local). "
            "The reconstruction MUST re-weight: with B = 1 the "
            "all-zero output wins, so put beta > 1 on the edges.")


# compact alias per ledger/NAMING.md
structuraldeepnetwork = sdne_loss

# public names resolved by fn/_lazy_map.json
sdne = sdne_loss
