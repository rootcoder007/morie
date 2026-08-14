# morie.fn -- function file (rootcoder007/morie)
r"""Latent semantic analysis: retrieval past the literal word.

Term matching fails in two symmetric ways. **Synonymy**: the same idea
is written differently by different people, so a relevant document
misses the query's words entirely. **Polysemy**: the same word means
different things, so an irrelevant document matches. Both are failures
of treating terms as independent, and both are addressed by assuming
there is a latent structure in how terms and documents co-occur.

**The construction.** Build the term-document matrix :math:`X` and take
its singular value decomposition,

.. math:: X = T_0 S_0 D_0^\top ,

then keep the :math:`k` largest singular values -- about 100 in the
paper's experiments -- to get the best rank-:math:`k` least-squares
approximation :math:`\hat X = TSD^\top`.

**The truncation is doing the work, and cutting too little is as bad as
cutting too much.** With :math:`k` at full rank, :math:`\hat X = X`
exactly and LSA reduces to plain term matching -- no generalisation at
all. Small :math:`k` forces terms that co-occur to share a
representation, which is what lets a document be retrieved without
containing the query term. The anchor checks both ends: full rank
reproduces :math:`X` to machine precision, and a small :math:`k`
retrieves a document sharing no term with the query.

**Queries are folded in, not re-decomposed.** A query is treated as a
pseudo-document, :math:`\hat q = q^\top T S^{-1}`, placing it in the
same :math:`k`-dimensional space; documents are then ranked by cosine.
Re-running the SVD per query would be absurd, and folding in is what
makes the method usable -- at the cost that new documents do not
change the space.

**Weighting matters before any of this.** Raw counts let frequent
terms dominate; log-entropy weighting is the standard choice and is
implemented alongside raw and TF-IDF.

References
----------
Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. &
Harshman, R. (1990) "Indexing by Latent Semantic Analysis", *Journal
of the American Society for Information Science* 41(6), 391-407,
doi:10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9.
The abstract and Sec. 1 (synonymy and polysemy as the two failures of
literal term matching; the assumption of an implicit higher-order
semantic structure). Sec. 2 (the SVD of the term-document matrix, the
truncation to about 100 factors, and the resulting best rank-k
approximation), and the treatment of queries as pseudo-documents
placed in the factor space and ranked by cosine.

Dumais, S. T. (1991) "Improving the retrieval of information from
external sources", *Behavior Research Methods, Instruments &
Computers* 23(2), 229-236, doi:10.3758/BF03203370. Log-entropy term
weighting.

Hofmann, T. (1999) "Probabilistic Latent Semantic Analysis", *UAI
1999*, 289-296, arXiv:1301.6705. The probabilistic successor;
implemented in :mod:`plsa`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["term_weighting", "lsa_decompose", "fold_in",
           "cosine_ranking", "reconstruct"]

_EPS = 1e-12
_WEIGHTS = ("raw", "log_entropy", "tfidf")


def term_weighting(X, how="log_entropy"):
    r"""Raw counts, log-entropy, or TF-IDF."""
    if how not in _WEIGHTS:
        raise ValueError("lsa: weighting must be one of %s, got %r"
                         % (", ".join(_WEIGHTS), how))
    A = [[float(v) for v in r] for r in k.mat(X)]
    t, d = len(A), len(A[0])
    if how == "raw":
        return A
    if how == "tfidf":
        out = []
        for i in range(t):
            df = sum(1 for j in range(d) if A[i][j] > 0.0)
            idf = math.log((1.0 + d) / (1.0 + df)) + 1.0
            out.append([A[i][j] * idf for j in range(d)])
        return out
    out = []
    for i in range(t):
        gf = sum(A[i])
        if gf <= _EPS:
            out.append([0.0] * d)
            continue
        ent = 0.0
        for j in range(d):
            p = A[i][j] / gf
            if p > 0.0:
                ent += p * math.log(p)
        g = 1.0 + ent / math.log(d) if d > 1 else 1.0
        out.append([g * math.log(1.0 + A[i][j]) for j in range(d)])
    return out


def lsa_decompose(X, k_dim=None, how="log_entropy"):
    r"""Weight, decompose, truncate to :math:`k` factors."""
    A = term_weighting(X, how)
    T, S, Dt = np.linalg.svd(A, full_matrices=False)
    full = len(S)
    kk = full if k_dim is None else int(k_dim)
    if kk < 1 or kk > full:
        raise ValueError("lsa: k must lie in 1..%d, got %d"
                         % (full, kk))
    return RichResult(payload={
        "estimate": [list(r[:kk]) for r in T], "T": [list(r[:kk])
                                                     for r in T],
        "S": [float(v) for v in S[:kk]],
        "D": [[Dt[q][j] for q in range(kk)] for j in range(len(Dt[0]))],
        "k": kk, "full_rank": full, "weighting": how,
        "method": "truncated SVD of the term-document matrix; "
                  "Deerwester et al. (1990)",
        "note": "k = full rank reproduces X exactly, which is plain "
                "term matching -- the TRUNCATION is what generalises",
    })


def reconstruct(model):
    r""":math:`\hat X = TSD^\top`."""
    T, S, D = model["T"], model["S"], model["D"]
    return [[sum(T[i][q] * S[q] * D[j][q] for q in range(len(S)))
             for j in range(len(D))] for i in range(len(T))]


def fold_in(query, model):
    r""":math:`\hat q = q^\top T S^{-1}` -- a pseudo-document.

    No re-decomposition, which is what makes querying practical; the
    cost is that new documents do not reshape the space.
    """
    q = [float(v) for v in k.vec(query)]
    T, S = model["T"], model["S"]
    if len(q) != len(T):
        raise ValueError("lsa: the query has %d terms but the model "
                         "has %d" % (len(q), len(T)))
    return [sum(q[i] * T[i][f] for i in range(len(q)))
            / max(S[f], _EPS) for f in range(len(S))]


def cosine_ranking(q_hat, model, top_k=5):
    r"""Rank documents by cosine in the :math:`k`-dimensional space."""
    D, S = model["D"], model["S"]
    out = []
    for j in range(len(D)):
        dv = [D[j][f] * S[f] for f in range(len(S))]
        na = math.sqrt(sum(v * v for v in q_hat))
        nb = math.sqrt(sum(v * v for v in dv))
        if na <= _EPS or nb <= _EPS:
            out.append((j, 0.0))
            continue
        out.append((j, sum(q_hat[f] * dv[f]
                           for f in range(len(S))) / (na * nb)))
    out.sort(key=lambda t: -t[1])
    return {"ranking": out[:int(top_k)], "n_documents": len(D)}


def cheatsheet():
    return ("lsa: literal term matching fails through SYNONYMY (the "
            "right document uses other words) and POLYSEMY (the wrong "
            "one shares a word). Take the SVD of the term-document "
            "matrix and keep ~100 factors: the TRUNCATION is the "
            "method, since k = full rank reproduces X exactly and "
            "generalises nothing. Queries are FOLDED IN as "
            "pseudo-documents, q' T S^-1, then ranked by cosine -- no "
            "re-decomposition, but new documents do not reshape the "
            "space. Weight the counts first; log-entropy is standard.")


# compact alias per ledger/NAMING.md
latentsemantic = lsa_decompose
