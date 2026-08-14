# morie.fn -- function file (rootcoder007/morie)
r"""Retrieval for RAG: top-k by inner product, and what it costs.

A parametric model stores what it knows in its weights, which makes
knowledge hard to inspect, hard to update, and impossible to cite.
RAG adds a **non-parametric** memory -- a dense index of passages --
and retrieves from it at generation time, so the source of an answer
is a document you can point at.

**Retrieval is maximum inner product search, and the normalisation
decides which question is being asked.** With unnormalised vectors,
:math:`\arg\max_j q^\top d_j` favours long documents whose embeddings
have large norm; with L2-normalised vectors it is cosine, which asks
about direction alone. ``top_k`` exposes ``metric`` because the two
return different documents, not different orderings of the same ones.

**Exact search is linear in the corpus, and that is the whole
problem.** ``top_k`` is exact and :math:`O(N d)`; ``ivf_search``
partitions the corpus by a coarse quantiser and scans only the
``nprobe`` nearest cells, which is approximate. The honest way to
report that is **recall against the exact answer**, so
``recall_at_k`` computes it rather than leaving "approximate" as an
adjective.

**Two ways to use what comes back, and they differ.** RAG-Sequence
conditions the *whole* output on one retrieved document and
marginalises over documents at the sequence level; RAG-Token lets each
token draw on a different document. The second can compose facts from
several passages in one answer; the first cannot.
``marginalise`` implements both, and the anchor shows them disagreeing
on a constructed case.

References
----------
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal,
N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S. &
Kiela, D. (2020) "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks", *Advances in Neural Information
Processing Systems 33 (NeurIPS 2020)*, 9459-9474, arXiv:2005.11401.
The combination of a parametric memory with a non-parametric dense
vector index of Wikipedia accessed by a pretrained neural retriever;
the two formulations, RAG-Sequence conditioning on the same retrieved
passage for the whole sequence and RAG-Token allowing a different
passage per token, with the output marginalised over the retrieved
documents in both; retrieval of the top-K passages by maximum inner
product search; and the claim that the non-parametric memory can be
replaced or updated without retraining.

Johnson, J., Douze, M. & Jegou, H. (2019) "Billion-scale similarity
search with GPUs", *IEEE Transactions on Big Data* 7(3), 535-547,
doi:10.1109/TBDATA.2019.2921572, arXiv:1702.08734. The inverted-file
index with a coarse quantiser, scanning only a few cells per query,
and the resulting speed/recall trade-off that must be measured rather
than assumed.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S.,
Chen, D. & Yih, W. (2020) "Dense Passage Retrieval for Open-Domain
Question Answering", *EMNLP 2020*, 6769-6781,
doi:10.18653/v1/2020.emnlp-main.550, arXiv:2004.04906. The dual
encoder producing the vectors indexed here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["normalise", "top_k", "ivf_index", "ivf_search",
           "recall_at_k", "marginalise"]

_EPS = 1e-12
_METRICS = ("inner_product", "cosine")


def normalise(v):
    r"""L2 normalisation. Turns inner product into cosine."""
    x = [float(t) for t in k.vec(v)]
    n = math.sqrt(sum(t * t for t in x))
    if n <= _EPS:
        raise ValueError("ragRet: a zero vector has no direction")
    return [t / n for t in x]


def top_k(query, corpus, k_top=5, metric="inner_product"):
    r"""Exact search: :math:`O(Nd)`, and correct by construction.

    ``metric`` matters: an unnormalised inner product rewards large
    norms, which for passage embeddings usually means long passages.
    """
    if metric not in _METRICS:
        raise ValueError("ragRet: metric must be one of %s, got %r"
                         % (", ".join(_METRICS), metric))
    q = [float(v) for v in k.vec(query)]
    D = [[float(v) for v in k.vec(d)] for d in corpus]
    if not D:
        raise ValueError("ragRet: the corpus is empty")
    if any(len(d) != len(q) for d in D):
        raise ValueError("ragRet: a document has a different width "
                         "from the query")
    if metric == "cosine":
        q = normalise(q)
        D = [normalise(d) for d in D]
    s = [sum(q[a] * D[j][a] for a in range(len(q)))
         for j in range(len(D))]
    order = sorted(range(len(D)), key=lambda j: -s[j])
    kk = min(int(k_top), len(order))
    return {"indices": order[:kk], "scores": [s[j] for j in
                                              order[:kk]],
            "all_scores": s, "metric": metric,
            "comparisons": len(D),
            "note": "exact, and linear in the corpus -- which is the "
                    "reason approximate indexes exist"}


def ivf_index(corpus, n_cells=4, iters=25, seed=0):
    r"""Coarse quantiser: assign every vector to its nearest centroid.

    The inverted file is the list of vectors per cell; a query then
    scans a few cells instead of the corpus.
    """
    D = [[float(v) for v in k.vec(d)] for d in corpus]
    n = len(D)
    c = int(n_cells)
    if n < 1 or c < 1:
        raise ValueError("ragRet: need a non-empty corpus and at "
                         "least one cell")
    if c > n:
        raise ValueError("ragRet: %d cells for %d vectors" % (c, n))
    rng = np.random.default_rng(seed)
    cent = [list(D[int(float(rng.uniform()) * n) % n])
            for _ in range(c)]
    assign = [0] * n
    for _ in range(int(iters)):
        for j in range(n):
            assign[j] = min(range(c), key=lambda t: sum(
                (D[j][a] - cent[t][a]) ** 2 for a in range(len(D[j]))))
        for t in range(c):
            mem = [j for j in range(n) if assign[j] == t]
            if mem:
                cent[t] = [sum(D[j][a] for j in mem) / len(mem)
                           for a in range(len(D[0]))]
    lists = {}
    for j in range(n):
        lists.setdefault(assign[j], []).append(j)
    return {"centroids": cent, "lists": lists, "assign": assign,
            "n_cells": c, "n": n,
            "note": "the inverted file: which vectors live in which "
                    "cell"}


def ivf_search(query, corpus, index, k_top=5, nprobe=1,
               metric="inner_product"):
    r"""Scan only the ``nprobe`` nearest cells. APPROXIMATE.

    Reports how many vectors were actually compared, so the saving is
    a number rather than a claim.
    """
    q = [float(v) for v in k.vec(query)]
    cent = index["centroids"]
    order = sorted(range(len(cent)), key=lambda t: sum(
        (q[a] - cent[t][a]) ** 2 for a in range(len(q))))
    probe = order[:max(1, int(nprobe))]
    cand = []
    for t in probe:
        cand.extend(index["lists"].get(t, []))
    if not cand:
        return {"indices": [], "scores": [], "comparisons": 0,
                "probed": probe,
                "note": "the probed cells were empty"}
    sub = [corpus[j] for j in cand]
    r = top_k(q, sub, min(int(k_top), len(sub)), metric)
    return {"indices": [cand[t] for t in r["indices"]],
            "scores": r["scores"], "comparisons": len(cand),
            "probed": probe, "n_cells": index["n_cells"],
            "fraction_scanned": len(cand) / float(index["n"]),
            "note": "approximate: the true nearest neighbour may sit "
                    "in a cell that was not probed"}


def recall_at_k(approximate, exact):
    r"""How much the index actually lost.

    "Approximate" is an adjective; this is the number.
    """
    A = set(int(v) for v in approximate)
    E = list(int(v) for v in exact)
    if not E:
        raise ValueError("ragRet: the exact result is empty")
    hit = sum(1 for j in E if j in A)
    return {"recall": hit / float(len(E)), "hits": hit,
            "k": len(E), "missed": [j for j in E if j not in A]}


def marginalise(doc_scores, token_probs, mode="sequence"):
    r"""Combine the retrieved documents into one output distribution.

    ``sequence``: one document conditions the whole output, and the
    marginal is over sequence likelihoods. ``token``: each position
    may draw on a different document, so facts from several passages
    can be composed in one answer.
    """
    p = [float(v) for v in k.vec(doc_scores)]
    if not p:
        raise ValueError("ragRet: no retrieved documents")
    if any(v < 0.0 for v in p):
        raise ValueError("ragRet: the document scores must be "
                         "non-negative probabilities")
    z = sum(p)
    if z <= _EPS:
        raise ValueError("ragRet: the document weights are all zero")
    w = [v / z for v in p]
    T = [[float(v) for v in k.vec(t)] for t in token_probs]
    if len(T) != len(w):
        raise ValueError("ragRet: %d documents but %d token "
                         "distributions" % (len(w), len(T)))
    if mode == "sequence":
        seq = [math.exp(sum(math.log(max(v, _EPS)) for v in T[d]))
               for d in range(len(w))]
        return RichResult(payload={
            "estimate": sum(w[d] * seq[d] for d in range(len(w))),
            "probability": sum(w[d] * seq[d] for d in range(len(w))),
            "per_document": seq, "weights": w, "mode": "sequence",
            "method": "RAG-Sequence marginalisation; Lewis et al. "
                      "(2020)",
            "note": "ONE document conditions the whole output",
        })
    if mode == "token":
        n_tok = len(T[0])
        if any(len(t) != n_tok for t in T):
            raise ValueError("ragRet: the token distributions differ "
                             "in length")
        per_tok = [sum(w[d] * T[d][t] for d in range(len(w)))
                   for t in range(n_tok)]
        return RichResult(payload={
            "estimate": math.exp(sum(math.log(max(v, _EPS))
                                     for v in per_tok)),
            "probability": math.exp(sum(math.log(max(v, _EPS))
                                        for v in per_tok)),
            "per_token": per_tok, "weights": w, "mode": "token",
            "method": "RAG-Token marginalisation; Lewis et al. "
                      "(2020)",
            "note": "each token may draw on a DIFFERENT document, so "
                    "facts can be composed across passages",
        })
    raise ValueError("ragRet: mode must be sequence or token, got %r"
                     % (mode,))


def cheatsheet():
    return ("ragRet: a parametric model hides what it knows in its "
            "weights -- hard to inspect, update or CITE. RAG adds a "
            "NON-PARAMETRIC index and retrieves at generation time, so "
            "an answer has a document behind it, and the index can be "
            "replaced without retraining. Retrieval is maximum inner "
            "product search; NORMALISE or not decides whether long "
            "passages win on norm alone. Exact search is O(Nd) -- an "
            "IVF index scans only nprobe cells and is APPROXIMATE, so "
            "report RECALL against exact, not the word 'approximate'. "
            "RAG-Sequence conditions the whole output on one document; "
            "RAG-Token lets each token use a different one and can "
            "compose facts across passages.")


# compact alias per ledger/NAMING.md
rag_retrieval = top_k

# public names resolved by fn/_lazy_map.json
ragretrieval = top_k
