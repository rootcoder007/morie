# morie.fn -- function file (rootcoder007/morie)
"""ColBERT late-interaction retrieval scoring."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .clipsi import l2_normalize

__all__ = ["colbert"]


def colbert(query, docs):
    """
    ColBERT late interaction

    Formula: sum_q max_d q_i . d_j

    Every query token keeps its own vector and is scored against its
    best-matching document token; the sum of those maxima is the
    document score.  Because the interaction happens after both sides
    are encoded, document vectors can be indexed offline -- that is the
    "late" in late interaction.  With unit-normalised vectors a document
    containing the query exactly scores n_q, the largest value possible.

    Parameters
    ----------
    query : array-like
        nq x d matrix of query token embeddings.
    docs : array-like
        List of documents, each an nd x d matrix of token embeddings.

    Returns
    -------
    result : dict
        Keys: estimate (best score), scores, ranking, best, max_sim,
        nq, n_docs.

    References
    ----------
    Khattab & Zaharia (2020), ColBERT: Efficient and Effective Passage
    Search via Contextualized Late Interaction over BERT, SIGIR
    2020:39-48.
    """
    Q = core.mat(query)
    nq = len(Q)
    if nq == 0:
        raise ValueError("empty input: query has no tokens")
    d = len(Q[0])
    Qn = [l2_normalize(r) for r in Q]
    if docs is None:
        raise ValueError("docs must hold at least one document")
    dl = list(docs)
    if not dl:
        raise ValueError("docs must hold at least one document")
    scores, maxsim = [], []
    for doc in dl:
        D = core.mat(doc)
        if not D:
            raise ValueError("a document has no tokens")
        if len(D[0]) != d:
            raise ValueError("query and document dimensions disagree")
        Dn = [l2_normalize(r) for r in D]
        row = []
        s = 0.0
        for i in range(nq):
            best = None
            for j in range(len(Dn)):
                v = sum(Qn[i][k] * Dn[j][k] for k in range(d))
                if best is None or v > best:
                    best = v
            row.append(best)
            s += best
        maxsim.append(row)
        scores.append(s)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return RichResult(payload={
        "estimate": scores[order[0]],
        "scores": scores,
        "ranking": order,
        "best": order[0],
        "max_sim": maxsim,
        "nq": nq,
        "n_docs": len(dl),
        "method": "ColBERT late-interaction retrieval scoring",
    })


def cheatsheet():
    return "colbrt: ColBERT late-interaction retrieval scoring"
